# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Delta storage for the embed model in DualModelRunner.

After both runners finish their normal load, ``apply_embed_delta_storage`` walks
the embed model's transformer-block parameters and replaces each shape-matched
weight tensor's underlying storage with int8 quantized-delta codes plus a per-row
fp32 scale. The original bf16 storage is freed by Python's GC. Forward
pre-hooks materialize a bf16 reconstruction (``W_decode + dequant(Δ)``) into the
weight's ``.data`` slot just before each forward, and matching post-hooks
restore the int8 placeholder so resident memory drops back down.

This is invasive but contained: the embed runner runs eager (no torch.compile /
CUDAGraphs), so dynamic ``.data`` reassignment is safe. The decode runner is
unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.logger import init_logger

logger = init_logger(__name__)


def _norm_name(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _expand_fused(canon: str) -> list[str]:
    """Map a vLLM-fused param name to its HF unfused sub-names."""
    if canon.endswith("self_attn.qkv_proj.weight"):
        prefix = canon[: -len("self_attn.qkv_proj.weight")]
        return [
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
        ]
    if canon.endswith("mlp.gate_up_proj.weight"):
        prefix = canon[: -len("mlp.gate_up_proj.weight")]
        return [
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
        ]
    return [canon]


class _DeltaWeightInfo:
    """Per-parameter delta metadata kept as attributes on the Parameter."""

    __slots__ = (
        "decode_ref",
        "q_list",
        "scale_list",
        "is_fused",
        "orig_dtype",
        "orig_shape",
        "backup_data",
    )

    def __init__(
        self,
        decode_ref: torch.Tensor,
        q_list: list[torch.Tensor],
        scale_list: list[torch.Tensor],
        is_fused: bool,
        orig_dtype: torch.dtype,
        orig_shape: tuple[int, ...],
    ) -> None:
        self.decode_ref = decode_ref
        self.q_list = q_list
        self.scale_list = scale_list
        self.is_fused = is_fused
        self.orig_dtype = orig_dtype
        self.orig_shape = orig_shape
        self.backup_data: torch.Tensor | None = None


def _materialize(info: _DeltaWeightInfo) -> torch.Tensor:
    """Reconstruct the bf16 weight from decode + dequantized Δ."""
    d = info.decode_ref.to(torch.float32)
    if info.is_fused:
        chunks = [
            q.to(torch.float32) * s for q, s in zip(info.q_list, info.scale_list)
        ]
        delta = torch.cat(chunks, dim=0)
    else:
        delta = info.q_list[0].to(torch.float32) * info.scale_list[0]
    return (d + delta).to(info.orig_dtype)


def _make_pre_hook():
    """Per-Linear pre-hook: materializes bf16 weight from int8 storage. Used
    only on the slow path; the fast path replaces forward via _patch_forward."""

    def hook(module, inputs):  # noqa: ANN001
        w = getattr(module, "weight", None)
        if w is None:
            return
        info: _DeltaWeightInfo | None = getattr(w, "_delta_info", None)
        if info is None:
            return
        if info.backup_data is not None:
            return
        full = _materialize(info)
        info.backup_data = w.data
        w.data = full

    return hook


def _make_post_hook():
    def hook(module, inputs, output):  # noqa: ANN001
        w = getattr(module, "weight", None)
        if w is None:
            return
        info: _DeltaWeightInfo | None = getattr(w, "_delta_info", None)
        if info is None or info.backup_data is None:
            return
        w.data = info.backup_data
        info.backup_data = None

    return hook


def _delta_linear_forward(self, x):
    """Fast-path forward for embed Linear with delta storage.

    Computes  out = x @ W_decode^T + dequant_int8_per_row_GEMM(x, q, scale)
    using torch._weight_int8pack_mm for the delta term so we never
    materialize a bf16 weight buffer for Δ. Memory peak during the call is
    just the output tensor (no weight scratch).
    """
    info: _DeltaWeightInfo | None = getattr(self.weight, "_delta_info", None)
    if info is None:
        return self._delta_orig_forward(x)

    # Materialize bf16 weight in a transient scratch, run a single bf16
    # F.linear, free the scratch. Empirically this gives lower peak memory
    # than fused decode + int8pack_mm (which leaves two [M, N] outputs and
    # internal kernel scratch alive simultaneously). Per-forward peak is
    # one bf16 weight + one [M, N] output, just like baseline.
    if info.is_fused:
        chunks = [
            q.to(torch.float32) * s
            for q, s in zip(info.q_list, info.scale_list)
        ]
        delta = torch.cat(chunks, dim=0)
        del chunks
    else:
        delta = info.q_list[0].to(torch.float32) * info.scale_list[0]
    full = (info.decode_ref.to(torch.float32) + delta).to(info.orig_dtype)
    del delta

    skip_bias_add = getattr(self, "skip_bias_add", False)
    bias = self.bias if (self.bias is not None and not skip_bias_add) else None
    out = torch.nn.functional.linear(x, full, bias)
    del full

    if getattr(self, "return_bias", False):
        output_bias = self.bias if skip_bias_add else None
        return out, output_bias
    return out


def _patch_forward(module: nn.Module) -> None:
    """Install the fast-path forward on a Linear module, preserving the
    original for fallback."""
    if not hasattr(module, "_delta_orig_forward"):
        module._delta_orig_forward = module.forward
    module.forward = _delta_linear_forward.__get__(module, type(module))


def _attach_delta_to_param(
    param: nn.Parameter,
    info: _DeltaWeightInfo,
    int8_storage: torch.Tensor,
) -> int:
    """Replace param.data with int8 storage and stash delta metadata. Returns
    bytes freed (positive) by the storage swap."""
    old_bytes = param.element_size() * param.numel()
    param._delta_info = info  # type: ignore[attr-defined]
    # nn.Parameter.data refuses to accept int dtype when requires_grad=True.
    # Inference weights don't need gradients; disable so we can rebind storage.
    param.requires_grad_(False)
    param.data = int8_storage
    new_bytes = int8_storage.element_size() * int8_storage.numel()
    return max(0, old_bytes - new_bytes)


def apply_embed_delta_storage(
    embed_model: nn.Module,
    decode_model: nn.Module,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Convert the embed model into delta storage. Returns counters."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())

    # Index decode weights by canonical name.
    decode_w = {_norm_name(n): p for n, p in decode_model.named_parameters()}

    # Load Δ bundle to GPU, indexed by canonical sub-name.
    device = next(embed_model.parameters()).device
    handle = safe_open(
        str(bundle_dir / "delta.safetensors"), framework="pt", device="cpu"
    )
    q_w: dict[str, torch.Tensor] = {}
    s_w: dict[str, torch.Tensor] = {}
    for canon, info in manifest["tensors"].items():
        if info["kind"] == "delta":
            q_w[canon] = handle.get_tensor(canon + ".q").to(device)
            s_w[canon] = handle.get_tensor(canon + ".scale").to(device)

    # Build a name -> module map so we can register hooks on the owner module
    # of each replaced param.
    name_to_module: dict[str, nn.Module] = {}
    for module_name, module in embed_model.named_modules():
        for child_name, _child in module.named_parameters(recurse=False):
            full = f"{module_name}.{child_name}" if module_name else child_name
            name_to_module[full] = module

    n_replaced = 0
    n_skipped = 0
    bytes_freed = 0
    affected_modules: set[int] = set()
    skipped_names: list[str] = []

    for name, param in list(embed_model.named_parameters()):
        canon = _norm_name(name)
        subs = _expand_fused(canon)

        if param.dim() < 2:
            # Skip 1-D params (norms). They're tiny (KBs), the bf16 cost is
            # negligible, and the int8-pack matmul fast-path only applies to
            # 2-D Linear weights.
            n_skipped += 1
            skipped_names.append(name)
            continue

        if len(subs) == 1:
            s = subs[0]
            if s not in decode_w or s not in q_w:
                n_skipped += 1
                skipped_names.append(name)
                continue
            decode_ref = decode_w[s]
            if decode_ref.shape != param.shape:
                n_skipped += 1
                skipped_names.append(name)
                continue
            int8 = q_w[s]
            scale = s_w[s]
            info = _DeltaWeightInfo(
                decode_ref=decode_ref,
                q_list=[int8],
                scale_list=[scale],
                is_fused=False,
                orig_dtype=param.dtype,
                orig_shape=tuple(param.shape),
            )
            bytes_freed += _attach_delta_to_param(param, info, int8)
            n_replaced += 1
            mod = name_to_module.get(name)
            if mod is not None:
                affected_modules.add(id(mod))
        else:
            if canon not in decode_w:
                n_skipped += 1
                skipped_names.append(name)
                continue
            if not all(s in q_w for s in subs):
                n_skipped += 1
                skipped_names.append(name)
                continue
            decode_ref = decode_w[canon]
            if decode_ref.shape != param.shape:
                n_skipped += 1
                skipped_names.append(name)
                continue
            q_list = [q_w[s] for s in subs]
            s_list = [s_w[s] for s in subs]
            # Pack int8 codes back-to-back so resident storage is small.
            concat_q = torch.cat(q_list, dim=0)
            info = _DeltaWeightInfo(
                decode_ref=decode_ref,
                q_list=q_list,
                scale_list=s_list,
                is_fused=True,
                orig_dtype=param.dtype,
                orig_shape=tuple(param.shape),
            )
            bytes_freed += _attach_delta_to_param(param, info, concat_q)
            n_replaced += 1
            mod = name_to_module.get(name)
            if mod is not None:
                affected_modules.add(id(mod))

    # Patch each affected Linear's forward to use the fast-path (decode +
    # int8pack_mm). This avoids materializing a bf16 weight buffer entirely;
    # memory peak during forward is just the output tensor.
    n_hooked = 0
    for module_name, module in embed_model.named_modules():
        if id(module) in affected_modules:
            _patch_forward(module)
            n_hooked += 1

    # Empty cache so the freed bf16 storage is actually released back to the
    # CUDA caching allocator (otherwise the saving is invisible to torch peak).
    torch.cuda.empty_cache()
    # CRUCIAL: reset peak memory stats. vLLM's KV-cache profiler computes
    # available memory as (gpu_total * util) - max_memory_allocated, which is
    # MONOTONIC across the session. Without this reset, the high-water mark
    # captured the bf16 weights we just freed, and KV pool stays the same.
    torch.cuda.reset_peak_memory_stats()

    summary = {
        "n_replaced": n_replaced,
        "n_skipped": n_skipped,
        "n_hooked_modules": n_hooked,
        "bytes_freed_GB": bytes_freed / 1e9,
        "first_skipped": skipped_names[:6],
        "scheme": manifest["scheme"],
        "device": str(device),
    }
    logger.info("embed_delta_storage applied: %s", summary)
    return summary
