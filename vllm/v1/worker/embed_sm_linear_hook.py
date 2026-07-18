"""Route embed-model prefill linears through cuBLASLt with
CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET pinned, ONLY on the embed stream.

Motivation (Track K, 2026-07-16): with decode (small-M, memory-bound GEMMs /
FA) and embed (large-M, tensor-bound GEMMs) co-located on two streams, capping
the embed GEMM to ~90 of 132 SMs leaves the memory-bound decode side a
dedicated minority SM share. Measured on isolated GEMM pairs this raises the
co-location win from ~5% to +6..14.7% vs sequential (+3..11.6% vs unsplit
two-stream). This hook applies the same cap inside the real dual-model engine.

Mirror of decode_sm_linear_hook, with the opposite routing condition: we route
LARGE-M (prefill-shaped, M >= HB_EMBED_SM_TARGET_MIN_M, default 256) GEMMs on
the embed stream; decode and small-M work is untouched.

Activation: env ``HB_EMBED_SM_COUNT_TARGET`` (int SM count, e.g. 90); unset/0
disables. ``register_embed_stream`` is called by dual_model_runner once the
embed stream exists.

CUDA-graph note: the cuBLASLt wrapper caches (shape, target) plans, so the
heuristic query happens during vLLM's pre-capture dummy/warmup run; inside
capture only cublasLtMatmul launches, which records into the graph.
"""
from __future__ import annotations

import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SM_TARGET: int = 0
_MIN_M: int = 256
_PATCHED: bool = False
_LT = None  # lazily constructed CublasLtMatmul
# Runtime override used by the phase-aware Slack Serve scheduler.  ``None``
# means use the target baked into the compiled custom-op call; ``0`` selects
# an ordinary, uncapped cuBLASLt plan.  Large embed batches execute outside a
# CUDA graph in vLLM's PIECEWISE mode, so this is read at every real dispatch.
# A captured graph still freezes the value observed during capture, which is
# fine for the <= capture-size tail where the large-M hook is normally idle.
_RUNTIME_TARGET: int | None = None


def set_runtime_sm_target(target: int | None) -> None:
    """Select the SM target for subsequent eager embed GEMM dispatches.

    ``None`` restores the configured target, while ``0`` selects an uncapped
    cuBLASLt algorithm.  Callers must switch only at embed-step boundaries;
    this does not alter kernels that have already been launched.
    """
    global _RUNTIME_TARGET
    if target is not None and target < 0:
        raise ValueError(f"runtime SM target must be >= 0, got {target}")
    _RUNTIME_TARGET = target


def get_runtime_sm_target() -> int | None:
    return _RUNTIME_TARGET


def _load_lt():
    global _LT
    if _LT is None:
        lt_dir = os.environ.get(
            "HB_CUBLASLT_DIR",
            "/mnt/weka/theo/heterogenious-batching-vllm/scratch/cublas_sm_target",
        )
        if lt_dir not in sys.path:
            sys.path.insert(0, lt_dir)
        from cublaslt import CublasLtMatmul  # type: ignore
        _LT = CublasLtMatmul()
    return _LT


# Custom op wrapper: the model forward is dynamo-traced (compilation mode 3),
# and a raw ctypes call would graph-break / fail fullgraph tracing. A custom
# op is opaque to dynamo and records normally under CUDA-graph capture.
@torch.library.custom_op("hb::embed_sm_capped_linear", mutates_args=())
def _embed_sm_capped_linear(
    x2: torch.Tensor, w: torch.Tensor, sm_target: int
) -> torch.Tensor:
    out = torch.empty((x2.shape[0], w.shape[0]), dtype=x2.dtype,
                      device=x2.device)
    target = sm_target if _RUNTIME_TARGET is None else _RUNTIME_TARGET
    _load_lt().linear(x2, w, out, target)
    return out


@_embed_sm_capped_linear.register_fake
def _embed_sm_capped_linear_fake(
    x2: torch.Tensor, w: torch.Tensor, sm_target: int
) -> torch.Tensor:
    return x2.new_empty((x2.shape[0], w.shape[0]))


def register_embed_model(model: torch.nn.Module) -> None:
    """Mark the embed model's linear modules and install the patch.

    Module-attribute gating (not stream gating): under CUDA-graph capture the
    current stream is the capture stream, so a stream gate would silently
    capture the default path (or misroute decode's own large-M prefill GEMMs).
    Marking the embed model's modules identifies them unambiguously in eager,
    dynamo-traced, and capture paths. No-op unless
    HB_EMBED_SM_COUNT_TARGET > 0.
    """
    global _SM_TARGET, _MIN_M, _PATCHED
    target = int(os.environ.get("HB_EMBED_SM_COUNT_TARGET", "0") or "0")
    if target <= 0:
        return
    n_marked = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_cuda:
            # Plain module attribute: dynamo treats it as a guard/constant
            # (a data_ptr() check would raise on FakeTensor during tracing).
            mod._hb_embed_sm_capped = True
            n_marked += 1
    _SM_TARGET = target
    _MIN_M = int(os.environ.get("HB_EMBED_SM_TARGET_MIN_M", "256") or "256")
    if not _PATCHED:
        _install_patch()
        _PATCHED = True
    logger.info(
        "[embed-sm-linear] marked %d embed linear modules, "
        "SM_COUNT_TARGET=%d, min_M=%d",
        n_marked, _SM_TARGET, _MIN_M,
    )


def _route(orig_apply, self, layer, x, bias):
    """Route to the SM-capped cuBLASLt linear iff this is an embed-model,
    prefill-shaped (large-M), bias-free, half-precision GEMM."""
    if _SM_TARGET > 0 and bias is None and x.is_cuda and x.dtype in (
        torch.bfloat16, torch.float16
    ) and getattr(layer, "_hb_embed_sm_capped", False):
        w = layer.weight
        if w.dim() == 2 and w.is_contiguous():
            m = x.numel() // x.shape[-1] if x.shape[-1] else 0
            if m >= _MIN_M:
                x2 = x.reshape(m, x.shape[-1])
                if x2.is_contiguous():
                    out = torch.ops.hb.embed_sm_capped_linear(
                        x2, w, _SM_TARGET
                    )
                    return out.view(*x.shape[:-1], w.shape[0])
    return orig_apply(self, layer, x, bias)


def _install_patch() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    orig_lin = UnquantizedLinearMethod.apply

    def patched_lin(self, layer, x, bias=None):
        return _route(orig_lin, self, layer, x, bias)

    UnquantizedLinearMethod.apply = patched_lin
    logger.info(
        "[embed-sm-linear] patched UnquantizedLinearMethod.apply "
        "(embed-stream-gated cuBLASLt SM_COUNT_TARGET path)."
    )
