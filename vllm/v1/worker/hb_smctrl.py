# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes binding for libsmctrl stream TPC masking (Bullet's fork of UNC
libsmctrl). Experimental placement control for Slack Serve lanes.

A stream mask pins every kernel launched into that CUDA stream onto an
explicit TPC range (2 SMs per TPC on Hopper). Unlike the cuBLASLt SM-count
target, this is placement, not an algorithm-selection hint: masked kernels
cannot run outside their range, and kernels on unmasked streams still roam
the whole GPU.

Env contract (read by SlackServeGPUWorker.init_device):
  HB_SMCTRL_LIB          absolute path to libsmctrl.so (enables the feature)
  HB_SMCTRL_PREFILL_TPCS "lo:hi", "spread:k", or "list:i,j,..." for prefill
  HB_SMCTRL_DECODE_TPCS  same forms for the decode stream

Requires driver 570.124.06 / CUDA 12.8 (offset-fragile: libsmctrl pokes a
hardcoded offset in the driver's stream struct, keyed on driver version).
"""

import ctypes
import logging
import os

logger = logging.getLogger(__name__)


class _CUint128(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]


def _disable_mask(lo: int, hi: int) -> int:
    """128-bit libsmctrl mask with TPCs [lo, hi) ENABLED (set bit = disabled)."""
    enabled = ((1 << (hi - lo)) - 1) << lo
    return ~enabled & ((1 << 128) - 1)


def spread_indices(count: int, total: int) -> list[int]:
    """``count`` TPC indices spread evenly across [0, total) — preserves
    GPC/L2 locality spread that a contiguous range destroys."""
    return [(j * total) // count for j in range(count)]


def _disable_mask_from_indices(indices: list[int]) -> int:
    enabled = 0
    for i in indices:
        enabled |= 1 << i
    return ~enabled & ((1 << 128) - 1)


class SmCtrl:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.libsmctrl_set_stream_mask_ext.argtypes = [
            ctypes.c_void_p,
            _CUint128,
        ]
        self.lib.libsmctrl_set_stream_mask_ext.restype = ctypes.c_int

    def _set_mask(self, stream, mask: int, desc: str, *, log: bool = True) -> None:
        arg = _mask_arg(mask)
        self.set_stream_mask_arg(stream, arg)
        if log:
            logger.info("smctrl: stream %#x pinned to %s", stream.cuda_stream, desc)

    def set_stream_mask_arg(self, stream, arg: _CUint128) -> None:
        """Hot path for a mask parsed once during hook registration."""
        ret = self.lib.libsmctrl_set_stream_mask_ext(
            ctypes.c_void_p(stream.cuda_stream), arg
        )
        if ret != 0:
            raise OSError(ret, f"libsmctrl_set_stream_mask_ext failed: {ret}")

    def set_stream_tpc_range(self, stream, lo: int, hi: int) -> None:
        """Pin ``stream`` (torch.cuda.Stream) to TPCs [lo, hi)."""
        self._set_mask(
            stream,
            _disable_mask(lo, hi),
            f"TPCs [{lo}, {hi}) (SMs [{2 * lo}, {2 * hi}))",
        )

    def set_stream_tpc_list(self, stream, indices: list[int]) -> None:
        """Pin ``stream`` to an explicit TPC index set (may be scattered)."""
        self._set_mask(stream, _disable_mask_from_indices(indices), f"TPCs {indices}")

    def set_stream_spec(self, stream, spec: str, *, log: bool = True) -> None:
        """Apply a parsed mask spec, optionally suppressing hot-path logging."""
        import torch

        total = torch.cuda.get_device_properties(stream.device).multi_processor_count // 2
        self.set_stream_mask_arg(stream, compile_stream_spec(spec, total))
        if log:
            logger.info("smctrl: stream %#x pinned to %s", stream.cuda_stream, spec)


def parse_range(spec: str) -> tuple[int, int]:
    lo_raw, hi_raw = spec.split(":")
    lo, hi = int(lo_raw), int(hi_raw)
    if not 0 <= lo < hi:
        raise ValueError(f"bad TPC range {spec!r}")
    return lo, hi


def _mask_arg(mask: int) -> _CUint128:
    return _CUint128(
        low=mask & 0xFFFFFFFFFFFFFFFF,
        high=(mask >> 64) & 0xFFFFFFFFFFFFFFFF,
    )


def compile_stream_spec(spec: str, total: int) -> _CUint128:
    """Parse a stream spec once and return the ctypes argument for hot hooks."""
    if spec.startswith("spread:"):
        count = int(spec.split(":", 1)[1])
        if not 1 <= count <= total:
            raise ValueError(f"spread count {count} is outside [1, {total}]")
        mask = _disable_mask_from_indices(spread_indices(count, total))
    elif spec.startswith("list:"):
        indices = [int(item) for item in spec.split(":", 1)[1].split(",") if item]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"invalid explicit TPC list: {spec!r}")
        if min(indices) < 0 or max(indices) >= total:
            raise ValueError(f"TPC list {spec!r} is outside [0, {total})")
        mask = _disable_mask_from_indices(indices)
    else:
        lo, hi = parse_range(spec)
        if hi > total:
            raise ValueError(f"TPC range {spec!r} is outside [0, {total})")
        mask = _disable_mask(lo, hi)
    return _mask_arg(mask)


def _apply_spec(ctrl: SmCtrl, stream, spec: str) -> None:
    """Apply a contiguous, evenly spread, or explicit TPC mask."""
    import torch

    total = torch.cuda.get_device_properties(stream.device).multi_processor_count // 2
    ctrl.set_stream_mask_arg(stream, compile_stream_spec(spec, total))
    logger.info("smctrl: stream %#x pinned to %s", stream.cuda_stream, spec)


_UNCAP_STATE: dict[int, bool] = {}
_UNCAP_CTRL: "SmCtrl | None" = None

# Decode-ready running count at prefill-dispatch time, written by the
# two-lane controller (same process under UniProcExecutor) just before each
# prefill dispatch. Consumed by maybe_mask_prefill_for_load.
RUNNING_DECODE_HINT: int = 0


def maybe_mask_prefill_for_load(stream) -> None:
    """HB_SMCTRL_MASK_MIN_RUNNING=k: load-conditional prefill mask.

    Mask the prefill stream (HB_SMCTRL_PREFILL_TPCS) only when at least k
    decode-ready requests are running; uncap it otherwise. Rationale
    (lambda<=1 open-loop, 8k corpus): the mask protects the decode ITL tail
    but dilates prefill ~29% (~+63ms TTFT on 8k prompts); with few running
    decodes there is nothing to protect and TTFT dominates mean e2el. ~us
    cost, no-ops unless the state changes or the smctrl envs are unset.
    """
    spec = os.environ.get("HB_SMCTRL_MASK_MIN_RUNNING")
    if not spec:
        return
    apply_prefill_uncap(stream, uncapped=int(spec) > RUNNING_DECODE_HINT)


def apply_prefill_uncap(stream, uncapped: bool) -> None:
    """Re-mask the prefill/dense stream for the embed-only tail: full GPU
    when ``uncapped``, the HB_SMCTRL_PREFILL_TPCS spec otherwise. Cheap
    (~us) but stateful — no-ops unless the state actually changes. No-op
    entirely when the smctrl envs are unset, so unmasked arms are untouched.
    """
    global _UNCAP_CTRL
    lib = os.environ.get("HB_SMCTRL_LIB")
    spec = os.environ.get("HB_SMCTRL_PREFILL_TPCS")
    if not lib or not spec:
        return
    key = stream.cuda_stream
    if _UNCAP_STATE.get(key) == uncapped:
        return
    if _UNCAP_CTRL is None:
        _UNCAP_CTRL = SmCtrl(lib)
    if uncapped:
        import torch

        total = torch.cuda.get_device_properties(0).multi_processor_count // 2
        _UNCAP_CTRL.set_stream_tpc_range(stream, 0, total)
    else:
        _apply_spec(_UNCAP_CTRL, stream, spec)
    _UNCAP_STATE[key] = uncapped


def apply_env_masks(named_streams: dict[str, "object"]) -> None:
    """Apply HB_SMCTRL_* env masks to streams named 'prefill'/'decode'."""
    lib_path = os.environ.get("HB_SMCTRL_LIB")
    if not lib_path:
        return
    ctrl = SmCtrl(lib_path)
    seen = set()
    for name, stream in named_streams.items():
        if stream is None or stream.cuda_stream in seen:
            continue
        seen.add(stream.cuda_stream)
        env = (
            "HB_SMCTRL_PREFILL_TPCS" if name.startswith("prefill")
            else "HB_SMCTRL_DECODE_TPCS" if name == "decode"
            else None
        )
        if env is None:
            continue
        spec = os.environ.get(env)
        if spec:
            _apply_spec(ctrl, stream, spec)


def register_linear_mask_hooks(model, prefill_streams: list["object"]) -> int:
    """Optionally give LinearBase launches a different mask from other ops.

    Enabled by HB_SMCTRL_LINEAR_TPCS.  A forward pre-hook changes only a
    configured prefill stream to that mask; the post-hook restores the base
    HB_SMCTRL_PREFILL_TPCS mask.  CUDA launches are asynchronous, but their
    QMD captures the stream mask before the Python call returns, so the restore
    affects subsequent operations rather than the queued linear kernels.
    """
    linear_spec = os.environ.get("HB_SMCTRL_LINEAR_TPCS")
    base_spec = os.environ.get("HB_SMCTRL_PREFILL_TPCS")
    lib = os.environ.get("HB_SMCTRL_LIB")
    if not (linear_spec and base_spec and lib):
        return 0

    incompatible = [
        name for name in ("HB_SMCTRL_MASK_MIN_RUNNING", "HB_SMCTRL_TAIL_UNCAP")
        if os.environ.get(name)
    ]
    if os.environ.get("HB_P2_COMPILE_PREFILL") == "1":
        incompatible.append("HB_P2_COMPILE_PREFILL=1")
    if incompatible:
        raise RuntimeError(
            "operator-class masking requires eager, fixed-mask prefill; "
            f"incompatible settings: {', '.join(incompatible)}"
        )
    if getattr(model, "_hb_linear_mask_hook_handles", None):
        raise RuntimeError("LinearBase mask hooks are already registered")

    import torch
    from vllm.model_executor.layers.linear import LinearBase

    ctrl = SmCtrl(lib)
    stream_by_handle = {stream.cuda_stream: stream for stream in prefill_streams}
    if not prefill_streams:
        raise RuntimeError("operator-class masking requires a prefill stream")
    devices = {stream.device for stream in prefill_streams}
    if len(devices) != 1:
        raise RuntimeError(f"prefill streams span multiple devices: {devices}")
    device = next(iter(devices))
    total = torch.cuda.get_device_properties(device).multi_processor_count // 2
    linear_mask_arg = compile_stream_spec(linear_spec, total)
    base_mask_arg = compile_stream_spec(base_spec, total)
    hook_hits = {"pre": False, "post": False}

    def pre_hook(_module, _args):
        stream = torch.cuda.current_stream()
        if stream.cuda_stream in stream_by_handle:
            ctrl.set_stream_mask_arg(stream, linear_mask_arg)
            if not hook_hits["pre"]:
                hook_hits["pre"] = True
                logger.info("smctrl: first LinearBase pre-hook mask applied")

    def post_hook(_module, _args, output):
        stream = torch.cuda.current_stream()
        if stream.cuda_stream in stream_by_handle:
            ctrl.set_stream_mask_arg(stream, base_mask_arg)
            if not hook_hits["post"]:
                hook_hits["post"] = True
                logger.info("smctrl: first LinearBase post-hook restore applied")
        return output

    linears = [module for module in model.modules()
               if isinstance(module, LinearBase)]
    for module in linears:
        if any(isinstance(child, LinearBase)
               for child in list(module.modules())[1:]):
            raise RuntimeError("nested LinearBase modules are not supported")
    handles = []
    for module in linears:
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook, always_call=True))
    model._hb_linear_mask_hook_handles = handles
    count = len(linears)
    logger.info(
        "smctrl: registered %d LinearBase mask hooks (%s -> %s)",
        count,
        base_spec,
        linear_spec,
    )
    return count
