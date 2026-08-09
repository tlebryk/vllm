"""ctypes binding for libsmctrl stream TPC masking (Bullet's fork of UNC
libsmctrl). Experimental placement control for Slack Serve lanes.

A stream mask pins every kernel launched into that CUDA stream onto an
explicit TPC range (2 SMs per TPC on Hopper). Unlike the cuBLASLt SM-count
target, this is placement, not an algorithm-selection hint: masked kernels
cannot run outside their range, and kernels on unmasked streams still roam
the whole GPU.

Env contract (read by SlackServeGPUWorker.init_device):
  HB_SMCTRL_LIB          absolute path to libsmctrl.so (enables the feature)
  HB_SMCTRL_PREFILL_TPCS "lo:hi" TPC range for the prefill/dense lane stream
  HB_SMCTRL_DECODE_TPCS  "lo:hi" TPC range for the decode stream

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

    def _set_mask(self, stream, mask: int, desc: str) -> None:
        arg = _CUint128(
            low=mask & 0xFFFFFFFFFFFFFFFF, high=(mask >> 64) & 0xFFFFFFFFFFFFFFFF
        )
        ret = self.lib.libsmctrl_set_stream_mask_ext(
            ctypes.c_void_p(stream.cuda_stream), arg
        )
        if ret != 0:
            raise OSError(ret, f"libsmctrl_set_stream_mask_ext failed: {ret}")
        logger.info("smctrl: stream %#x pinned to %s", stream.cuda_stream, desc)

    def set_stream_tpc_range(self, stream, lo: int, hi: int) -> None:
        """Pin ``stream`` (torch.cuda.Stream) to TPCs [lo, hi)."""
        self._set_mask(
            stream,
            _disable_mask(lo, hi),
            f"TPCs [{lo}, {hi}) (SMs [{2 * lo}, {2 * hi}))",
        )

    def set_stream_tpc_list(self, stream, indices: list[int]) -> None:
        """Pin ``stream`` to an explicit TPC index set (may be scattered)."""
        self._set_mask(
            stream, _disable_mask_from_indices(indices), f"TPCs {indices}"
        )


def parse_range(spec: str) -> tuple[int, int]:
    lo, hi = spec.split(":")
    lo, hi = int(lo), int(hi)
    if not 0 <= lo < hi:
        raise ValueError(f"bad TPC range {spec!r}")
    return lo, hi


def _apply_spec(ctrl: SmCtrl, stream, spec: str) -> None:
    """Apply "lo:hi" (contiguous) or "spread:k" (k TPCs evenly spaced)."""
    if spec.startswith("spread:"):
        import torch

        total = torch.cuda.get_device_properties(0).multi_processor_count // 2
        count = int(spec.split(":", 1)[1])
        ctrl.set_stream_tpc_list(stream, spread_indices(count, total))
    else:
        lo, hi = parse_range(spec)
        ctrl.set_stream_tpc_range(stream, lo, hi)


_UNCAP_STATE: dict[int, bool] = {}
_UNCAP_CTRL: "SmCtrl | None" = None


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
    for name, env in (
        ("prefill", "HB_SMCTRL_PREFILL_TPCS"),
        ("decode", "HB_SMCTRL_DECODE_TPCS"),
    ):
        spec = os.environ.get(env)
        stream = named_streams.get(name)
        if spec and stream is not None:
            _apply_spec(ctrl, stream, spec)
