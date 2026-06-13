"""Route decode-model unquantized linears through a cuBLASLt GEMM with
CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET pinned, ONLY when executing on the
masked decode stream.

Background: under a libsmctrl stream mask exposing <32 TPC, the default
cuBLASLt Hopper GEMM (sized for the full 132-SM device) deadlocks. Pinning
SM_COUNT_TARGET to a reduced value clears the hang. PyTorch's F.linear /
torch.matmul do not expose that descriptor attribute, so we monkeypatch
``UnquantizedLinearMethod.apply`` to detect "am I on the registered decode
stream?" at call time and, if so (and if HB_DECODE_SM_COUNT_TARGET is set),
route through the custom cuBLASLt linear. Otherwise fall through to the
original implementation. This leaves the embed model (which runs unmasked on
the embed/default stream) completely untouched.

Activation is gated by env var ``HB_DECODE_SM_COUNT_TARGET`` (int):
  unset / 0  -> disabled, original behavior.
  > 0        -> use cuBLASLt linear with that SM target on the decode stream.

The decode stream's ``cuda_stream`` integer handle is registered by
dual_model_runner once masking is set up (see register_decode_stream).
"""
from __future__ import annotations

import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Set once by dual_model_runner after the (possibly upgraded) decode stream is
# created. Integer CUstream handle of the decode stream.
_DECODE_STREAM_PTR: int | None = None
_SM_TARGET: int = 0
# Empirical hang map for the o_proj-shaped GEMM (K=4096,N=2560) under a 16-TPC
# (32-SM) libsmctrl mask, M = rows (tokens):
#       M       reduced(smt=2*TPC)    default(F.linear / smt=0)
#       64        OK                    HANG  <- decode steady-state batch
#       128       HANG                  HANG  <- unavoidable dead zone
#       256       HANG                  OK
#       384       OK                    OK
#       512       HANG                  OK
#       2816      OK (smt=0/32)         OK
#       32768     OK (smt=0)            OK    <- chunked-prefill batch
# Takeaway: the reduced SM target ONLY reliably clears the hang for the SMALL
# decode-shaped M (= n_decode, e.g. 64). At larger M the default cuBLAS sizing
# works and the reduced target instead INDUCES hangs. So we route ONLY small-M
# (M <= _MAX_M) GEMMs through the reduced-SM cuBLASLt path; everything larger
# (chunked prefill at M>=256) falls through to the default torch path which is
# fine at those sizes. The 128-token point hangs on BOTH paths, but the decode
# / chunked-prefill workload here does not schedule that size.
_MAX_M: int = 64
_PATCHED: bool = False
_cublaslt_sm_linear = None  # lazily imported callable


def _load_cublaslt_fn():
    global _cublaslt_sm_linear
    if _cublaslt_sm_linear is not None:
        return _cublaslt_sm_linear
    # The module lives in the project scripts dir, not the vllm package.
    scripts_dir = os.environ.get(
        "HB_CUBLASLT_SM_LINEAR_DIR",
        "/mnt/weka/theo/heterogenious-batching-vllm/scripts",
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from cublaslt_sm_linear import cublaslt_sm_linear  # type: ignore
    _cublaslt_sm_linear = cublaslt_sm_linear
    return _cublaslt_sm_linear


def register_decode_stream(stream_ptr: int) -> None:
    """Register the decode stream handle and install the monkeypatch.

    Called by dual_model_runner once the masked decode stream is ready. No-op
    unless HB_DECODE_SM_COUNT_TARGET > 0.
    """
    global _DECODE_STREAM_PTR, _SM_TARGET, _PATCHED, _MAX_M
    target = int(os.environ.get("HB_DECODE_SM_COUNT_TARGET", "0") or "0")
    if target <= 0:
        logger.info(
            "[decode-sm-linear] HB_DECODE_SM_COUNT_TARGET unset/0; "
            "decode linears use default torch path."
        )
        return
    _DECODE_STREAM_PTR = int(stream_ptr)
    _SM_TARGET = target
    _MAX_M = int(os.environ.get("HB_DECODE_SM_TARGET_MAX_M", "64") or "64")
    if not _PATCHED:
        _install_patch()
        _PATCHED = True
    logger.info(
        "[decode-sm-linear] registered decode stream ptr=0x%x, "
        "SM_COUNT_TARGET=%d",
        _DECODE_STREAM_PTR, _SM_TARGET,
    )


def _route(orig_apply, self, layer, x, bias):
    """Shared body: route to cuBLASLt SM-target linear iff on decode stream."""
    if _DECODE_STREAM_PTR is not None and _SM_TARGET > 0:
        try:
            cur = torch.cuda.current_stream(x.device).cuda_stream
        except Exception:
            cur = None
        if cur == _DECODE_STREAM_PTR and x.dtype in (
            torch.bfloat16, torch.float16
        ) and x.is_cuda:
            # M = product of all-but-last dims (tokens). Only route small-M
            # (decode-shaped) GEMMs through the reduced-SM path; large-M
            # (prefill) GEMMs use the default path (see _MAX_M comment).
            m = x.numel() // x.shape[-1] if x.shape[-1] else 0
            if 0 < m <= _MAX_M:
                fn = _load_cublaslt_fn()
                return fn(x, layer.weight, bias, _SM_TARGET)
    return orig_apply(self, layer, x, bias)


def _install_patch() -> None:
    # Patch BOTH the standard linear method AND the embedding/lm_head method.
    # The lm_head logits GEMM (hidden -> vocab) goes through
    # UnquantizedEmbeddingMethod.apply, NOT UnquantizedLinearMethod.apply --
    # it is a huge narrow-M cuBLAS GEMM and would deadlock under a <32-TPC
    # mask if left on the default torch path.
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    orig_lin = UnquantizedLinearMethod.apply

    def patched_lin(self, layer, x, bias=None):
        return _route(orig_lin, self, layer, x, bias)

    UnquantizedLinearMethod.apply = patched_lin

    orig_emb = UnquantizedEmbeddingMethod.apply

    def patched_emb(self, layer, x, bias=None):
        return _route(orig_emb, self, layer, x, bias)

    UnquantizedEmbeddingMethod.apply = patched_emb

    logger.info(
        "[decode-sm-linear] patched UnquantizedLinearMethod.apply and "
        "UnquantizedEmbeddingMethod.apply (decode-stream-gated cuBLASLt "
        "SM-target path)."
    )
