"""Small cuBLASLt wrapper exposing the per-matmul SM-count target.

This is intentionally local to the microexperiment.  PyTorch does not expose
``CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET``, so the benchmark calls cuBLASLt via
ctypes while continuing to use PyTorch-owned tensors and streams.
"""

from __future__ import annotations

import ctypes as ct
import glob
import os
import sys
from dataclasses import dataclass

import torch

CUDA_R_16F = 2
CUDA_R_16BF = 14
CUDA_R_32F = 0
CUBLAS_COMPUTE_32F = 68
CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
CUBLAS_OP_T = 1

_DTYPE_TO_CUDA = {
    torch.float16: CUDA_R_16F,
    torch.bfloat16: CUDA_R_16BF,
}


class _HeuristicResult(ct.Structure):
    _fields_ = [
        ("algo", ct.c_byte * 64),
        ("workspace_size", ct.c_size_t),
        ("state", ct.c_int),
        ("waves_count", ct.c_float),
        ("reserved", ct.c_int * 4),
    ]


@dataclass
class _Plan:
    desc: ct.c_void_p
    layout_a: ct.c_void_p
    layout_b: ct.c_void_p
    layout_c: ct.c_void_p
    heuristic_results: object
    algo_index: int

    @property
    def workspace_size(self) -> int:
        return int(self.heuristic_results[self.algo_index].workspace_size)

    @property
    def waves_count(self) -> float:
        return float(self.heuristic_results[self.algo_index].waves_count)


def _find_cublaslt() -> ct.CDLL:
    for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return ct.CDLL(name)
        except OSError:
            pass

    for base in (path for path in sys.path if "site-packages" in path):
        matches = glob.glob(
            os.path.join(base, "nvidia", "cublas", "lib", "libcublasLt.so*")
        )
        if matches:
            return ct.CDLL(sorted(matches)[0])
    raise OSError("could not find libcublasLt")


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with cuBLAS status {status}")


class CublasLtMatmul:
    """Cached row-major ``out = left @ right`` plans with an SM target."""

    def __init__(self, workspace_bytes: int = 32 * 1024 * 1024) -> None:
        self.workspace_bytes = workspace_bytes
        self._library = _find_cublaslt()
        self._configure_signatures()
        self._handle = ct.c_void_p()
        _check(self._library.cublasLtCreate(ct.byref(self._handle)), "cublasLtCreate")
        self._plans: dict[tuple, _Plan] = {}
        self._workspaces: dict[torch.device, torch.Tensor] = {}

    def _configure_signatures(self) -> None:
        pointer = ct.c_void_p
        signatures = {
            "cublasLtCreate": [ct.POINTER(pointer)],
            "cublasLtMatmulDescCreate": [ct.POINTER(pointer), ct.c_int, ct.c_int],
            "cublasLtMatmulDescSetAttribute": [
                pointer,
                ct.c_int,
                pointer,
                ct.c_size_t,
            ],
            "cublasLtMatrixLayoutCreate": [
                ct.POINTER(pointer),
                ct.c_int,
                ct.c_uint64,
                ct.c_uint64,
                ct.c_int64,
            ],
            "cublasLtMatmulPreferenceCreate": [ct.POINTER(pointer)],
            "cublasLtMatmulPreferenceSetAttribute": [
                pointer,
                ct.c_int,
                pointer,
                ct.c_size_t,
            ],
            "cublasLtMatmulAlgoGetHeuristic": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                ct.c_int,
                pointer,
                ct.POINTER(ct.c_int),
            ],
            "cublasLtMatmul": [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                ct.c_size_t,
                pointer,
            ],
        }
        for name, argument_types in signatures.items():
            function = getattr(self._library, name)
            function.restype = ct.c_int
            function.argtypes = argument_types

    def _workspace(self, device: torch.device) -> torch.Tensor:
        workspace = self._workspaces.get(device)
        if workspace is None:
            workspace = torch.empty(
                self.workspace_bytes, dtype=torch.uint8, device=device
            )
            self._workspaces[device] = workspace
        return workspace

    def _make_layout(
        self, dtype: int, rows: int, columns: int, leading_dimension: int
    ) -> ct.c_void_p:
        layout = ct.c_void_p()
        _check(
            self._library.cublasLtMatrixLayoutCreate(
                ct.byref(layout),
                dtype,
                ct.c_uint64(rows),
                ct.c_uint64(columns),
                ct.c_int64(leading_dimension),
            ),
            "cublasLtMatrixLayoutCreate",
        )
        return layout

    def _build_plan(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        sm_target: int,
        native_linear_weight: bool,
    ) -> _Plan:
        cuda_dtype = _DTYPE_TO_CUDA[dtype]
        desc = ct.c_void_p()
        _check(
            self._library.cublasLtMatmulDescCreate(
                ct.byref(desc), CUBLAS_COMPUTE_32F, CUDA_R_32F
            ),
            "cublasLtMatmulDescCreate",
        )
        if sm_target > 0:
            target = ct.c_int32(sm_target)
            _check(
                self._library.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                    ct.cast(ct.byref(target), ct.c_void_p),
                    ct.sizeof(target),
                ),
                "set CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET",
            )

        if native_linear_weight:
            # A row-major linear weight is W[N,K]. Reinterpreted as a
            # column-major matrix its storage is [K,N], so transpose that
            # view to compute C_col[N,M] = W_row[N,K] @ X_row.T[K,M].
            transa = ct.c_int32(CUBLAS_OP_T)
            _check(
                self._library.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_TRANSA,
                    ct.cast(ct.byref(transa), ct.c_void_p),
                    ct.sizeof(transa),
                ),
                "set CUBLASLT_MATMUL_DESC_TRANSA",
            )

        # cuBLASLt defaults to column-major.  Reversing the operands gives a
        # zero-copy row-major matmul:
        #   C_col[N,M] = right_col[N,K] @ left_col[K,M]
        # whose storage is exactly out_row[M,N] = left_row @ right_row.
        if native_linear_weight:
            layout_a = self._make_layout(cuda_dtype, k, n, k)
        else:
            layout_a = self._make_layout(cuda_dtype, n, k, n)
        layout_b = self._make_layout(cuda_dtype, k, m, k)
        layout_c = self._make_layout(cuda_dtype, n, m, n)

        preference = ct.c_void_p()
        _check(
            self._library.cublasLtMatmulPreferenceCreate(ct.byref(preference)),
            "cublasLtMatmulPreferenceCreate",
        )
        max_workspace = ct.c_size_t(self.workspace_bytes)
        _check(
            self._library.cublasLtMatmulPreferenceSetAttribute(
                preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ct.cast(ct.byref(max_workspace), ct.c_void_p),
                ct.sizeof(max_workspace),
            ),
            "set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",
        )

        results = (_HeuristicResult * 16)()
        returned = ct.c_int()
        _check(
            self._library.cublasLtMatmulAlgoGetHeuristic(
                self._handle,
                desc,
                layout_a,
                layout_b,
                layout_c,
                layout_c,
                preference,
                16,
                results,
                ct.byref(returned),
            ),
            "cublasLtMatmulAlgoGetHeuristic",
        )
        if returned.value == 0:
            raise RuntimeError(
                f"no cuBLASLt algorithm for M={m}, N={n}, K={k}, "
                f"dtype={dtype}, sm_target={sm_target}"
            )
        algorithm_index = next(
            (
                index
                for index in range(returned.value)
                if results[index].state == 0
                and results[index].workspace_size <= self.workspace_bytes
            ),
            None,
        )
        if algorithm_index is None:
            raise RuntimeError(
                f"cuBLASLt returned no successful algorithm within "
                f"{self.workspace_bytes} workspace bytes"
            )
        return _Plan(
            desc=desc,
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            heuristic_results=results,
            algo_index=algorithm_index,
        )

    def plan(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        sm_target: int,
        device: torch.device,
        native_linear_weight: bool = False,
    ) -> _Plan:
        key = (m, n, k, dtype, sm_target, device, native_linear_weight)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build_plan(m, n, k, dtype, sm_target, native_linear_weight)
            self._plans[key] = plan
        return plan

    def _launch(
        self,
        left: torch.Tensor,
        right_or_weight: torch.Tensor,
        out: torch.Tensor,
        plan: _Plan,
    ) -> None:
        workspace = self._workspace(left.device)
        alpha = ct.c_float(1.0)
        beta = ct.c_float(0.0)
        stream = torch.cuda.current_stream(left.device).cuda_stream

        _check(
            self._library.cublasLtMatmul(
                self._handle,
                plan.desc,
                ct.cast(ct.byref(alpha), ct.c_void_p),
                ct.c_void_p(right_or_weight.data_ptr()),
                plan.layout_a,
                ct.c_void_p(left.data_ptr()),
                plan.layout_b,
                ct.cast(ct.byref(beta), ct.c_void_p),
                ct.c_void_p(out.data_ptr()),
                plan.layout_c,
                ct.c_void_p(out.data_ptr()),
                plan.layout_c,
                ct.cast(
                    ct.byref(plan.heuristic_results[plan.algo_index].algo),
                    ct.c_void_p,
                ),
                ct.c_void_p(workspace.data_ptr()),
                ct.c_size_t(self.workspace_bytes),
                ct.c_void_p(stream),
            ),
            "cublasLtMatmul",
        )

    def matmul(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        out: torch.Tensor,
        sm_target: int,
    ) -> None:
        """Queue ``out = left @ right`` on PyTorch's current CUDA stream."""
        if left.ndim != 2 or right.ndim != 2 or out.ndim != 2:
            raise ValueError("left, right, and out must be rank-2 tensors")
        if left.dtype not in _DTYPE_TO_CUDA:
            raise ValueError(f"unsupported dtype: {left.dtype}")
        if not (left.dtype == right.dtype == out.dtype):
            raise ValueError("left, right, and out must have the same dtype")
        if not (left.is_contiguous() and right.is_contiguous() and out.is_contiguous()):
            raise ValueError("left, right, and out must be contiguous")
        m, k = left.shape
        right_k, n = right.shape
        if right_k != k or out.shape != (m, n):
            raise ValueError(
                f"incompatible shapes: {left.shape}, {right.shape}, {out.shape}"
            )

        plan = self.plan(m, n, k, left.dtype, sm_target, left.device)
        self._launch(left, right, out, plan)

    def linear(
        self,
        left: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        sm_target: int,
    ) -> None:
        """Queue ``out = left @ weight.T`` with native contiguous ``weight[N,K]``."""
        if left.ndim != 2 or weight.ndim != 2 or out.ndim != 2:
            raise ValueError("left, weight, and out must be rank-2 tensors")
        if left.dtype not in _DTYPE_TO_CUDA:
            raise ValueError(f"unsupported dtype: {left.dtype}")
        if not (left.dtype == weight.dtype == out.dtype):
            raise ValueError("left, weight, and out must have the same dtype")
        if not (
            left.is_contiguous() and weight.is_contiguous() and out.is_contiguous()
        ):
            raise ValueError("left, weight, and out must be contiguous")
        m, k = left.shape
        n, weight_k = weight.shape
        if weight_k != k or out.shape != (m, n):
            raise ValueError(
                f"incompatible shapes: {left.shape}, {weight.shape}, {out.shape}"
            )

        plan = self.plan(
            m,
            n,
            k,
            left.dtype,
            sm_target,
            left.device,
            native_linear_weight=True,
        )
        self._launch(left, weight, out, plan)

    def plan_info(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        sm_target: int,
        device: torch.device,
        native_linear_weight: bool = False,
    ) -> dict[str, float | int]:
        plan = self.plan(m, n, k, dtype, sm_target, device, native_linear_weight)
        return {
            "workspace_bytes": plan.workspace_size,
            "waves_count": plan.waves_count,
        }
