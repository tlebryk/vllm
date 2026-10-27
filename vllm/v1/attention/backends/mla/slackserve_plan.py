# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size FlashInfer's cooperative MLA grid for the current ticket's SM budget.

The attention/merge kernel is unchanged. Only its CPU work partitioner receives
the enabled SM count instead of querying the whole device. Unrestricted work
delegates to FlashInfer's original planner, including all decode graph captures.
"""

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from types import SimpleNamespace

_SM_BUDGET: ContextVar[int | None] = ContextVar("mla_sm_budget", default=None)


def current_sm_budget() -> int | None:
    return _SM_BUDGET.get()


@contextmanager
def use_mla_sm_budget(sm_count: int | None):
    if sm_count is not None and (sm_count <= 0 or sm_count % 2):
        raise ValueError("An MLA TPC mask must enable a positive even SM count")
    token = _SM_BUDGET.set(sm_count)
    try:
        yield
    finally:
        _SM_BUDGET.reset(token)


@lru_cache(maxsize=1)
def planner_module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec

    header = (
        jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/scheduler.cuh"
    ).read_text()
    start = header.index("template <typename IdType>\ninline cudaError_t MLAPlan(")
    end = header.index("\n}", start) + 2
    function = header[start:end]
    signature = "cudaStream_t stream) {"
    query = (
        "FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, "
        "cudaDevAttrMultiProcessorCount, dev_id));"
    )
    if function.count(signature) != 1 or function.count(query) != 1:
        raise RuntimeError("FlashInfer MLA planner changed; review the budget adapter")
    function = function.replace("MLAPlan(", "SlackServeMLAPlan(", 1)
    function = function.replace(signature, "cudaStream_t stream, int enabled_sms) {")
    function = function.replace(
        query,
        query + "\n  if (enabled_sms <= 0 || enabled_sms > num_sm) "
        "return cudaErrorInvalidValue;\n  num_sm = enabled_sms;",
    )
    # Retain FlashInfer's license and reuse its helpers and exact scheduling
    # algorithm. The generated copy differs only in the explicit SM budget.
    source = (
        header[: header.index("#ifndef")]
        + "\n"
        + r"""
#include <flashinfer/attention/scheduler.cuh>
#include "tvm/ffi/container/array.h"
#include "tvm/ffi/container/tuple.h"
#include "tvm_ffi_utils.h"
namespace flashinfer {
"""
        + function
        + r"""
}
using tvm::ffi::Array;
using tvm::ffi::Tuple;
Tuple<Array<int64_t>, int64_t> plan(
    TensorView floats, TensorView ints, TensorView pinned,
    TensorView qo, TensorView kv, TensorView lens,
    int64_t heads, int64_t dim, bool causal, int64_t sms) {
  TVM_FFI_ICHECK(qo.dtype().code == kDLInt && qo.dtype().bits == 32);
  TVM_FFI_ICHECK(kv.dtype().code == kDLInt && kv.dtype().bits == 32);
  TVM_FFI_ICHECK(lens.dtype().code == kDLInt && lens.dtype().bits == 32);
  ffi::CUDADeviceGuard guard(floats.device().device_id);
  flashinfer::MLAPlanInfo info;
  size_t staged = 0;
  auto status = flashinfer::SlackServeMLAPlan(
      floats.data_ptr(), floats.size(0) * get_element_size(floats),
      ints.data_ptr(), pinned.data_ptr(), ints.size(0) * get_element_size(ints),
      info, staged, static_cast<int32_t*>(qo.data_ptr()),
      static_cast<int32_t*>(kv.data_ptr()), static_cast<int32_t*>(lens.data_ptr()),
      lens.size(0), heads, dim, causal, get_stream(floats.device()), sms);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  return Tuple<Array<int64_t>, int64_t>(Array(info.ToVector()), staged);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, plan);
"""
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    root = jit_env.FLASHINFER_GEN_SRC_DIR / f"slackserve_mla_plan_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.cu"
    if not path.exists():
        path.write_text(source)
    return gen_jit_spec(
        f"slackserve_mla_plan_{digest}",
        [path],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    ).build_and_load()


class _BudgetPlanModule:
    def __init__(self, original, sms):
        self.original = original
        self.sms = sms

    def __getattr__(self, name):
        return getattr(self.original, name)

    def plan_with_staged_workspace_bytes(self, *args):
        from flashinfer.jit.attention.modules import _BatchMLAModuleProxy

        # Preserve FlashInfer's pinned staging copy and allocator lifetime;
        # its C++ planner only writes host workspace, not device metadata.
        proxy = _BatchMLAModuleProxy(
            SimpleNamespace(
                plan=lambda *values: planner_module().plan(*values, self.sms)
            )
        )
        info, staged = proxy.plan_with_staged_workspace_bytes(*args)
        info = list(info)
        if info[0] * info[1] > self.sms:
            raise RuntimeError("Cooperative MLA grid exceeds enabled SM budget")
        return info, int(staged)


def budget_backend_type():
    from flashinfer.mla._batch_mla._backends.fa3_backend import (
        _BatchMLAPagedAttentionFa3Backend,
    )

    class BudgetFa3Backend(_BatchMLAPagedAttentionFa3Backend):
        def _plan_generated_fa(self, *, module_loader, **kwargs):
            sms = current_sm_budget()
            loader = module_loader
            if sms is not None:
                loader = lambda: _BudgetPlanModule(module_loader(), sms)
            return super()._plan_generated_fa(module_loader=loader, **kwargs)

    return BudgetFa3Backend
