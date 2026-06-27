# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import inspect
import os
import queue
import threading
from contextlib import contextmanager
from dataclasses import is_dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.dual_model_helpers import (
    DualModelConfig,
    merge_model_runner_outputs,
    split_scheduler_output_by_model,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _defer_decode_sample_d2h() -> bool:
    """HB_DEFER_DECODE_SAMPLE_D2H feature flag (default OFF).

    When ON, the decode child runner's sample step runs in async-output mode so
    its blocking D2H token-id sync (``_to_list`` -> ``transfer_event.synchronize``
    in gpu_model_runner.py) is replaced by a deferred ``get_output()``. The dual
    runner then resolves that decode ``get_output()`` AFTER it has dispatched the
    embed pool onto the (separate) embed stream, so the decode D2H drain overlaps
    with the embed pool's GPU work instead of stalling the engine hot thread
    before embed gets a chance to run.

    This is NOT engine-wide async scheduling: the engine scheduler stays
    synchronous, decode and embed keep their own CUDA streams + persistent
    workers, and the merged output is still produced in the SAME step (no output
    lag, no placeholder tokens leaking to the scheduler). When OFF the code path
    is byte-identical to before.
    """
    return os.environ.get("HB_DEFER_DECODE_SAMPLE_D2H", "0").lower() in (
        "1", "true", "yes", "on"
    )


def _dual_graph_both_streams() -> bool:
    """HB_DUAL_GRAPH_BOTH_STREAMS feature flag (default OFF).

    T1 lever (see project_engine_overlap_blocker). When ON, the dual runner
    dispatches the decode forward and the embed forward BACK-TO-BACK from a
    single host thread, each as a single CUDA-graph replay on its own stream,
    with NO thread spawn / join / cross-stream event between them and NO
    blocking D2H sync inside the step. The intent: collapse the ~495 per-kernel
    host launches per decode step into ~1 graph replay so the decode stream's
    kernels become dense, then immediately enqueue the embed graph on the embed
    stream so both stream queues stay full and the GPU scheduler can co-issue
    decode-memory-bound + embed-compute-bound kernels (the 26.5% microbench
    ceiling).

    Requirements when ON:
      - Both child models MUST be running FULL CUDA graphs (``--no-enforce-eager
        --cudagraph-mode FULL``); otherwise each forward is still ~495 host
        launches and there is no dense window. The flag still runs correctly
        (it just won't densify), and a warning is logged at first dispatch.
    This flag runs at ``max_concurrent_batches=1`` by default: the back-to-back
    replay is WITHIN a single engine step, so it does NOT require the executor's
    2-deep batch queue. At mcb=1 the engine calls ``sample_tokens`` blocking and
    the decode token-id D2H is resolved inline (the merged ModelRunnerOutput is
    returned synchronously). The 2-deep batch queue + deferred decode token-id
    D2H pipeline is an ORTHOGONAL optimization gated separately by
    ``HB_DEFER_DECODE_SAMPLE_D2H``; set BOTH flags to get the graph replay AND
    the deferred-D2H pipeline (mcb=2). With only this flag set, the scheduler can
    pack all running decode seqs into one step (the 2-deep queue would otherwise
    split them across two in-flight steps).

    When OFF the code path is byte-identical to before.
    """
    return os.environ.get("HB_DUAL_GRAPH_BOTH_STREAMS", "0").lower() in (
        "1", "true", "yes", "on"
    )


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def _temporary_async_outputs(runner: GPUModelRunner, enabled: bool):
    old_value = runner.use_async_scheduling
    if enabled:
        runner.use_async_scheduling = True
    try:
        yield
    finally:
        runner.use_async_scheduling = old_value


def _resolve_model_runner_output(
    output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
    label: str,
) -> ModelRunnerOutput | None:
    if isinstance(output, AsyncModelRunnerOutput):
        with _nvtx_range(label):
            return output.get_output()
    return output


class DualAsyncModelRunnerOutput(AsyncModelRunnerOutput):
    """Async wrapper that defers decode+embed merge until get_output().

    Each child slot may be a sync ModelRunnerOutput, an AsyncModelRunnerOutput
    (whose D2H copy is in flight on the child runner's async copy stream), or
    None for a step that scheduled no work in that branch. get_output() is
    invoked on the executor's async output thread; it resolves both children
    (blocking on their D2H copy events) and emits the merged ModelRunnerOutput.

    The scheduled_req_order is captured at construction time so this object is
    safe to resolve after the next engine step has begun mutating the parent
    DualModelRunner's pending_* instance attributes.
    """

    def __init__(
        self,
        scheduled_req_order: list[str],
        decode_output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
        embed_output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
    ):
        self._scheduled_req_order = list(scheduled_req_order)
        self._decode_output = decode_output
        self._embed_output = embed_output

    def get_output(self) -> ModelRunnerOutput:
        decode_resolved = _resolve_model_runner_output(
            self._decode_output, "decode_output_get"
        )
        embed_resolved = _resolve_model_runner_output(
            self._embed_output, "embed_output_get"
        )
        with _nvtx_range("merge_outputs"):
            return merge_model_runner_outputs(
                scheduled_req_order=self._scheduled_req_order,
                decode_output=decode_resolved,
                embed_output=embed_resolved,
            )


class _PersistentWorker:
    """Long-lived worker thread that owns a CUDA stream and serializes
    submitted callables on it.

    Replaces the per-step `threading.Thread(...).start()/join()` pattern used
    by the embed_concurrent dispatch path. Eliminates the ~1 ms-per-step
    Python-side thread-spawn cost, and lets decode_sample + embed_pool overlap
    when each model has its own dedicated worker (each worker runs its
    callable on its own stream + thread, so the two streams' kernels can
    actually run in parallel on the GPU).

    Lifecycle:
      - Spawned in DualModelRunner.__init__ when execute_order ==
        "persistent_workers".
      - Daemon thread; .shutdown() on a sentinel value at runner teardown
        (best-effort -- daemon ensures process exit doesn't hang).
      - Each submission returns (done_event, result_holder). Caller waits via
        wait(); exceptions raised on the worker re-raise on the waiter.
    """

    def __init__(self, name: str, stream: torch.cuda.Stream,
                 device: torch.device):
        self.name = name
        self.stream = stream
        self.device = device
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"dual_model_{name}_worker",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        # CUDA current-device is thread-local; set once per worker thread so
        # `torch.cuda.stream(...)` and any tensor allocs land on the right
        # device.
        torch.cuda.set_device(self.device)
        while True:
            item = self._queue.get()
            if item is None:
                return  # shutdown sentinel
            func, done_event, result_holder = item
            try:
                with torch.cuda.stream(self.stream):
                    result_holder["value"] = func()
            except BaseException as e:  # noqa: BLE001 -- re-raised on waiter
                result_holder["exception"] = e
            finally:
                done_event.set()

    def submit(self, func: Callable[[], Any]) -> tuple[threading.Event, dict]:
        done_event = threading.Event()
        result_holder: dict = {}
        self._queue.put((func, done_event, result_holder))
        return done_event, result_holder

    @staticmethod
    def wait(done_event: threading.Event, result_holder: dict) -> Any:
        done_event.wait()
        if "exception" in result_holder:
            raise result_holder["exception"]
        return result_holder.get("value")

    def shutdown(self) -> None:
        self._queue.put(None)
        # Daemon thread; best-effort join. We don't want shutdown to block
        # forever if the worker is stuck on a hung kernel.
        self._thread.join(timeout=2.0)


class DualModelRunner:
    """Experimental wrapper that hosts one decode and one embed runner."""

    EMBED_MODEL_PREFIX = "embed_model"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if not envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("DualModelRunner currently requires V2 model runner.")
        # POC (worktree async-dual-runner): async scheduling is now supported.
        # When True, both child runners return AsyncModelRunnerOutput, and
        # execute_model/sample_tokens return DualAsyncModelRunnerOutput so the
        # decode+embed merge happens on the executor's async output thread,
        # in parallel with the next engine step's CPU-side scheduling.
        self.use_async_scheduling = bool(
            vllm_config.scheduler_config.async_scheduling
        )
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "DualModelRunner currently requires pipeline_parallel_size=1."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DualModelRunner currently does not support speculative decoding."
            )
        self.decode_stream = torch.cuda.default_stream(device)
        stream_mode = os.environ.get("VLLM_DUAL_MODEL_STREAM_MODE", "two_stream")
        if stream_mode == "default_stream":
            self.embed_stream = self.decode_stream
        elif stream_mode == "two_stream":
            embed_priority = int(
                os.environ.get("VLLM_DUAL_MODEL_EMBED_STREAM_PRIORITY", "0")
            )
            self.embed_stream = torch.cuda.Stream(
                device, priority=embed_priority
            )
        else:
            raise ValueError(
                "VLLM_DUAL_MODEL_STREAM_MODE must be 'two_stream' or "
                f"'default_stream', got {stream_mode!r}."
            )

        # libsmctrl TPC masking for the two streams (no-op unless env vars set).
        # Masks are 64-bit ints (bit set = TPC DISABLED, per libsmctrl convention).
        # Accept hex (0x...) or decimal. HB_LIBSMCTRL_SO must point to the .so.
        #
        # NOTE: libsmctrl masking writes into the per-stream metadata struct, so
        # it requires a real (non-default) CUstream*. When decode masking is
        # requested, we replace `decode_stream` with a fresh CUDA stream — the
        # legacy default stream's handle (NULL) can't be masked.
        decode_mask_env = os.environ.get("HB_LIBSMCTRL_DECODE_TPC_MASK")
        embed_mask_env = os.environ.get("HB_LIBSMCTRL_EMBED_TPC_MASK")
        if decode_mask_env or embed_mask_env:
            import ctypes as _ct
            so_path = os.environ.get(
                "HB_LIBSMCTRL_SO",
                "/n/home07/tlebryk1/heterobatchvllm/scratch/libsmctrl/libsmctrl.so",
            )
            _libsm = _ct.CDLL(so_path)
            _libsm.libsmctrl_set_stream_mask.argtypes = [
                _ct.c_void_p, _ct.c_uint64
            ]
            _libsm.libsmctrl_set_stream_mask.restype = None
            # 128-bit mask path for GPUs with >64 TPCs (e.g. H200 has 66).
            # libsmctrl declares this as `void f(void*, unsigned __int128)`. On
            # SysV x86_64 the __int128 is passed in two consecutive integer
            # registers (RSI:RDX after the void* in RDI), so we declare two
            # c_uint64 args at the Python level -- the ABI matches.
            _libsm.libsmctrl_set_stream_mask_ext.argtypes = [
                _ct.c_void_p, _ct.c_uint64, _ct.c_uint64
            ]
            _libsm.libsmctrl_set_stream_mask_ext.restype = None
            _UINT64_MAX = (1 << 64) - 1

            def _parse(v):
                return int(v, 16) if v.startswith("0x") else int(v)

            def _apply_mask(label: str, stream_handle: int, m: int) -> None:
                if m > _UINT64_MAX:
                    lo = m & _UINT64_MAX
                    hi = (m >> 64) & _UINT64_MAX
                    _libsm.libsmctrl_set_stream_mask_ext(
                        _ct.c_void_p(stream_handle),
                        _ct.c_uint64(lo), _ct.c_uint64(hi),
                    )
                    print(f"[libsmctrl] {label} mask_ext = "
                          f"0x{hi:016x}{lo:016x}")
                else:
                    _libsm.libsmctrl_set_stream_mask(
                        _ct.c_void_p(stream_handle), _ct.c_uint64(m),
                    )
                    print(f"[libsmctrl] {label} mask = 0x{m:016x}")

            if decode_mask_env:
                # Replace default stream with a fresh one before masking
                if self.decode_stream == torch.cuda.default_stream(device):
                    new_decode = torch.cuda.Stream(device, priority=0)
                    self.decode_stream = new_decode
                    print(f"[libsmctrl] decode_stream upgraded "
                          f"from default to fresh stream (cuda_stream="
                          f"{self.decode_stream.cuda_stream:#x})")
                m = _parse(decode_mask_env)
                _apply_mask("decode_stream",
                            self.decode_stream.cuda_stream, m)
                # Route decode-model unquantized linears through a cuBLASLt
                # GEMM with CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET pinned, so the
                # Hopper GEMM does not deadlock when the mask exposes <32 TPC.
                # Gated by HB_DECODE_SM_COUNT_TARGET (no-op when unset/0). The
                # monkeypatch is decode-stream-gated, so embed is untouched.
                from vllm.v1.worker.decode_sm_linear_hook import (
                    register_decode_stream,
                )
                register_decode_stream(self.decode_stream.cuda_stream)
            if embed_mask_env and self.embed_stream is not self.decode_stream:
                m = _parse(embed_mask_env)
                _apply_mask("embed_stream ",
                            self.embed_stream.cuda_stream, m)

        # Optional: bind both streams to disjoint CUDA green contexts to give
        # decode/embed hard SM partitions (constrains cuBLAS too, unlike
        # FA3 sm_margin which is FA3-only). Requires --enforce-eager.
        if int(os.environ.get("VLLM_DUAL_MODEL_GREEN_CTX_DECODE_SM", "0")) > 0:
            import sys
            sys.path.insert(0, os.environ.get(
                "VLLM_DUAL_MODEL_GREEN_CTX_HELPER_DIR",
                "/mnt/weka/theo/heterogenious-batching-vllm/scripts",
            ))
            from green_ctx_helper import maybe_make_streams
            d, e, info = maybe_make_streams(
                device, self.decode_stream, self.embed_stream
            )
            self.decode_stream = d
            self.embed_stream = e
            self._green_ctx_info = info

        # NOTE: tried labeling the streams in nsys-ui via
        # nvtxNameCudaStreamA/CuStreamA — nsys 2025.6 ignores those events
        # from the system libnvToolsExt. In nsys-ui, identify streams via
        # the NVTX ranges that live on them: kernels under `decode_execute`
        # are the decode stream, kernels under `embed_execute`/`embed_pool`
        # are the embed stream. You can right-click → Rename a swim lane to
        # make this persistent for one report.

        execute_order = os.environ.get(
            "VLLM_DUAL_MODEL_EXECUTE_ORDER", "embed_concurrent"
        )
        if execute_order not in (
            "decode_first", "embed_first", "embed_concurrent",
            "persistent_workers",
        ):
            raise ValueError(
                "VLLM_DUAL_MODEL_EXECUTE_ORDER must be 'decode_first', "
                "'embed_first', 'embed_concurrent', or 'persistent_workers', "
                f"got {execute_order!r}."
            )
        self.execute_order = execute_order
        # Persistent worker threads (one per model) -- created lazily to keep
        # the existing execute paths zero-cost when this mode isn't selected.
        # When enabled, each persistent worker owns its model's CUDA stream
        # and handles BOTH execute_model and the post-forward step
        # (decode_sample for the decode worker, embed_pool for the embed
        # worker). This lets decode_sample + embed_pool overlap on the GPU
        # (different streams) and eliminates the ~1 ms per-step Python thread
        # spawn the old embed_concurrent path paid.
        self._decode_worker: _PersistentWorker | None = None
        self._embed_worker: _PersistentWorker | None = None
        if execute_order == "persistent_workers":
            self._decode_worker = _PersistentWorker(
                "decode", self.decode_stream, device
            )
            # Reuse the same worker for both streams if they alias (default
            # stream mode) -- no concurrency to gain, but keeps the dispatch
            # code uniform.
            if self.embed_stream is self.decode_stream:
                self._embed_worker = self._decode_worker
            else:
                self._embed_worker = _PersistentWorker(
                    "embed", self.embed_stream, device
                )
        self.async_outputs = os.environ.get(
            "VLLM_DUAL_MODEL_ASYNC_OUTPUTS", "0"
        ).lower() in ("1", "true", "yes", "on")

        dual_cfg = DualModelConfig.from_vllm_config(vllm_config)
        if dual_cfg is None:
            raise ValueError(
                "DualModelRunner requires additional_config['dual_model']."
            )

        self.vllm_config = vllm_config
        self.dual_cfg = dual_cfg
        self.device = device
        self._fuse_kv_cache = bool(dual_cfg.fuse_kv_cache)
        if self._fuse_kv_cache and vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Fused KV cache requires --no-enable-prefix-caching. With "
                "decode/embed sharing physical KV memory, a same-prefix cache "
                "hit across models would read the wrong model's K/V."
            )
        self.decode_runner = GPUModelRunner(vllm_config, device)
        self.embed_vllm_config = self._build_embed_vllm_config(vllm_config, dual_cfg)
        self.embed_runner = GPUModelRunner(self.embed_vllm_config, device)

        self.model_memory_usage = 0
        self.is_pooling_model = False
        self.model: nn.Module | None = None
        self.req_id_to_model_id: dict[str, str] = {}
        self.pending_embed_output: ModelRunnerOutput | None = None
        self.pending_embed_needs_pool = False
        self.pending_req_order: list[str] = []
        self.decode_kv_group_indices: tuple[int, ...] = ()
        self.embed_kv_group_indices: tuple[int, ...] = ()
        self._graph_both_warned = False
        self.decode_kv_cache_spec: dict[str, Any] = {}
        self.embed_kv_cache_spec: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        decode_runner = self.__dict__.get("decode_runner")
        if decode_runner is not None:
            return getattr(decode_runner, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @staticmethod
    def _replace_dataclass(instance: Any, **updates: Any) -> Any:
        cls = type(instance)
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass.")
        init_kwargs: dict[str, Any] = {}
        for name, parameter in inspect.signature(cls).parameters.items():
            if name == "self":
                continue
            if name in updates:
                init_kwargs[name] = updates.pop(name)
            elif hasattr(instance, name):
                init_kwargs[name] = getattr(instance, name)
            elif parameter.default is inspect._empty:
                raise ValueError(
                    f"Missing required constructor argument {name!r} for "
                    f"{cls.__name__}."
                )
        init_kwargs.update(updates)
        return cls(**init_kwargs)

    def _build_embed_vllm_config(
        self,
        vllm_config: VllmConfig,
        dual_cfg: DualModelConfig,
    ) -> VllmConfig:
        # The embed runner must own an independent config tree so model-load
        # side effects such as static_forward_context registration do not alias
        # the decode runner's state.
        embed_vllm_config = copy.deepcopy(vllm_config)
        embed_model_config = self._replace_dataclass(
            embed_vllm_config.model_config,
            model=dual_cfg.embed_model,
            tokenizer=dual_cfg.embed_tokenizer or dual_cfg.embed_model,
            runner="pooling",
            served_model_name=dual_cfg.embed_model,
            dtype=dual_cfg.embed_dtype or embed_vllm_config.model_config.dtype,
            max_model_len=(
                dual_cfg.embed_max_model_len
                or embed_vllm_config.model_config.max_model_len
            ),
            enforce_eager=(
                dual_cfg.embed_enforce_eager
                if dual_cfg.embed_enforce_eager is not None
                else embed_vllm_config.model_config.enforce_eager
            ),
            model_weights="",
            hf_config_path=None,
        )
        embed_scheduler_config = self._replace_dataclass(
            embed_vllm_config.scheduler_config,
            max_model_len=embed_model_config.max_model_len,
            is_encoder_decoder=embed_model_config.is_encoder_decoder,
            runner_type="pooling",
        )
        embed_vllm_config.model_config = embed_model_config
        embed_vllm_config.scheduler_config = embed_scheduler_config
        embed_vllm_config.additional_config = {}
        embed_vllm_config.compilation_config.static_forward_context.clear()
        embed_vllm_config.compilation_config.static_all_moe_layers.clear()
        # If embed_enforce_eager is set OR a delta bundle is in use, also
        # disable torch.compile / Inductor for the embed runner. The model_config
        # eager flag alone doesn't suppress compilation_config.mode, and
        # forward_pre_hooks that re-bind weight.data don't reach AOT-captured
        # parameter inputs inside the compiled graph.
        if (
            dual_cfg.embed_enforce_eager is True
            or dual_cfg.embed_delta_bundle is not None
        ):
            from vllm.config.compilation import CompilationMode, CUDAGraphMode

            embed_vllm_config.compilation_config.mode = CompilationMode.NONE
            embed_vllm_config.compilation_config.cudagraph_mode = (
                CUDAGraphMode.NONE
            )
        return embed_vllm_config

    def update_max_model_len(self, max_model_len: int) -> None:
        self.decode_runner.update_max_model_len(max_model_len)
        self.embed_runner.update_max_model_len(max_model_len)

    @staticmethod
    def _project_kv_cache_config(
        kv_cache_config: KVCacheConfig,
        runner_kv_cache_spec: dict[str, Any],
    ) -> tuple[KVCacheConfig, tuple[int, ...]]:
        runner_layer_names = set(runner_kv_cache_spec)
        projected_groups: list[KVCacheGroupSpec] = []
        group_indices: list[int] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            runner_group_layers = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in runner_layer_names
            ]
            if not runner_group_layers:
                continue
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={
                        layer_name: group_spec.kv_cache_specs[layer_name]
                        for layer_name in runner_group_layers
                    },
                )
            projected_groups.append(
                KVCacheGroupSpec(
                    layer_names=runner_group_layers,
                    kv_cache_spec=group_spec,
                )
            )
            group_indices.append(group_idx)

        projected_tensors = [
            KVCacheTensor(
                size=kv_cache_tensor.size,
                shared_by=[
                    layer_name
                    for layer_name in kv_cache_tensor.shared_by
                    if layer_name in runner_layer_names
                ],
            )
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
            if any(
                layer_name in runner_layer_names
                for layer_name in kv_cache_tensor.shared_by
            )
        ]
        return (
            KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=projected_tensors,
                kv_cache_groups=projected_groups,
            ),
            tuple(group_indices),
        )

    def get_supported_tasks(self) -> tuple[str, ...]:
        tasks = list(self.decode_runner.get_supported_tasks())
        for task in self.embed_runner.get_supported_tasks():
            if task not in tasks:
                tasks.append(task)
        return tuple(tasks)

    def load_model(self, *args, **kwargs) -> None:
        # Per-MODEL FA3 sm_margin: the DECODE model's attention backends capture
        # HB_FA3_MODEL_SM_MARGIN at build; the EMBED model's are forced to 0 by
        # temporarily clearing the env during embed build (build is single-threaded
        # at startup, so this toggle is race-free). This is the ONLY way to margin
        # decode-flash without touching embed-flash, since BOTH models are causal
        # (so the causal-keyed VLLM_FA3_DECODE_SM_MARGIN hits both).
        import os as _os
        self.decode_runner.load_model(*args, **kwargs)
        _save = _os.environ.get("HB_FA3_MODEL_SM_MARGIN")
        _os.environ["HB_FA3_MODEL_SM_MARGIN"] = "0"
        try:
            self.embed_runner.load_model(
                *args,
                prefix=self.EMBED_MODEL_PREFIX,
                **kwargs,
            )
        finally:
            if _save is None:
                _os.environ.pop("HB_FA3_MODEL_SM_MARGIN", None)
            else:
                _os.environ["HB_FA3_MODEL_SM_MARGIN"] = _save

        # Tier C: optionally swap embed transformer-block bf16 weights for
        # int8-Δ storage so vLLM's KV-cache profile sees a smaller embed
        # footprint and grows the KV pool. Forward hooks materialize bf16
        # weight on demand for each Linear forward.
        bundle_path = self.dual_cfg.embed_delta_bundle
        if bundle_path is not None:
            from pathlib import Path as _Path

            from vllm.v1.worker.embed_delta_storage import (
                apply_embed_delta_storage,
            )

            import torch as _torch

            mem_before = _torch.cuda.memory_allocated()
            self._embed_delta_summary = apply_embed_delta_storage(
                self.embed_runner.model,
                self.decode_runner.model,
                _Path(bundle_path),
            )
            mem_after = _torch.cuda.memory_allocated()
            self._embed_delta_summary["torch_mem_freed_GB"] = (
                (mem_before - mem_after) / 1e9
            )
            # Reflect saved bytes in vLLM's accounting so cudagraph / KV
            # profiles size correctly.
            freed_bytes = int(self._embed_delta_summary["bytes_freed_GB"] * 1e9)
            self.embed_runner.model_memory_usage = max(
                0, self.embed_runner.model_memory_usage - freed_bytes
            )

        self.decode_kv_cache_spec = self.decode_runner.get_kv_cache_spec()
        self.embed_kv_cache_spec = self.embed_runner.get_kv_cache_spec()
        overlap = set(self.decode_kv_cache_spec) & set(self.embed_kv_cache_spec)
        if overlap:
            raise ValueError(
                "Decode/embed KV layer names must be disjoint; overlapping names: "
                f"{sorted(overlap)[:4]}"
            )
        self.model_memory_usage = (
            self.decode_runner.model_memory_usage
            + self.embed_runner.model_memory_usage
        )
        self.model = self.decode_runner.model

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self):
        if self._fuse_kv_cache:
            # Hide embed layers from vLLM's global KV planner so num_blocks is
            # computed for one model's worth of layers (~2x larger pool).
            # Embed reuses decode's physical tensors via initialize_kv_cache.
            return dict(self.decode_kv_cache_spec)
        return {
            **self.decode_kv_cache_spec,
            **self.embed_kv_cache_spec,
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        if self._fuse_kv_cache:
            self._initialize_kv_cache_fused(kv_cache_config)
            return
        decode_kv_cache_config, self.decode_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.decode_kv_cache_spec)
        )
        embed_kv_cache_config, self.embed_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.embed_kv_cache_spec)
        )
        self.decode_runner.initialize_kv_cache(decode_kv_cache_config)
        self.embed_runner.initialize_kv_cache(embed_kv_cache_config)

    def _initialize_kv_cache_fused(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate one physical KV pool and alias it across decode + embed.

        Requires identical per-layer KV shape (block_size, num_kv_heads,
        head_size, dtype, page_size) on both runners — enforced implicitly by
        the shared global config that vLLM built from decode's spec only.
        """
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.worker.gpu import attn_utils as _attn_utils

        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError(
                "Fused KV cache expects a single global KV group; got "
                f"{len(kv_cache_config.kv_cache_groups)}."
            )
        decode_group = kv_cache_config.kv_cache_groups[0]
        decode_layer_names = list(decode_group.layer_names)
        embed_layer_names = list(self.embed_kv_cache_spec.keys())
        if len(decode_layer_names) != len(embed_layer_names):
            raise ValueError(
                "Fused KV cache requires equal decode/embed layer counts; "
                f"got {len(decode_layer_names)} decode vs "
                f"{len(embed_layer_names)} embed."
            )

        decode_sorted = sorted(decode_layer_names, key=extract_layer_index)
        embed_sorted = sorted(embed_layer_names, key=extract_layer_index)
        decode_to_embed = dict(zip(decode_sorted, embed_sorted))
        embed_to_decode = {v: k for k, v in decode_to_embed.items()}

        decode_spec = next(iter(self.decode_kv_cache_spec.values()))
        embed_spec = next(iter(self.embed_kv_cache_spec.values()))
        if decode_spec.page_size_bytes != embed_spec.page_size_bytes:
            raise ValueError(
                "Fused KV cache requires equal page_size_bytes; got "
                f"decode={decode_spec.page_size_bytes} vs "
                f"embed={embed_spec.page_size_bytes}."
            )

        # The V2 model runner allocates raw KV tensors via the module-level
        # function `vllm.v1.worker.gpu.attn_utils._allocate_kv_cache`. We
        # swap it temporarily to (a) capture decode's int8 buffers, then (b)
        # hand those same buffers to the embed runner. Both runners' per-
        # layer kv_caches alias the same GPU memory after init.
        original_alloc = _attn_utils._allocate_kv_cache
        saved_raw: dict[str, torch.Tensor] = {}

        def _capture_decode_alloc(cfg, device):
            result = original_alloc(cfg, device)
            saved_raw.update(result)
            return result

        _attn_utils._allocate_kv_cache = _capture_decode_alloc
        try:
            self.decode_runner.initialize_kv_cache(kv_cache_config)
        finally:
            _attn_utils._allocate_kv_cache = original_alloc

        # Build embed-side config that mirrors decode's shape but references
        # embed layer names.
        embed_kv_cache_tensors = [
            KVCacheTensor(
                size=tensor.size,
                shared_by=[decode_to_embed[n] for n in tensor.shared_by],
            )
            for tensor in kv_cache_config.kv_cache_tensors
        ]
        embed_kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=[
                    decode_to_embed[n] for n in decode_group.layer_names
                ],
                kv_cache_spec=decode_group.kv_cache_spec,
            )
        ]
        embed_kv_cache_config = KVCacheConfig(
            num_blocks=kv_cache_config.num_blocks,
            kv_cache_tensors=embed_kv_cache_tensors,
            kv_cache_groups=embed_kv_cache_groups,
        )

        def _reuse_embed_alloc(cfg, device):
            result: dict[str, torch.Tensor] = {}
            for kv_tensor in cfg.kv_cache_tensors:
                for layer_name in kv_tensor.shared_by:
                    decode_name = embed_to_decode[layer_name]
                    result[layer_name] = saved_raw[decode_name]
            return result

        _attn_utils._allocate_kv_cache = _reuse_embed_alloc
        try:
            self.embed_runner.initialize_kv_cache(embed_kv_cache_config)
        finally:
            _attn_utils._allocate_kv_cache = original_alloc

        # Both runners read block_ids from the single global KV group.
        self.decode_kv_group_indices = (0,)
        self.embed_kv_group_indices = (0,)

    def profile_run(self) -> None:
        self.decode_runner.profile_run()
        self.embed_runner.profile_run()

    def reset_mm_cache(self) -> None:
        self.decode_runner.reset_mm_cache()
        self.embed_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.decode_runner.reset_encoder_cache()
        self.embed_runner.reset_encoder_cache()

    def profile_cudagraph_memory(self) -> int:
        return (
            self.decode_runner.profile_cudagraph_memory()
            + self.embed_runner.profile_cudagraph_memory()
        )

    def capture_model(self) -> int:
        return (
            self.decode_runner.capture_model()
            + self.embed_runner.capture_model()
        )

    def _dummy_run(self, *args, **kwargs):
        return self.decode_runner._dummy_run(*args, **kwargs)

    def _dummy_sampler_run(self, *args, **kwargs) -> None:
        self.decode_runner._dummy_sampler_run(*args, **kwargs)

    def _dummy_pooler_run(self, *args, **kwargs) -> None:
        self.embed_runner._dummy_pooler_run(*args, **kwargs)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ):
        if dummy_run:
            self.decode_runner.execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            )
            self.embed_runner.profile_run()
            return None

        split_outputs = split_scheduler_output_by_model(
            scheduler_output=scheduler_output,
            req_id_to_model_id=self.req_id_to_model_id,
            decode_model_id=self.dual_cfg.decode_model_id,
            embed_model_id=self.dual_cfg.embed_model_id,
            decode_kv_group_indices=self.decode_kv_group_indices,
            embed_kv_group_indices=self.embed_kv_group_indices,
        )
        self.pending_req_order = split_outputs.scheduled_req_order

        def run_decode() -> ModelRunnerOutput | None:
            with torch.cuda.stream(self.decode_stream):
                with _nvtx_range("decode_execute"):
                    return self.decode_runner.execute_model(
                        split_outputs.decode,
                        intermediate_tensors=intermediate_tensors,
                    )

        def run_embed() -> ModelRunnerOutput | None:
            with torch.cuda.stream(self.embed_stream):
                with _nvtx_range("embed_execute"):
                    return self.embed_runner.execute_model(
                        split_outputs.embed,
                    )

        has_decode_work = (
            split_outputs.decode.num_scheduled_tokens
            or split_outputs.decode.finished_req_ids
        )
        has_embed_work = (
            split_outputs.embed.num_scheduled_tokens
            or split_outputs.embed.finished_req_ids
        )

        decode_output = None
        embed_exec_output = None
        embed_output: ModelRunnerOutput | None = None
        # Dispatch strategy is controlled by VLLM_DUAL_MODEL_EXECUTE_ORDER:
        #   embed_concurrent (default): spawn embed forward on a bg thread,
        #     run decode forward on the main thread, then join. Both Python
        #     forward calls overlap on CPU; embed kernels hit the embed
        #     stream BEFORE decode kernels hit the decode stream. Requires
        #     thread-local forward_context (see vllm/forward_context.py).
        #   embed_first: serial on main thread, embed before decode. Safe
        #     fallback if the thread-local fix is reverted.
        #   decode_first: defer embed forward to sample_tokens so embed
        #     dispatch overlaps with decode_sample. Embed kernels arrive
        #     on-stream LATER than decode kernels — only beneficial when
        #     decode is the long pole.
        will_defer_embed = (
            has_embed_work and has_decode_work
            and self.execute_order == "decode_first"
        )
        will_concurrent_embed = (
            has_embed_work and has_decode_work
            and self.execute_order == "embed_concurrent"
        )
        will_use_persistent = (
            self.execute_order == "persistent_workers"
            and self._decode_worker is not None
        )
        # T1 graph-both-streams dispatch: back-to-back single-thread replay of
        # decode forward then embed forward on their own streams, no thread
        # spawn / join / cross-stream sync. Only meaningful (densifying) under
        # FULL cudagraphs but always correct. Overrides execute_order.
        will_graph_both = (
            _dual_graph_both_streams() and has_decode_work and has_embed_work
        )

        if will_graph_both:
            self._warn_if_not_full_graphs()
            # Single host thread: enqueue decode-graph replay on decode_stream,
            # then IMMEDIATELY enqueue embed-graph replay on embed_stream. Each
            # forward is one replay() (FULL graph) so the host enqueues both in
            # microseconds with no GIL ping-pong and no intervening sync; both
            # stream queues are full at once for the GPU to co-issue.
            if has_decode_work:
                decode_output = run_decode()
            if has_embed_work:
                embed_exec_output = run_embed()
        elif will_use_persistent:
            # Persistent worker dispatch. Each worker runs forever on its own
            # thread + CUDA stream; submissions are queued and overlap on the
            # GPU because the streams are independent. No per-step thread
            # spawn (the old embed_concurrent path paid ~1 ms here).
            decode_evt = decode_res = None
            embed_evt = embed_res = None
            if has_decode_work:
                decode_evt, decode_res = self._decode_worker.submit(run_decode)
            if has_embed_work:
                embed_evt, embed_res = self._embed_worker.submit(run_embed)
            if has_decode_work:
                decode_output = _PersistentWorker.wait(decode_evt, decode_res)
            if has_embed_work:
                embed_exec_output = _PersistentWorker.wait(embed_evt, embed_res)
        elif will_concurrent_embed:
            embed_holder: list[Any] = [None]

            def _run_embed_thread() -> None:
                embed_holder[0] = run_embed()

            embed_thread = threading.Thread(
                target=_run_embed_thread, daemon=True
            )
            with _nvtx_range("embed_thread_start"):
                embed_thread.start()
            decode_output = run_decode()
            with _nvtx_range("embed_thread_join"):
                embed_thread.join()
            embed_exec_output = embed_holder[0]
        elif self.execute_order == "embed_first":
            if has_embed_work:
                embed_exec_output = run_embed()
            if has_decode_work:
                decode_output = run_decode()
        else:
            if has_decode_work:
                decode_output = run_decode()
            if has_embed_work and not will_defer_embed:
                embed_exec_output = run_embed()

        decode_needs_sample = (
            decode_output is None and bool(split_outputs.decode.num_scheduled_tokens)
        )
        if decode_needs_sample:
            self.pending_embed_output = embed_exec_output
            self.pending_embed_needs_pool = (
                split_outputs.embed.num_scheduled_tokens
                or split_outputs.embed.finished_req_ids
            ) and embed_exec_output is None and not will_defer_embed
            self.pending_embed_split_output = (
                split_outputs.embed if will_defer_embed else None
            )
            self.pending_embed_needs_execute = will_defer_embed
            return None

        embed_output_unresolved: ModelRunnerOutput | AsyncModelRunnerOutput | None = (
            None
        )
        if has_embed_work:
            if embed_exec_output is None:
                with torch.cuda.stream(self.embed_stream):
                    with _nvtx_range("embed_pool"):
                        with _temporary_async_outputs(
                            self.embed_runner,
                            self.async_outputs or self.use_async_scheduling,
                        ):
                            embed_pool_output = self.embed_runner.pool()
                embed_output_unresolved = embed_pool_output
            else:
                embed_output_unresolved = embed_exec_output

        self.pending_embed_output = None
        self.pending_embed_needs_pool = False

        if self.use_async_scheduling:
            # Defer the merge: DualAsyncModelRunnerOutput.get_output() runs on
            # the executor's async output thread and resolves both children
            # there, overlapping with the engine's next-step CPU work.
            return DualAsyncModelRunnerOutput(
                scheduled_req_order=self.pending_req_order,
                decode_output=decode_output,
                embed_output=embed_output_unresolved,
            )

        embed_output = _resolve_model_runner_output(
            embed_output_unresolved,
            "embed_output_get",
        )
        if has_embed_work and embed_output is None:
            raise RuntimeError("Embed runner failed to produce pooling output.")
        with _nvtx_range("merge_outputs"):
            return merge_model_runner_outputs(
                scheduled_req_order=self.pending_req_order,
                decode_output=decode_output,
                embed_output=embed_output,
            )

    def sample_tokens(self, grammar_output: GrammarOutput | None):
        # The deferred decode-sample D2H (async DualAsyncModelRunnerOutput
        # return) requires the executor's 2-deep batch queue + async output
        # thread (mcb=2) to resolve get_output() off the engine hot thread; it
        # is gated SOLELY by HB_DEFER_DECODE_SAMPLE_D2H. HB_DUAL_GRAPH_BOTH_STREAMS
        # by itself runs at mcb=1 (see uniproc_executor.max_concurrent_batches):
        # its win is the WITHIN-step back-to-back graph replay in execute_model,
        # which is independent of the D2H defer. At mcb=1 there is no async output
        # thread, so the step path (engine_core.step) calls sample_tokens
        # BLOCKING and the scheduler needs a resolved ModelRunnerOutput; returning
        # a DualAsyncModelRunnerOutput would never get resolved. So only defer
        # (and only then return the async wrapper) when DEFER is explicitly set.
        defer_d2h = _defer_decode_sample_d2h()
        # When deferring the decode D2H, the DECODE child runs in async-output
        # mode so its sample returns an AsyncModelRunnerOutput (no inline
        # blocking transfer_event.synchronize); we resolve that decode output
        # LAST, after the embed pool has been dispatched on the embed stream, so
        # the decode token D2H drain overlaps embed GPU work. The embed child is
        # left in its normal (possibly sync) mode; only decode is forced async.
        async_outputs_for_children = self.async_outputs or self.use_async_scheduling
        decode_async_child = async_outputs_for_children or defer_d2h

        # ---- persistent_workers path ----
        # Submit decode_sample to the decode worker (its own stream + thread)
        # and embed_pool to the embed worker (its own stream + thread) in
        # parallel. Both finish concurrently; main thread waits for both then
        # merges. This eliminates the ~0.4 ms of serializing embed_pool after
        # decode_sample that the legacy path paid.
        if (
            self.execute_order == "persistent_workers"
            and self._decode_worker is not None
            and not _dual_graph_both_streams()
        ):
            def _do_decode_sample():
                with _nvtx_range("decode_sample"):
                    with _temporary_async_outputs(
                        self.decode_runner, decode_async_child
                    ):
                        # Stream context is set by the worker's _loop wrapper
                        # too, but be explicit here so this closure is correct
                        # regardless of who invokes it.
                        with torch.cuda.stream(self.decode_stream):
                            return self.decode_runner.sample_tokens(grammar_output)

            decode_evt, decode_res = self._decode_worker.submit(_do_decode_sample)

            embed_evt = embed_res = None
            if self.pending_embed_needs_pool:
                def _do_embed_pool():
                    with _nvtx_range("embed_pool"):
                        with _temporary_async_outputs(
                            self.embed_runner, async_outputs_for_children
                        ):
                            with torch.cuda.stream(self.embed_stream):
                                return self.embed_runner.pool()
                embed_evt, embed_res = self._embed_worker.submit(_do_embed_pool)

            decode_sample_output = _PersistentWorker.wait(decode_evt, decode_res)
            if embed_evt is not None:
                embed_output_unresolved = _PersistentWorker.wait(embed_evt, embed_res)
            else:
                embed_output_unresolved = self.pending_embed_output

            scheduled_req_order = self.pending_req_order
            self.pending_embed_output = None
            self.pending_embed_needs_pool = False
            self.pending_req_order = []

            if self.use_async_scheduling or defer_d2h:
                # Defer decode token-id D2H + merge to get_output(), which the
                # executor runs on its WorkerAsyncOutput thread (because the
                # batch queue / non_block=True path is enabled under the flag).
                # This overlaps step N's D2H drain with step N+1's CPU dispatch
                # while decode/embed keep their separate streams. NOT engine-wide
                # async scheduling (scheduler stays sync; the 2-deep batch queue
                # pairs each output with its own scheduler_output).
                return DualAsyncModelRunnerOutput(
                    scheduled_req_order=scheduled_req_order,
                    decode_output=decode_sample_output,
                    embed_output=embed_output_unresolved,
                )

            decode_output = _resolve_model_runner_output(
                decode_sample_output, "decode_output_get",
            )
            embed_output = _resolve_model_runner_output(
                embed_output_unresolved, "embed_output_get",
            )
            with _nvtx_range("merge_outputs"):
                return merge_model_runner_outputs(
                    scheduled_req_order=scheduled_req_order,
                    decode_output=decode_output,
                    embed_output=embed_output,
                )

        # ---- legacy paths below (embed_concurrent / decode_first / embed_first) ----
        # If execute_model deferred embed (decode_first mode), start the
        # embed forward NOW on a background thread, then run decode_sample
        # on the main thread. The two overlap: embed forward on
        # self.embed_stream, decode_sample on self.decode_stream.
        # Safe under thread-local forward_context (see
        # vllm/forward_context.py) regardless of dispatch order; the
        # decode_first path keeps embed_dispatch hidden under decode_sample
        # CPU cost, useful when decode is the long pole.
        embed_thread = None
        embed_holder: list[Any] = [None]
        if getattr(self, "pending_embed_needs_execute", False):
            deferred_split = self.pending_embed_split_output

            def _run_embed_bg() -> None:
                with torch.cuda.stream(self.embed_stream):
                    with _nvtx_range("embed_execute_threaded"):
                        embed_holder[0] = self.embed_runner.execute_model(
                            deferred_split
                        )

            embed_thread = threading.Thread(target=_run_embed_bg, daemon=True)
            embed_thread.start()

        with _nvtx_range("decode_sample"):
            with _temporary_async_outputs(
                self.decode_runner, decode_async_child
            ):
                # Run sample on the decode stream so the sampler is in-stream
                # with the model forward (which produced hidden_states on the
                # decode stream). Without this wrapper sample_tokens would run
                # on the caller's stream (the engine default stream), which
                # creates a cross-stream race that fires asynchronously as a
                # device-side assert under FA3 + green-ctx mode=both.
                # With defer_d2h, decode_async_child forces async output here so
                # the blocking token-id D2H sync is deferred to get_output()
                # below, AFTER embed_pool is dispatched (overlap, not a stall).
                with torch.cuda.stream(self.decode_stream):
                    decode_sample_output = self.decode_runner.sample_tokens(grammar_output)

        # Join the background embed thread (was started before decode_sample).
        # If decode_sample finished first, we wait a bit; if embed finished
        # first, this returns immediately.
        if embed_thread is not None:
            embed_thread.join()
            self.pending_embed_needs_execute = False
            self.pending_embed_split_output = None

        embed_output_unresolved: ModelRunnerOutput | AsyncModelRunnerOutput | None = (
            embed_holder[0] if embed_holder[0] is not None
            else self.pending_embed_output
        )
        if self.pending_embed_needs_pool:
            with torch.cuda.stream(self.embed_stream):
                with _nvtx_range("embed_pool"):
                    with _temporary_async_outputs(
                        self.embed_runner, async_outputs_for_children
                    ):
                        embed_output_unresolved = self.embed_runner.pool()

        scheduled_req_order = self.pending_req_order
        self.pending_embed_output = None
        self.pending_embed_needs_pool = False
        self.pending_req_order = []

        if self.use_async_scheduling or defer_d2h:
            # Defer decode token-id D2H + merge to get_output() on the executor's
            # WorkerAsyncOutput thread (batch-queue / non_block path enabled by
            # the flag). Overlaps step N D2H drain with step N+1 dispatch; streams
            # stay separate; scheduler stays synchronous.
            return DualAsyncModelRunnerOutput(
                scheduled_req_order=scheduled_req_order,
                decode_output=decode_sample_output,
                embed_output=embed_output_unresolved,
            )

        decode_output = _resolve_model_runner_output(
            decode_sample_output,
            "decode_output_get",
        )
        embed_output = _resolve_model_runner_output(
            embed_output_unresolved,
            "embed_output_get",
        )
        if self.pending_embed_needs_pool and embed_output is None:
            raise RuntimeError("Embed runner failed to produce pooling output.")
        with _nvtx_range("merge_outputs"):
            return merge_model_runner_outputs(
                scheduled_req_order=scheduled_req_order,
                decode_output=decode_output,
                embed_output=embed_output,
            )

    def _warn_if_not_full_graphs(self) -> None:
        """One-time check that both child runners actually have FULL graphs.

        HB_DUAL_GRAPH_BOTH_STREAMS only densifies the streams if each forward is
        a single graph replay. If either runner captured no FULL graphs (e.g.
        ran eager / --enforce-eager), the dispatch is still correct but each
        forward is ~495 host launches with no dense window for overlap; warn so
        the measurement isn't misread.
        """
        if self._graph_both_warned:
            return
        self._graph_both_warned = True

        def _has_full_graphs(runner: GPUModelRunner) -> bool:
            mgr = getattr(runner, "cudagraph_manager", None)
            graphs = getattr(mgr, "graphs", None)
            return bool(graphs)

        decode_ok = _has_full_graphs(self.decode_runner)
        embed_ok = _has_full_graphs(self.embed_runner)
        print(
            "[HB_DUAL_GRAPH_BOTH_STREAMS] active: "
            f"decode_full_graphs={decode_ok} embed_full_graphs={embed_ok}",
            flush=True,
        )
        if not (decode_ok and embed_ok):
            print(
                "[HB_DUAL_GRAPH_BOTH_STREAMS] WARNING: at least one runner has "
                "no captured FULL cudagraph; its forward is still ~495 host "
                "launches (no dense window). Run with --no-enforce-eager "
                "--cudagraph-mode FULL for the intended densification.",
                flush=True,
            )

    def take_draft_token_ids(self):
        return self.decode_runner.take_draft_token_ids()
