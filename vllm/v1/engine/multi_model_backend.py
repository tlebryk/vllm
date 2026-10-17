# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arrival-driven, single-owner backend for native async model clients."""

import asyncio
import json
import logging
import queue
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from vllm import EngineArgs
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.model_masks import ModelLaneMasks, ModelMaskPlan, library_provenance
from vllm.v1.engine.model_streams import ModelDecodeStreams
from vllm.v1.engine.multi_model_config import load_specs
from vllm.v1.engine.multi_model_controller import ModelEndpoint, MultiModelController

logger = logging.getLogger(__name__)


def prepare_models(path, *, mode, budget, decode_depth):
    """Validate model configurations before starting the CUDA owner thread."""
    specs = load_specs(path if isinstance(path, (list, tuple)) else Path(path), 0, 0)
    if len(specs) == 1 and mode != "private":
        raise ValueError("single-model native serving requires private KV")
    if mode != "private" and any(
        s.kv_cache_memory_bytes is None or s.dtype == "auto" for s in specs
    ):
        raise ValueError("auto-sized KV/dtype currently require private KV")
    prepared = []
    for spec in specs:
        config = EngineArgs(
            **spec.engine_kwargs(eager=False, decode_depth=decode_depth)
        ).create_engine_config()
        if (
            config.model_config.hf_config.architectures
            not in (
                ["LlamaForCausalLM"],
                ["Qwen3ForCausalLM"],
                ["LlamaForSequenceClassification"],
                ["Qwen3ForSequenceClassification"],
            )
            or config.model_config.quantization is not None
        ):
            raise ValueError(
                "native multi-model serving supports unquantized dense Llama/Qwen3"
            )
        config.model_config.served_model_name = spec.model_id
        if spec.runner == "pooling":
            from vllm.config.compilation import CUDAGraphMode

            if config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                raise ValueError("native pooling requires cudagraph_mode NONE")
        prepared.append(config)
    if mode not in ("layer", "head", "private"):
        raise ValueError("invalid shared KV layout")
    if budget is not None and (budget <= 0 or mode == "private"):
        raise ValueError("explicit shared budget requires a positive head/layer pool")
    return specs, prepared, budget or sum(s.kv_cache_memory_bytes or 0 for s in specs)


@dataclass(eq=False)
class Submission:
    model_id: str
    request: object
    cancelled: bool = False
    started: bool = False


class MultiModelBackend:
    """All core, allocator and GPU work belongs to one dedicated native thread.

    Data queues are bounded. Cancellation/failure/shutdown use an independent
    signal, so a saturated output queue cannot prevent control-plane progress.
    """

    def __init__(
        self,
        specs,
        configs,
        *,
        mode="layer",
        budget=None,
        admission="current",
        topology="three-stream-overlap",
        decode_depth=2,
        max_requests=8192,
        output_capacity=64,
        offline_serial_counts=None,
        step_trace=None,
        smctrl_mask=None,
        smctrl_lib=None,
        smctrl_min_decode_requests=1,
        dense_step_tokens=0,
        smctrl_indices=None,
        smctrl_max_dense_wait_ms=0,
        adaptive_dense_steps=False,
    ):
        if max_requests <= 0 or output_capacity <= decode_depth + 2:
            raise ValueError("invalid ingress/output capacity")
        if admission == "full-output" and mode == "private":
            raise ValueError("full-output admission requires shared KV")
        if topology not in (
            "single-stream",
            "two-stream",
            "three-stream-ordered",
            "three-stream-overlap",
        ):
            raise ValueError("invalid stream topology")
        self.specs, self.configs = specs, configs
        self.mode, self.budget, self.admission = mode, budget, admission
        self.topology, self.decode_depth = topology, decode_depth
        self.max_requests = max_requests
        self.model_order = [s.model_id for s in specs]
        self.mask_plan = ModelMaskPlan.parse(smctrl_mask) if smctrl_mask else None
        if smctrl_min_decode_requests < 1:
            raise ValueError("minimum decode requests must be positive")
        self.smctrl_min_decode_requests = smctrl_min_decode_requests
        if dense_step_tokens < 0:
            raise ValueError("dense step tokens must be nonnegative")
        self.dense_step_tokens = dense_step_tokens
        self.smctrl_indices = smctrl_indices
        self.smctrl_max_dense_wait_ms = smctrl_max_dense_wait_ms
        if adaptive_dense_steps and (self.mask_plan is None or not dense_step_tokens):
            raise ValueError(
                "adaptive dense steps require a mask and a positive dense step size"
            )
        self.adaptive_dense_steps = adaptive_dense_steps
        self.smctrl_lib = Path(smctrl_lib).resolve() if smctrl_lib else None
        self.lane_masks = None
        self.mask_metadata = None
        if self.mask_plan is not None:
            if (
                (
                    self.mask_plan.model_id not in self.model_order
                    and self.mask_plan.model_id != "*"
                )
                or self.mask_plan.lane != "prefill"
                or topology == "single-stream"
                or offline_serial_counts is not None
            ):
                raise ValueError(
                    "native masking supports concurrent model-owned prefill only"
                )
            if self.smctrl_lib is None or not self.smctrl_lib.is_file():
                raise ValueError("native masking requires --slackserve-smctrl-lib")
        if offline_serial_counts is not None and (
            not isinstance(offline_serial_counts, dict)
            or set(offline_serial_counts) != set(self.model_order)
            or any(type(n) is not int or n <= 0 for n in offline_serial_counts.values())
            or sum(offline_serial_counts.values()) > max_requests
        ):
            raise ValueError(
                "offline serial counts must name both models with positive integer "
                "counts fitting the ingress capacity"
            )
        self.offline_serial_counts = (
            dict(offline_serial_counts) if offline_serial_counts is not None else None
        )
        self.serial_index = 0
        self.serial_received = dict.fromkeys(self.model_order, 0)
        self.serial_completed = dict.fromkeys(self.model_order, 0)
        self.parked = {mid: deque() for mid in self.model_order}
        self.loop = asyncio.get_running_loop()
        self.inputs = queue.Queue(maxsize=max_requests)
        self.outputs = {s.model_id: queue.Queue(maxsize=output_capacity) for s in specs}
        self.output_ready = {s.model_id: asyncio.Event() for s in specs}
        self.lock = threading.Lock()
        self.active = {}
        self.aborts = set()
        self.wakeup = threading.Event()
        self.stop = threading.Event()
        self.ready = Future()
        self.error = None
        self.cores = {}
        self.supported_tasks = {}
        self.manager = None
        self.last_pool_state = None
        self.last_engine_state = None
        self.idle_free_blocks = {}
        self.execution_stream_ids = None
        self.step_trace = Path(step_trace) if step_trace is not None else None
        self.step_trace_file = None
        self.thread = threading.Thread(
            target=self._run, name="vllm-multi-model-core", daemon=True
        )

    async def start(self):
        self.thread.start()
        await asyncio.wrap_future(self.ready)

    def _notify_outputs(self, model_id):
        self.loop.call_soon_threadsafe(self.output_ready[model_id].set)

    def submit(self, mid, request):
        params = request.sampling_params
        pooling = getattr(request, "pooling_params", None)
        spec = next((s for s in self.specs if s.model_id == mid), None)
        if spec is None:
            raise ValueError(f"unknown model {mid!r}")
        runner = getattr(spec, "runner", "generate")
        if (
            (runner == "generate" and (params is None or pooling is not None))
            or (runner == "pooling" and (pooling is None or params is not None))
            or (params is not None and params.structured_outputs is not None)
            or (
                pooling is not None
                and pooling.task not in self.supported_tasks.get(mid, ())
            )
            or request.mm_features
            or request.prompt_embeds is not None
            or request.lora_request is not None
            or request.resumable
        ):
            raise ValueError(
                "multi-model serving supports model-local text generation/pooling "
                "without "
                "structured outputs, LoRA or streaming input"
            )
        key = mid, request.request_id
        with self.lock:
            if self.error is not None or self.stop.is_set():
                raise EngineDeadError()
            if key in self.active:
                raise ValueError("duplicate active model/request ID")
            if len(self.active) >= self.max_requests:
                raise ValueError("multi-model request queue is full; retry later")
            if (
                self.offline_serial_counts is not None
                and self.serial_received[mid] >= self.offline_serial_counts[mid]
            ):
                raise ValueError("offline serial request count already reached")
            entry = Submission(mid, request)
            self.inputs.put_nowait(entry)
            self.active[key] = entry
            self.serial_received[mid] += 1
        self.wakeup.set()

    def cancel(self, mid, ids):
        with self.lock:
            for rid in ids:
                entry = self.active.get((mid, rid))
                if entry is not None:
                    entry.cancelled = True
                    self.aborts.add(entry)
        self.wakeup.set()

    def _remove(self, entry):
        key = entry.model_id, entry.request.request_id
        with self.lock:
            if self.active.get(key) is entry:
                del self.active[key]
                self.serial_completed[entry.model_id] += 1

    def _advance_serial_gate(self, controller):
        if self.offline_serial_counts is None:
            return
        while self.serial_index < len(self.model_order):
            mid = self.model_order[self.serial_index]
            with self.lock:
                complete = self.serial_completed[mid] == self.offline_serial_counts[mid]
            # Retire all prior GPU tickets, finished-ID flushes and KV fences
            # before the next model is allowed to enter its scheduler.
            if not complete or controller.has_work():
                break
            self.parked[mid].clear()
            with self.lock:
                self.serial_index += 1

    def _has_dispatchable_parked(self):
        return self.serial_index < len(self.model_order) and bool(
            self.parked[self.model_order[self.serial_index]]
        )

    def _process_inputs(self, controller):
        with self.lock:
            cancelled, self.aborts = self.aborts, set()
        for entry in cancelled:
            if entry.started:
                self.cores[entry.model_id].abort_requests([entry.request.request_id])
                # Retain identity until native finished publication so a reused
                # ID cannot collide with an older abort/ticket still in flight.
            else:
                self._remove(entry)
        self._advance_serial_gate(controller)
        changed = bool(cancelled)
        for _ in range(64):
            current = (
                self.model_order[self.serial_index]
                if self.offline_serial_counts is not None
                and self.serial_index < len(self.model_order)
                else None
            )
            if current is not None and self.parked[current]:
                entry = self.parked[current].popleft()
            else:
                try:
                    entry = self.inputs.get_nowait()
                except queue.Empty:
                    break
            if entry.cancelled or self.stop.is_set():
                self._remove(entry)
                continue
            if self.offline_serial_counts is not None and entry.model_id != current:
                self.parked[entry.model_id].append(entry)
                continue
            core = self.cores[entry.model_id]
            request, wave = core.preprocess_add_request(entry.request)
            core.add_request(request, wave)
            entry.started = True
            changed = True
        if changed:
            controller.notify_requests_changed()

    def _publish(self, mid, outputs):
        for result in outputs.values():
            if (
                not result.outputs
                and not result.finished_requests
                and result.scheduler_stats is None
            ):
                continue
            # The loop reserves enough room for all already-in-flight tickets.
            self.outputs[mid].put_nowait(result)
            with self.lock:
                for output in result.outputs:
                    if (
                        output.finish_reason is not None
                        and self.active.pop((mid, output.request_id), None) is not None
                    ):
                        self.serial_completed[mid] += 1
                for rid in result.finished_requests or ():
                    if self.active.pop((mid, rid), None) is not None:
                        self.serial_completed[mid] += 1
            self._notify_outputs(mid)

    def _initialize(self):
        import torch

        from vllm.v1.core.shared_kv import SharedKVLayout, SharedKVManager
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor import Executor

        if self.mode != "private":
            layouts = [
                SharedKVLayout(
                    s.model_id,
                    c.model_config.get_num_layers(c.parallel_config),
                    c.model_config.get_num_kv_heads(c.parallel_config),
                    c.model_config.get_head_size(),
                    s.block_size,
                    s.dtype,
                )
                for s, c in zip(self.specs, self.configs)
            ]
            self.manager = SharedKVManager(layouts, self.budget, self.mode)
        for spec, config in zip(self.specs, self.configs):
            if self.manager is not None:
                config.additional_config["shared_kv"] = {
                    "pool_id": self.manager.pool_id,
                    "model_id": spec.model_id,
                    "mode": self.mode,
                    "budget_bytes": self.budget,
                }
                config.cache_config.kv_cache_memory_bytes = self.budget
            with (
                self.manager.startup_scope()
                if self.manager is not None
                else nullcontext()
            ):
                core = EngineCore(
                    config,
                    Executor.get_class(config),
                    log_stats=True,
                    include_finished_set=True,
                )
            self.cores[spec.model_id] = core
            self.supported_tasks[spec.model_id] = core.get_supported_tasks()
            if (
                spec.runner == "pooling"
                and spec.pooling_task not in self.supported_tasks[spec.model_id]
            ):
                raise ValueError(
                    f"model {spec.model_id} does not support {spec.pooling_task}"
                )
            if self.step_trace_file is not None:
                original_dispatch = core.dispatch

                def traced_dispatch(
                    lane, dispatch=original_dispatch, mid=spec.model_id
                ):
                    with torch.cuda.nvtx.range(f"{mid}:{lane}"):
                        return dispatch(lane)

                core.dispatch = traced_dispatch
        endpoints, stream_order = self._bind_execution_streams()
        torch.cuda.synchronize()
        self.idle_free_blocks = {
            mid: core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            for mid, core in self.cores.items()
        }
        return MultiModelController(
            endpoints,
            decode_depth=self.decode_depth,
            kv_admission=self.admission,
            independent_decode=self.topology.startswith("three-stream"),
            decode_streams=stream_order,
            record_steps=False,
            lane_masks=self.lane_masks,
            adaptive_dense_steps=self.adaptive_dense_steps,
            step_observer=self._observe_step
            if self.step_trace_file is not None
            else None,
        )

    def _observe_step(self, record):
        self.step_trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _bind_execution_streams(self):
        import torch

        serial = self.topology == "single-stream"
        prefill = torch.cuda.Stream()
        shared_decode = prefill if serial else torch.cuda.Stream()
        streams = {
            s.model_id: shared_decode
            if self.topology in ("single-stream", "two-stream")
            else torch.cuda.Stream()
            for s in self.specs
        }
        endpoints = []
        workers = {}
        for spec in self.specs:
            core = self.cores[spec.model_id]
            wrapped = core.model_executor.driver_worker
            worker = getattr(wrapped, "worker", wrapped)
            workers[spec.model_id] = worker
            worker.bind_lane_streams(prefill, streams[spec.model_id], serial=serial)
            endpoints.append(
                ModelEndpoint(
                    spec.model_id,
                    core,
                    lambda out, mid=spec.model_id: self._publish(mid, out),
                    max(
                        spec.prefill_reserve_blocks,
                        int(
                            core.scheduler.kv_cache_manager.block_pool.num_gpu_blocks
                            * spec.prefill_reserve_fraction
                        )
                        if getattr(spec, "prefill_reserve_fraction", 0)
                        else 0,
                    ),
                    self.decode_depth > 1
                    and getattr(spec, "runner", "generate") == "generate",
                    getattr(spec, "runner", "generate") == "generate",
                    spec.prefill_step_tokens
                    or (self.dense_step_tokens if spec.runner == "pooling" else 0),
                    exclusive_execution=spec.exclusive_execution,
                )
            )
        self.execution_stream_ids = {
            "prefill": prefill.cuda_stream,
            "decode": {mid: stream.cuda_stream for mid, stream in streams.items()},
        }
        if self.mask_plan is not None:
            from vllm.v1.worker.hb_smctrl import SmCtrl, spread_indices

            total = torch.cuda.get_device_properties(0).multi_processor_count // 2
            if not 0 < self.mask_plan.tpcs < total:
                raise ValueError("mask TPC count must be below full GPU")
            # Prefill is eager; decode's captured placement is left intact.
            if not all(w.model_runner.decode_graphs_only for w in workers.values()):
                raise ValueError("prefill stream masking requires eager prefill")
            indices = (
                list(self.smctrl_indices)
                if self.smctrl_indices is not None
                else spread_indices(self.mask_plan.tpcs, total)
            )
            self.lane_masks = ModelLaneMasks(
                self.mask_plan,
                endpoints,
                workers,
                {"prefill": prefill},
                total,
                SmCtrl(str(self.smctrl_lib)),
                indices,
                activation="decode",
                min_decode_requests=self.smctrl_min_decode_requests,
                max_dense_wait_ms=self.smctrl_max_dense_wait_ms,
            )
            self.mask_metadata = {
                "model_id": self.mask_plan.model_id,
                "lane": "prefill",
                "enabled_tpcs": self.mask_plan.tpcs,
                "total_tpcs": total,
                "geometry": "explicit" if self.smctrl_indices is not None else "spread",
                "enabled_indices": indices,
                "min_decode_requests": self.smctrl_min_decode_requests,
                "max_dense_wait_ms": self.smctrl_max_dense_wait_ms,
                "activation": (
                    "decode-ready resident or pending batch load reaches "
                    "min_decode_requests; pipeline tickets are not summed"
                ),
                "decode_graphs": "unchanged",
                "library": library_provenance(self.smctrl_lib),
            }
        stream_order = (
            None
            if self.topology in ("single-stream", "two-stream")
            else ModelDecodeStreams(
                streams,
                torch.cuda.Event,
                ordered=self.topology == "three-stream-ordered",
            )
        )
        return endpoints, stream_order

    def output_has_headroom(self):
        # A pass can publish old decode tickets, old prefill, and a new
        # empty prefill synchronously. Reserve room for every publication.
        return all(
            q.qsize() <= q.maxsize - self.decode_depth - 2
            for q in self.outputs.values()
        )

    def _run(self):
        try:
            if self.step_trace is not None:
                self.step_trace.parent.mkdir(parents=True, exist_ok=True)
                self.step_trace_file = self.step_trace.open("w", buffering=131072)
            controller = self._initialize()
            self.ready.set_result(None)
            observed = set()
            while True:
                self.wakeup.clear()
                self._process_inputs(controller)
                if self.stop.is_set():
                    with self.lock:
                        active = list(self.active.values())
                    for entry in active:
                        if entry.started:
                            self.cores[entry.model_id].abort_requests(
                                [entry.request.request_id]
                            )
                        self._remove(entry)
                    for parked in self.parked.values():
                        parked.clear()
                    # Discard transport on teardown, while retiring GPU tickets.
                    for output_queue in self.outputs.values():
                        while not output_queue.empty():
                            output_queue.get_nowait()
                    if not controller.has_work() and self.inputs.empty():
                        break
                headroom = self.output_has_headroom()
                if headroom and controller.advance():
                    continue
                for future in controller._waiting_futures:
                    if future not in observed:
                        observed.add(future)
                        future.add_done_callback(lambda _: self.wakeup.set())
                observed = {f for f in observed if not f.done()}
                if self.manager is not None:
                    self.last_pool_state = self.manager.snapshot()
                free_blocks = {}
                for mid, core in self.cores.items():
                    pool = core.scheduler.kv_cache_manager.block_pool
                    free_blocks[mid] = pool.get_num_free_blocks()
                # Read scheduler/KV state on their owner thread. HTTP readers
                # receive a completed snapshot, including younger decode
                # tickets that may outlive an EOS response. No GPU sync.
                engine_state = {
                    "has_work": controller.has_work(),
                    "pending_tickets": sum(
                        len(q) for q in controller._pending.values()
                    ),
                    "inputs_queued": self.inputs.qsize(),
                    "models": {
                        mid: {
                            "free_blocks": free_blocks[mid],
                            "idle_free_blocks": self.idle_free_blocks[mid],
                            "running": len(core.scheduler.running),
                            "waiting": len(core.scheduler.waiting),
                            "async_decode_refs": len(
                                getattr(core.scheduler, "_hb_async_decode_refs", ())
                            ),
                        }
                        for mid, core in self.cores.items()
                    },
                }
                if (
                    not engine_state["has_work"]
                    and not engine_state["inputs_queued"]
                    and self.step_trace_file is not None
                ):
                    self.step_trace_file.flush()
                self.last_engine_state = engine_state
                self.wakeup.wait(
                    0.001
                    if controller.has_work()
                    or not headroom
                    or not self.inputs.empty()
                    or self._has_dispatchable_parked()
                    else None
                )
        except BaseException as exc:
            logger.exception("multi-model backend failed")
            self.error = exc
            if self.manager is not None:
                self.manager.poison(repr(exc))
            if not self.ready.done():
                self.ready.set_exception(exc)
        finally:
            self.stop.set()
            for mid in self.outputs:
                self._notify_outputs(mid)
            try:
                if self.cores:
                    import torch

                    torch.cuda.synchronize()
                if self.lane_masks is not None:
                    self.lane_masks.reset()
                for core in self.cores.values():
                    core.shutdown()
            except BaseException:
                logger.exception("multi-model backend teardown failed")
            finally:
                if self.step_trace_file is not None:
                    self.step_trace_file.close()

    async def close(self, timeout=30):
        self.stop.set()
        self.wakeup.set()
        await asyncio.to_thread(self.thread.join, timeout)
        if self.thread.is_alive():
            self.error = RuntimeError(
                "CUDA owner did not shut down; terminate the service process"
            )
            for mid in self.outputs:
                self.output_ready[mid].set()
            raise self.error

    def snapshot(self):
        with self.lock:
            count = len(self.active)
            serial = (
                {
                    "expected": dict(self.offline_serial_counts),
                    "received": dict(self.serial_received),
                    "completed": dict(self.serial_completed),
                    "current_model": (
                        self.model_order[self.serial_index]
                        if self.serial_index < len(self.model_order)
                        else None
                    ),
                    "parked": {mid: len(q) for mid, q in self.parked.items()},
                }
                if self.offline_serial_counts is not None
                else None
            )
        return {
            "active_requests": count,
            "max_requests": self.max_requests,
            "kv": self.last_pool_state,
            "engine_state": self.last_engine_state,
            "admission": self.admission,
            "topology": self.topology,
            "execution_stream_ids": self.execution_stream_ids,
            "offline_model_serial": serial,
            "step_trace": str(self.step_trace) if self.step_trace is not None else None,
            "smctrl_mask": self.mask_metadata,
            "dense_step_tokens": self.dense_step_tokens,
            "adaptive_dense_steps": self.adaptive_dense_steps,
            "failed": self.error is not None,
        }


class ModelCoreClient(EngineCoreClient):
    """Model-bound native async transport; the backend owns shared lifetime."""

    engine_ranks_managed = [0]

    def __init__(self, backend, model_id):
        self.backend, self.model_id = backend, model_id

    @property
    def engine_dead(self):
        return self.backend.error is not None or self.backend.stop.is_set()

    def shutdown(self, timeout=None):
        # Closing one AsyncLLM facade must not tear down its sibling.
        pass

    async def get_supported_tasks_async(self):
        return self.backend.supported_tasks[self.model_id]

    async def add_request_async(self, request):
        self.backend.submit(self.model_id, request)

    async def abort_requests_async(self, request_ids):
        self.backend.cancel(self.model_id, request_ids)

    async def get_output_async(self):
        output_queue = self.backend.outputs[self.model_id]
        event = self.backend.output_ready[self.model_id]
        while True:
            if self.engine_dead:
                raise EngineDeadError()
            try:
                output = output_queue.get_nowait()
                self.backend.wakeup.set()
                return output
            except queue.Empty:
                event.clear()
                if not output_queue.empty() or self.engine_dead:
                    continue
                await event.wait()

    async def reset_mm_cache_async(self):
        # No multimodal receiver/cache is supported in this text-only mode.
        return None

    async def reset_encoder_cache_async(self):
        return None

    async def is_sleeping_async(self):
        return False

    def dp_engines_running(self):
        return not self.engine_dead
