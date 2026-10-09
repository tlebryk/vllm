# SPDX-License-Identifier: Apache-2.0
"""Arrival-driven, single-owner backend for two native async model clients."""

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from vllm import EngineArgs
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.model_streams import ModelDecodeStreams
from vllm.v1.engine.multi_model_config import load_specs
from vllm.v1.engine.multi_model_controller import ModelEndpoint, MultiModelController

logger = logging.getLogger(__name__)


def prepare_models(path, *, mode, budget, decode_depth):
    """Validate both models on CPU before starting the CUDA owner thread."""
    specs = load_specs(Path(path), 0, 0)
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
            )
            or config.model_config.quantization is not None
        ):
            raise ValueError(
                "native multi-model serving supports unquantized dense Llama/Qwen3"
            )
        config.model_config.served_model_name = spec.model_id
        prepared.append(config)
    if mode not in ("layer", "head", "private"):
        raise ValueError("invalid shared KV layout")
    if budget is not None and (budget <= 0 or mode == "private"):
        raise ValueError("explicit shared budget requires a positive head/layer pool")
    return specs, prepared, budget or sum(s.kv_cache_memory_bytes for s in specs)


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
    ):
        if max_requests <= 0 or output_capacity <= decode_depth + 2:
            raise ValueError("invalid ingress/output capacity")
        if admission == "full-output" and mode == "private":
            raise ValueError("full-output admission requires shared KV")
        if topology not in (
            "two-stream",
            "three-stream-ordered",
            "three-stream-overlap",
        ):
            raise ValueError("invalid stream topology")
        self.specs, self.configs = specs, configs
        self.mode, self.budget, self.admission = mode, budget, admission
        self.topology, self.decode_depth = topology, decode_depth
        self.max_requests = max_requests
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
        if (
            params is None
            or params.structured_outputs is not None
            or request.mm_features
            or request.prompt_embeds is not None
            or request.lora_request is not None
            or request.resumable
        ):
            raise ValueError(
                "multi-model serving supports plain text generation without "
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
            entry = Submission(mid, request)
            self.inputs.put_nowait(entry)
            self.active[key] = entry
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
        changed = bool(cancelled)
        for _ in range(64):
            try:
                entry = self.inputs.get_nowait()
            except queue.Empty:
                break
            if entry.cancelled or self.stop.is_set():
                self._remove(entry)
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
                    if output.finish_reason is not None:
                        self.active.pop((mid, output.request_id), None)
                for rid in result.finished_requests or ():
                    self.active.pop((mid, rid), None)
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
        prefill = torch.cuda.Stream()
        shared_decode = torch.cuda.Stream()
        streams = {
            s.model_id: shared_decode
            if self.topology == "two-stream"
            else torch.cuda.Stream()
            for s in self.specs
        }
        endpoints = []
        for spec in self.specs:
            core = self.cores[spec.model_id]
            wrapped = core.model_executor.driver_worker
            worker = getattr(wrapped, "worker", wrapped)
            worker.bind_lane_streams(prefill, streams[spec.model_id])
            endpoints.append(
                ModelEndpoint(
                    spec.model_id,
                    core,
                    lambda out, mid=spec.model_id: self._publish(mid, out),
                    spec.prefill_reserve_blocks,
                    self.decode_depth > 1,
                )
            )
        torch.cuda.synchronize()
        stream_order = (
            None
            if self.topology == "two-stream"
            else ModelDecodeStreams(
                streams,
                torch.cuda.Event,
                ordered=self.topology == "three-stream-ordered",
            )
        )
        return MultiModelController(
            endpoints,
            decode_depth=self.decode_depth,
            kv_admission=self.admission,
            independent_decode=self.topology != "two-stream",
            decode_streams=stream_order,
            record_steps=False,
        )

    def output_has_headroom(self):
        # A pass can publish old decode tickets, old prefill, and a new
        # empty prefill synchronously. Reserve room for every publication.
        return all(
            q.qsize() <= q.maxsize - self.decode_depth - 2
            for q in self.outputs.values()
        )

    def _run(self):
        try:
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
                self.wakeup.wait(
                    0.001
                    if controller.has_work() or not headroom or not self.inputs.empty()
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
                for core in self.cores.values():
                    core.shutdown()
            except BaseException:
                logger.exception("multi-model backend teardown failed")

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
        return {
            "active_requests": count,
            "max_requests": self.max_requests,
            "kv": self.last_pool_state,
            "admission": self.admission,
            "topology": self.topology,
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
