# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-labelled round-robin control of shared prefill and decode lanes.

All scheduler mutations and CPU model launches run on the calling thread.
Only the asynchronous output waiters run off-thread. A model's existing
lane contexts protect its buffers; each model has a distinct runner/KV cache.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelEndpoint:
    model_id: str
    core: Any
    publish: Callable[[Any], None]
    prefill_reserve_blocks: int = 0
    async_decode: bool = False
    decode_enabled: bool = True
    prefill_step_tokens: int = 0
    exclusive_execution: bool = False

    def __post_init__(self) -> None:
        if (
            not self.model_id
            or min(self.prefill_reserve_blocks, self.prefill_step_tokens) < 0
        ):
            raise ValueError("invalid model ID or prefill KV reserve")
        if self.exclusive_execution and self.decode_enabled:
            raise ValueError("exclusive execution requires a pooling endpoint")


@dataclass(frozen=True)
class ModelTicket:
    model_id: str
    lane: str
    step: Any
    dispatched_at: float
    dispatch_ms: float
    kv_free_before: int
    running_before: int
    waiting_before: int
    policy_phase: str = "round-robin"
    kv_admission_policy: str = "current"
    smctrl_mask: dict | None = None


@dataclass
class DrainResult:
    steps: list[dict[str, Any]] = field(default_factory=list)
    waits: int = 0
    watermark_skips: int = 0
    idle_watermark_bypasses: int = 0
    hybrid_steps: int = 0
    model_phases: list[dict[str, Any]] = field(default_factory=list)
    policy_transitions: list[dict[str, Any]] = field(default_factory=list)


class BulkTailPolicy:
    """Keep one model's bulk batch; prefill the next during its decode tail."""

    def __init__(self, models: Sequence[ModelEndpoint], tail_fraction: float):
        if len(models) != 2 or not 0 < tail_fraction <= 1:
            raise ValueError("bulk-tail requires two models and 0 < tail fraction <= 1")
        self.models = models
        self.tail_fraction = tail_fraction
        self.phase = "bulk"
        self.peak_running = 0

    @staticmethod
    def waiting(sched) -> int:
        return len(
            {
                r.request_id
                for queue in (
                    sched.waiting,
                    getattr(sched, "skipped_waiting", ()),
                    getattr(sched, "kv_holding_waiting", ()),
                    getattr(sched, "deferred_waiting", ()),
                )
                for r in queue
            }
        )

    def update(self, pending_models: set[str]) -> None:
        first = self.models[0]
        sched = first.core.scheduler
        running = len(sched.running)
        self.peak_running = max(self.peak_running, running)
        if not sched.has_requests() and first.model_id not in pending_models:
            self.phase = "handoff"
        elif (
            self.phase == "bulk"
            and self.peak_running > 0
            and self.waiting(sched) == 0
            and all(r.num_computed_tokens >= r.num_prompt_tokens for r in sched.running)
            and running <= self.tail_fraction * self.peak_running
        ):
            self.phase = "tail"

    def permits(self, model: ModelEndpoint, lane: str) -> bool:
        first = model.model_id == self.models[0].model_id
        if self.phase == "bulk":
            return first
        if self.phase == "handoff":
            return not first
        if lane == "decode":
            return first
        # A preemption can put a tail request back into prefill. Resume it
        # before admitting more dormant B requests to avoid starving A.
        sched = self.models[0].core.scheduler
        return first or (
            self.waiting(sched) == 0
            and all(r.num_computed_tokens >= r.num_prompt_tokens for r in sched.running)
        )


class SharedKVAdmissionBudget:
    """CPU capacity accounting for requested output limits, not future tokens.

    Running requests reserve their complete prompt/output footprint. Retired
    pages still behind GPU fences also occupy capacity. Blocks are allocated
    normally during execution; this object only controls waiting admission.
    One budget is used for one synchronous scheduler pass on the launch thread.
    """

    def __init__(self, models: Sequence[ModelEndpoint], target: ModelEndpoint):
        pool = target.core.scheduler.kv_cache_manager.block_pool
        self.manager = pool.shared_manager
        self.target = target
        state = self.manager.snapshot()
        occupied = state["pending_pages"]
        accounted = set()
        for model in models:
            sched = model.core.scheduler
            other = sched.kv_cache_manager.block_pool
            if (
                getattr(other, "shared_manager", None) is not self.manager
                or other.shared_model_id != model.model_id
            ):
                raise ValueError("output reservation requires one shared KV manager")
            if sched.num_lookahead_tokens:
                raise ValueError("output reservation does not support speculative KV")
            reserved = sum(self.pages(model, request) for request in sched.running)
            occupied += max(reserved, state["owned_pages"][model.model_id])
            accounted.add(model.model_id)
        occupied += sum(
            pages for mid, pages in state["owned_pages"].items() if mid not in accounted
        )
        self.available_pages = self.manager.num_pages - 1 - occupied

    def pages(self, model: ModelEndpoint, request) -> int:
        layout = self.manager.layouts[model.model_id]
        # max_tokens is a public request limit. Actual sampled outputs and the
        # benchmark manifest are not consulted by this admission policy.
        tokens = min(
            request.num_prompt_tokens + request.max_tokens,
            model.core.scheduler.max_model_len,
        )
        blocks = (tokens + layout.block_size - 1) // layout.block_size
        return blocks * self.manager.units[model.model_id]

    def can_admit(self, request) -> bool:
        pages = self.pages(self.target, request)
        if pages > self.manager.num_pages - 1:
            raise ValueError("request prompt/output limit exceeds shared KV capacity")
        return pages <= self.available_pages

    def __call__(self, request) -> bool:
        if not self.can_admit(request):
            return False
        self.available_pages -= self.pages(self.target, request)
        return True


class DecodeHeadroomGuard:
    """Leave allocator-derived space for decode growth during one prefill."""

    name = "headroom"

    def __init__(self, sched, growth_tokens: int, step_tokens: int | None):
        from vllm.v1.engine.kv_headroom import CacheHeadroom

        self.budget = CacheHeadroom(
            sched.kv_cache_manager,
            sched.running,
            max(1, growth_tokens),
            step_tokens or sched.max_num_scheduled_tokens,
        )

    def can_admit(self, request) -> bool:
        return self.budget.can_admit(request)

    def __call__(self, request) -> bool:
        return self.budget(request)


class MultiModelController:
    """One shared prefill ticket and a bounded set of decode tickets.

    Decode publication is FIFO, including when two models alternate. Per-core
    FIFOs track placeholder ownership so one model cannot release another's
    requests. No same-lane input context is used on two different streams.
    ``serialized`` is a co-resident correctness reference, not a full-GPU solo.
    """

    def __init__(
        self,
        models: Sequence[ModelEndpoint],
        *,
        decode_depth: int = 2,
        serialized: bool = False,
        timeout_s: float = 900,
        kv_admission: str = "current",
        model_policy: str = "round-robin",
        tail_fraction: float = 0.5,
        lane_masks: Any = None,
        independent_decode: bool = False,
        decode_streams: Any = None,
        record_steps: bool = True,
        step_observer: Callable[[dict], None] | None = None,
        adaptive_dense_steps: bool = False,
        hybrid_min_decode: int = 0,
    ) -> None:
        self.models = list(models)
        # Stock-fallback policy: below this many decode-ready requests the
        # generation model runs the scheduler's merged ("default") batch on the
        # decode stream, unmasked, instead of split prefill/decode steps.
        if hybrid_min_decode < 0:
            raise ValueError("hybrid_min_decode must be nonnegative")
        self.hybrid_min_decode = hybrid_min_decode
        self.hybrid_hysteresis = max(1, hybrid_min_decode // 4)
        self._stock_mode = False
        # Run-time timing for the decode-headroom admission guard: EMA of the
        # prefill ticket duration and of the decode step period (completion
        # to completion, so launch-ahead queueing does not inflate it).
        self._prefill_ticket_ms: float | None = None
        self._decode_step_ms: float | None = None
        self._last_decode_done: float | None = None
        self.has_exclusive_models = any(m.exclusive_execution for m in self.models)
        self.record_steps = record_steps
        self.step_observer = step_observer
        self._session = None
        self._pending = {}
        self._waiting_futures = []
        self.lane_masks = lane_masks
        if adaptive_dense_steps and lane_masks is None:
            raise ValueError("adaptive dense steps require a live mask policy")
        self.adaptive_dense_steps = adaptive_dense_steps
        self.independent_decode = independent_decode
        self.decode_streams = decode_streams
        if not self.models or len({m.model_id for m in models}) != len(models):
            raise ValueError("models must have unique, nonempty IDs")
        if len({id(m.core) for m in models}) != len(models):
            raise ValueError("each model must own a distinct engine core")
        if not 1 <= decode_depth <= 4 or timeout_s <= 0:
            raise ValueError("invalid decode depth or drain timeout")
        self.decode_depth = decode_depth
        self.serialized = serialized
        self.timeout_s = timeout_s
        self.by_id = {m.model_id: m for m in models}
        if kv_admission not in ("current", "full-output"):
            raise ValueError("unknown KV admission policy")
        self.kv_admission = kv_admission
        if model_policy not in ("round-robin", "bulk-tail"):
            raise ValueError("unknown model policy")
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail fraction must be in (0, 1]")
        self.model_policy, self.tail_fraction = model_policy, tail_fraction
        if model_policy == "bulk-tail" and len(self.models) != 2:
            raise ValueError("bulk-tail requires exactly two models")
        if kv_admission == "full-output" or model_policy == "bulk-tail":
            for model in self.models:
                SharedKVAdmissionBudget(self.models, model)

    @staticmethod
    def _ready(model: ModelEndpoint, lane: str) -> bool:
        if lane == "decode" and not model.decode_enabled:
            return False
        sched = model.core.scheduler
        # Prefill must also exclude requests owned by async decode tickets.
        inflight = set(sched._hb_inflight_req_ids)
        if lane == "prefill":
            inflight.update(getattr(sched, "_hb_async_decode_refs", ()))
            return any(r.request_id not in inflight for r in sched.waiting) or any(
                r.request_id not in inflight
                and r.num_computed_tokens < r.num_prompt_tokens
                for r in sched.running
            )
        return any(
            r.request_id not in inflight
            and r.num_computed_tokens >= r.num_prompt_tokens
            for r in sched.running
        )

    def drain_model_serial(
        self, *, synchronize: Callable[[], None] | None = None
    ) -> DrainResult:
        """Drain each model's entire queue before launching the next model.

        All models remain resident with their original KV allocations. Each
        model retains its configured lane/decode pipeline; only cross-model
        execution is serialized. The optional device barrier runs at every
        model boundary and is included in the measured phase.
        """
        result = DrainResult()
        self.last_result = result
        deadline = time.perf_counter() + self.timeout_s
        for model in self.models:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("model-serial fixed queue did not drain in time")
            controller = MultiModelController(
                [model],
                decode_depth=self.decode_depth,
                serialized=self.serialized,
                timeout_s=remaining,
            )
            started = time.perf_counter()
            completed = False
            try:
                controller.drain()
                if synchronize is not None:
                    synchronize()
                completed = True
            finally:
                partial = getattr(controller, "last_result", DrainResult())
                result.steps.extend(partial.steps)
                result.waits += partial.waits
                result.watermark_skips += partial.watermark_skips
                result.idle_watermark_bypasses += partial.idle_watermark_bypasses
                result.model_phases.append(
                    {
                        "model_id": model.model_id,
                        "t_start": started,
                        "t_end": time.perf_counter(),
                        "completed": completed,
                    }
                )
        return result

    def notify_requests_changed(self) -> None:
        """Retry admission after owner-thread arrivals or cancellations."""
        for state in getattr(self, "_eligibility", ()):
            state.clear()

    def _observe_timing(self, lane: str, dispatched_at: float) -> None:
        now = time.perf_counter()
        if lane == "decode":
            if self._last_decode_done is not None:
                period = (now - self._last_decode_done) * 1000
                self._decode_step_ms = (
                    period
                    if self._decode_step_ms is None
                    else 0.9 * self._decode_step_ms + 0.1 * period
                )
            self._last_decode_done = now
        elif lane == "prefill":
            elapsed = (now - dispatched_at) * 1000
            self._prefill_ticket_ms = (
                elapsed
                if self._prefill_ticket_ms is None
                else 0.8 * self._prefill_ticket_ms + 0.2 * elapsed
            )

    def _growth_tokens(self) -> int:
        """Tokens one decoding request grows by during one prefill ticket."""
        if not self._prefill_ticket_ms or not self._decode_step_ms:
            return 1
        steps = self._prefill_ticket_ms / self._decode_step_ms
        return max(1, math.ceil(steps))

    def has_work(self) -> bool:
        if any(self._pending.values()):
            return True
        for model in self.models:
            scheduler = model.core.scheduler
            if scheduler.has_requests() or scheduler.has_finished_requests():
                return True
            manager = getattr(
                scheduler.kv_cache_manager.block_pool, "shared_manager", None
            )
            if manager is not None and manager.snapshot()["pending_pages"]:
                return True
        return False

    def advance(self) -> bool:
        """One bounded owner-thread pass; never wait for GPU completion."""
        self._waiting_futures = []
        if self._session is None:
            self._session = self._passes()
        return next(self._session)

    def drain(self) -> DrainResult:
        deadline = time.perf_counter() + self.timeout_s
        while self.has_work():
            if time.perf_counter() >= deadline:
                raise TimeoutError("multi-model fixed queue did not drain in time")
            if not self.advance():
                if self._waiting_futures:
                    wait(
                        self._waiting_futures,
                        timeout=max(0, deadline - time.perf_counter()),
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    time.sleep(0.0001)
        if self._session is None:
            self.advance()
        return self.last_result

    def _passes(self):
        result = DrainResult()
        self.last_result = result
        pending: dict[str, deque[ModelTicket]] = {
            "decode": deque(),
            "prefill": deque(),
        }
        self._pending = pending
        cursor = {"decode": 0, "prefill": 0}
        blocked: set[tuple[str, str]] = set()
        reclamation_blocked: dict[tuple[str, str], int] = {}
        admission_blocked: set[tuple[str, str]] = set()
        self._eligibility = blocked, reclamation_blocked, admission_blocked
        bulk_tail = (
            BulkTailPolicy(self.models, self.tail_fraction)
            if self.model_policy == "bulk-tail"
            else None
        )
        if bulk_tail is not None:
            result.policy_transitions.append(
                {"phase": "bulk", "t_start": time.perf_counter(), "peak_running": 0}
            )
        for model in self.models:
            core = model.core
            if getattr(core, "_hb_inflight_lanes", set()):
                raise RuntimeError(f"model {model.model_id} already has pending steps")
            if (
                model.decode_enabled
                and self.decode_depth > 1
                and not model.async_decode
            ):
                raise ValueError("decode launch-ahead requires the async scheduler")
            core._hb_decode_depth = self.decode_depth if model.decode_enabled else 1
            core._hb_decode_fifo = deque()

        def complete(ticket: ModelTicket) -> None:
            model = self.by_id[ticket.model_id]
            core = model.core
            if ticket.lane == "decode":
                owned = core._hb_decode_fifo.popleft()
                if owned is not ticket.step:
                    raise RuntimeError("per-model decode publication is out of order")
            output = core.finish(ticket.step)
            outputs = core.complete(ticket.step, output)
            scheduled = ticket.step.scheduler_output
            productive = scheduled.total_num_scheduled_tokens > 0
            if productive:
                self._observe_timing(ticket.lane, ticket.dispatched_at)
            core.post_step(model_executed=productive)
            model.publish(outputs)
            if self.record_steps or self.step_observer is not None:
                record = {
                    "model_id": ticket.model_id,
                    "lane": ticket.lane,
                    "t_start": ticket.dispatched_at,
                    "t_end": time.perf_counter(),
                    "dispatch_ms": ticket.dispatch_ms,
                    "scheduled_tokens": scheduled.total_num_scheduled_tokens,
                    "scheduled_requests": len(scheduled.num_scheduled_tokens),
                    "num_scheduled_tokens": dict(scheduled.num_scheduled_tokens),
                    "running_before": ticket.running_before,
                    "waiting_before": ticket.waiting_before,
                    "policy_phase": ticket.policy_phase,
                    "exclusive_execution": model.exclusive_execution,
                    "kv_admission_policy": ticket.kv_admission_policy,
                    "smctrl_mask": ticket.smctrl_mask,
                    "kv_free_before": ticket.kv_free_before,
                    "kv_free_after": (
                        core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                    ),
                    "preemptions": len(
                        getattr(scheduled, "preempted_req_ids", ()) or ()
                    ),
                    "kv_reclamation_needed_blocks": getattr(
                        scheduled, "kv_reclamation_needed_blocks", 0
                    ),
                    "kv_admission_deferred": getattr(
                        scheduled, "kv_admission_deferred", False
                    ),
                }
                pool = core.scheduler.kv_cache_manager.block_pool
                if getattr(pool, "shared_manager", None) is not None:
                    record["shared_kv"] = pool.shared_manager.snapshot()
                if self.record_steps:
                    result.steps.append(record)
                if self.step_observer is not None:
                    self.step_observer(record)
            if productive:
                blocked.clear()
                reclamation_blocked.clear()
                admission_blocked.clear()

        def dispatch(model: ModelEndpoint, lane: str) -> None:
            sched = model.core.scheduler
            free = sched.kv_cache_manager.block_pool.get_num_free_blocks()
            running, waiting = len(sched.running), len(sched.waiting)
            started = time.perf_counter()
            guard = (
                SharedKVAdmissionBudget(self.models, model)
                if lane == "prefill"
                and (
                    self.kv_admission == "full-output"
                    or bulk_tail is not None
                    and bulk_tail.phase == "tail"
                )
                else None
            )
            if (
                guard is None
                and lane == "prefill"
                and model.decode_enabled
                and getattr(sched.kv_cache_manager.block_pool, "shared_manager", None)
                is None
            ):
                guard = DecodeHeadroomGuard(
                    sched,
                    self._growth_tokens(),
                    model.prefill_step_tokens,
                )
            previous_guard = getattr(sched, "_hb_kv_admission_guard", None)
            previous_budget = getattr(sched, "_hb_step_token_budget", None)
            sched._hb_kv_admission_guard = guard
            if lane == "prefill" and model.prefill_step_tokens:
                sched._hb_step_token_budget = model.prefill_step_tokens
            try:
                if self.decode_streams is not None:
                    self.decode_streams.before_dispatch(model, lane)
                mask_record = (
                    self.lane_masks.before_dispatch(
                        model,
                        lane,
                        {t.model_id for fifo in pending.values() for t in fifo},
                        pending_tickets=[t for fifo in pending.values() for t in fifo],
                    )
                    if self.lane_masks is not None and lane != "default"
                    else None
                )
                if (
                    self.adaptive_dense_steps
                    and not model.decode_enabled
                    and lane == "prefill"
                    and not mask_record["masked"]
                ):
                    # Spend small units only while protecting decode. Full
                    # capacity serves low load, urgent work, and the tail.
                    sched._hb_step_token_budget = None
                step = model.core.dispatch(lane)
                if self.decode_streams is not None:
                    self.decode_streams.after_dispatch(model, lane)
            finally:
                sched._hb_kv_admission_guard = previous_guard
                sched._hb_step_token_budget = previous_budget
            ticket = ModelTicket(
                model.model_id,
                lane,
                step,
                started,
                (time.perf_counter() - started) * 1000,
                free,
                running,
                waiting,
                bulk_tail.phase if bulk_tail is not None else "round-robin",
                getattr(guard, "name", "full-output")
                if guard is not None
                else "current",
                mask_record,
            )
            if lane == "decode":
                model.core._hb_decode_fifo.append(step)
            # Merged ("default") batches run on the decode stream and publish
            # through the decode FIFO; they are synchronous inflight tickets.
            publish_lane = "decode" if lane == "default" else lane
            if step.scheduler_output.total_num_scheduled_tokens or lane == "decode":
                # Empty decode passes also mutate worker/request state. Publish
                # them behind older tickets, not inline ahead of placeholders.
                pending[publish_lane].append(ticket)
            if not step.scheduler_output.total_num_scheduled_tokens:
                # An empty pass can flush finished request IDs to the worker.
                blocked.add((model.model_id, lane))
                needed = getattr(
                    step.scheduler_output, "kv_reclamation_needed_blocks", 0
                )
                if needed:
                    reclamation_blocked[model.model_id, lane] = needed
                if getattr(step.scheduler_output, "kv_admission_deferred", False):
                    admission_blocked.add((model.model_id, lane))
                if lane != "decode":
                    complete(ticket)

        def choose(
            lane: str, *, bypass_watermark: bool = False, exclusive_only: bool = False
        ):
            if bulk_tail is not None:
                before = bulk_tail.phase
                bulk_tail.update(
                    {ticket.model_id for fifo in pending.values() for ticket in fifo}
                )
                if before != bulk_tail.phase:
                    sched = self.models[0].core.scheduler
                    result.policy_transitions.append(
                        {
                            "phase": bulk_tail.phase,
                            "t_start": time.perf_counter(),
                            "first_model_running": len(sched.running),
                            "first_model_waiting": bulk_tail.waiting(sched),
                            "peak_running": bulk_tail.peak_running,
                            "tail_fraction": self.tail_fraction,
                        }
                    )
                    # Admission decisions made with tail reservations must be
                    # retried under the ordinary policy once A has drained.
                    blocked.difference_update(admission_blocked)
                    admission_blocked.clear()
            indices = [
                (cursor[lane] + offset) % len(self.models)
                for offset in range(len(self.models))
            ]
            if lane == "prefill":
                # Stable partition retains round-robin within each role.
                # Pooling fills dense-stream slack only when generation
                # prefill cannot dispatch (including KV watermark blocking).
                indices.sort(key=lambda i: not self.models[i].decode_enabled)
                if self.lane_masks is not None and hasattr(
                    self.lane_masks, "prefill_priority_indices"
                ):
                    indices = self.lane_masks.prefill_priority_indices(indices)
            for index in indices:
                model = self.models[index]
                if model.exclusive_execution != exclusive_only:
                    continue
                if bulk_tail is not None and not bulk_tail.permits(model, lane):
                    continue
                key = (model.model_id, lane)
                if key in admission_blocked:
                    sched = model.core.scheduler
                    if (
                        getattr(
                            sched.kv_cache_manager.block_pool, "shared_manager", None
                        )
                        is None
                    ):
                        budget = DecodeHeadroomGuard(
                            sched,
                            self._growth_tokens(),
                            model.prefill_step_tokens,
                        )
                    else:
                        budget = SharedKVAdmissionBudget(self.models, model)
                    inflight = set(sched._hb_inflight_req_ids)
                    inflight.update(getattr(sched, "_hb_async_decode_refs", ()))
                    request = next(
                        (r for r in sched.waiting if r.request_id not in inflight), None
                    )
                    if request is None or budget.can_admit(request):
                        blocked.discard(key)
                        admission_blocked.discard(key)
                if key in reclamation_blocked:
                    pool = model.core.scheduler.kv_cache_manager.block_pool
                    free = pool.get_num_free_blocks()
                    if (
                        free >= reclamation_blocked[key]
                        or pool.shared_manager.snapshot()["pending_pages"] == 0
                    ):
                        # Capacity may have been consumed by the other model
                        # in the meantime. If nothing remains pending, retry
                        # normal scheduling/preemption rather than deadlock.
                        blocked.discard(key)
                        del reclamation_blocked[key]
                if (model.model_id, lane) in blocked or not self._ready(model, lane):
                    continue
                if lane == "prefill" and not bypass_watermark:
                    pool = model.core.scheduler.kv_cache_manager.block_pool
                    if pool.get_num_free_blocks() < model.prefill_reserve_blocks:
                        result.watermark_skips += 1
                        continue
                cursor[lane] = (index + 1) % len(self.models)
                return model
            return None

        while True:
            if not self.has_work():
                yield False
                continue
            progress = False
            for lane in ("decode", "prefill"):
                # Distinct decode streams have no cross-model publication
                # dependency. Only each model's oldest ticket may publish.
                # Keep the global quota unchanged to isolate GPU overlap.
                while pending[lane]:
                    heads = []
                    seen = set()
                    for ticket in pending[lane]:
                        if ticket.model_id not in seen:
                            heads.append(ticket)
                            seen.add(ticket.model_id)
                        if lane != "decode" or not self.independent_decode:
                            break
                    ready = next((t for t in heads if t.step.future.done()), None)
                    if ready is None:
                        break
                    pending[lane] = deque(t for t in pending[lane] if t is not ready)
                    complete(ready)
                    progress = True
            # Drain already-issued work before exclusive auxiliary execution.
            # Stop decode launch-ahead while it waits, otherwise continuously
            # ready decodes can starve embedding indefinitely. Completion is
            # observed through existing futures; no device synchronization.
            exclusive_pending = self.has_exclusive_models and any(
                self.by_id[t.model_id].exclusive_execution
                for fifo in pending.values()
                for t in fifo
            )
            exclusive = (
                None
                if exclusive_pending or not self.has_exclusive_models
                else choose("prefill", exclusive_only=True)
            )
            if exclusive is not None and not any(pending.values()):
                dispatch(exclusive, "prefill")
                progress = True
                exclusive_pending = True
            # Stock-mode policy (per step): below hybrid_min_decode decode-ready
            # requests the generation model stops issuing split prefill/decode
            # tickets, drains, and runs merged ("default") batches on the
            # decode stream, unmasked, like stock. It leaves stock mode once
            # the decode-ready count reaches hybrid_min_decode + hysteresis.
            hybrid_inflight = any(t.lane == "default" for t in pending["decode"])
            stock_mode = False
            if self.hybrid_min_decode and not exclusive_pending and exclusive is None:
                for model in self.models:
                    if not model.decode_enabled:
                        continue
                    sched = model.core.scheduler
                    ready = sum(
                        r.num_computed_tokens >= r.num_prompt_tokens
                        for r in sched.running
                    )
                    enter, leave = (
                        self.hybrid_min_decode,
                        self.hybrid_min_decode + self.hybrid_hysteresis,
                    )
                    if self._stock_mode and ready >= leave:
                        self._stock_mode = False
                    elif not self._stock_mode and ready < enter:
                        self._stock_mode = True
                    stock_mode = self._stock_mode
                    if (
                        stock_mode
                        and not any(pending.values())
                        and sched.has_requests()
                        and (model.model_id, "default") not in blocked
                    ):
                        dispatch(model, "default")
                        result.hybrid_steps += 1
                        progress = True
                        hybrid_inflight = True
                    break
            for lane, capacity in (("decode", self.decode_depth), ("prefill", 1)):
                if (
                    exclusive_pending
                    or exclusive is not None
                    or hybrid_inflight
                    or stock_mode
                ):
                    continue
                if len(pending[lane]) >= capacity:
                    continue
                if self.serialized and any(pending.values()):
                    continue
                model = choose(lane)
                if model is not None:
                    dispatch(model, lane)
                    progress = True
            if progress:
                yield True
                continue
            inflight = []
            for lane, fifo in pending.items():
                seen = set()
                for ticket in fifo:
                    if ticket.model_id not in seen:
                        inflight.append(ticket.step.future)
                        seen.add(ticket.model_id)
                    if lane != "decode" or not self.independent_decode:
                        break
            if inflight:
                # Only FIFO heads can publish; waiting on a younger, ready
                # decode ticket would spin while its predecessor is running.
                self._waiting_futures = inflight
                result.waits += 1
                yield False
                continue
            # Finished IDs require a final scheduler/worker pass even with no
            # productive work. These are still routed to their owning model.
            flushed = False
            for model in self.models:
                if model.core.scheduler.has_finished_requests():
                    dispatch(model, "decode" if model.decode_enabled else "prefill")
                    flushed = True
                    break
            if flushed:
                yield True
                continue
            # With no active work, KV cannot recover without another prefill.
            idle_model = (
                choose("prefill", bypass_watermark=True, exclusive_only=True)
                if self.has_exclusive_models
                else None
            )
            if idle_model is None:
                idle_model = choose("prefill", bypass_watermark=True)
            if idle_model is not None:
                result.idle_watermark_bypasses += 1
                dispatch(idle_model, "prefill")
                yield True
                continue
            if reclamation_blocked or admission_blocked:
                # Completion events can outlive the last published ticket.
                # Poll their nonblocking queries on the next iteration, without
                # a CUDA/device barrier or an artificial resident-request cap.
                yield False
                continue
            active = [
                m.model_id for m in self.models if m.core.scheduler.has_requests()
            ]
            if not active:
                yield False
                continue
            raise RuntimeError(f"multi-model controller stalled: {active=}, {blocked=}")
