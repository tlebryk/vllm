# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-labelled round-robin control of shared prefill and decode lanes.

All scheduler mutations and CPU model launches run on the calling thread.
Only the asynchronous output waiters run off-thread. A model's existing
lane contexts protect its buffers; each model has a distinct runner/KV cache.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.request import RequestStatus


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
    mode_switch: bool = False


@dataclass
class DrainResult:
    steps: list[dict[str, Any]] = field(default_factory=list)
    waits: int = 0
    watermark_skips: int = 0
    idle_watermark_bypasses: int = 0
    hybrid_steps: int = 0
    merged_switches: int = 0
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
        return len(sched.waiting) + len(getattr(sched, "skipped_waiting", ()))

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
    """Private-KV prefill admission that leaves room for decode to grow.

    The prefill lane admits waiting requests on its own ticket, so without a
    guard it can take the last free blocks and the next decode ticket preempts
    a resident request (whose prompt is then recomputed). Admission keeps
    ``growth_blocks`` free blocks per decode-ready request: the growth those
    requests need while one prefill ticket runs, measured at run time from the
    controller's prefill-ticket and decode-step durations.
    """

    name = "headroom"

    def __init__(self, sched, growth_blocks: int, step_tokens: int | None):
        pool = sched.kv_cache_manager.block_pool
        self.free = pool.get_num_free_blocks()
        self.block_size = sched.block_size
        self.step_tokens = step_tokens or sched.max_num_scheduled_tokens
        resident = sum(
            1 for r in sched.running if r.num_computed_tokens >= r.num_prompt_tokens
        )
        self.headroom = resident * max(1, growth_blocks)

    def _need(self, request) -> int:
        remaining = max(0, request.num_prompt_tokens - request.num_computed_tokens)
        return (
            min(remaining, self.step_tokens) + self.block_size - 1
        ) // self.block_size

    def can_admit(self, request) -> bool:
        return self.free - self._need(request) >= self.headroom

    def __call__(self, request) -> bool:
        if not self.can_admit(request):
            return False
        self.free -= self._need(request)
        return True


class ResidencyCap:
    """Admission guard that caps the number of running requests on a model.

    ``SLACKSERVE_MAX_RUNNING`` (per generation model) refuses new admissions once
    the scheduler's running set reaches the cap; resumptions of preempted requests
    pass through. It wraps the lane's KV guard, so KV sizing, memory profiling and
    the profiled MoE workspace are unchanged (a ``max_num_seqs`` cap shrinks the
    profiled workspace; a fixed ``kv_cache_memory_bytes`` skips profiling).
    """

    def __init__(self, sched, cap: int, inner=None):
        self.sched = sched
        self.cap = cap
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'none')}+cap{cap}"

    def __call__(self, request) -> bool:
        if (
            len(self.sched.running) >= self.cap
            and request.status != RequestStatus.PREEMPTED
        ):
            return False
        return self.inner is None or self.inner(request)


def _residency_cap() -> int | None:
    value = os.environ.get("SLACKSERVE_MAX_RUNNING")
    return int(value) if value else None


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
        # Merged-batch ("default" lane) execution: the generation model runs
        # the scheduler's stock mixed batch (prefill chunks and decodes in one
        # forward) on the decode stream, unmasked, with the same launch-ahead
        # depth as decode tickets. Entered either below ``hybrid_min_decode``
        # decode-ready requests (fixed gate with hysteresis) or when the
        # calibrated policy's merged candidate has the least cost for the
        # pending prefill (see ModelLaneMasks.merged_choice).
        if hybrid_min_decode < 0:
            raise ValueError("hybrid_min_decode must be nonnegative")
        self.hybrid_min_decode = hybrid_min_decode
        self.hybrid_hysteresis = max(1, hybrid_min_decode // 4)
        self._stock_mode = False
        self._merged_mode: dict[str, bool] = {}
        self._merged_record: dict[str, dict | None] = {}
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
        self._residency_cap = _residency_cap()
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
        if lane == "default":
            # Merged batch on the decode stream: any request not owned by a
            # prefill-stream ticket (requests in older decode-stream tickets
            # are rescheduled behind them, like decode launch-ahead).
            return any(
                r.request_id not in inflight for r in (*sched.waiting, *sched.running)
            )
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

    def _merged_now(self, model: ModelEndpoint, pending) -> bool:
        """Whether ``model``'s next decode-stream ticket is a merged batch.

        Precedence: a calibration probe forcing the mode; the calibrated
        policy's per-ticket choice when its table has a merged candidate and
        prefill work is pending (no pending prefill means split: a merged
        batch without prefill is a decode ticket); else the fixed
        ``hybrid_min_decode`` gate with hysteresis.
        """
        masks = self.lane_masks
        mid = model.model_id
        forced = getattr(masks, "force_merged", None) if masks is not None else None
        if forced is not None:
            self._merged_record[mid] = (
                masks.merged_record(model, policy="forced") if forced else None
            )
            return bool(forced)
        if masks is not None and masks.has_merged_candidate(model):
            tickets = [t for fifo in pending.values() for t in fifo]
            decision = masks.merged_choice(model, tickets)
            if decision is None:
                self._merged_record[mid] = None
                return False
            merged, record = decision
            self._merged_record[mid] = record if merged else None
            return merged
        if not self.hybrid_min_decode:
            self._merged_record[mid] = None
            return False
        sched = model.core.scheduler
        ready = sum(r.num_computed_tokens >= r.num_prompt_tokens for r in sched.running)
        enter = self.hybrid_min_decode
        leave = self.hybrid_min_decode + self.hybrid_hysteresis
        if self._stock_mode and ready >= leave:
            self._stock_mode = False
        elif not self._stock_mode and ready < enter:
            self._stock_mode = True
        self._merged_record[mid] = (
            {
                "masked": False,
                "policy": "gate",
                "candidate": "merged",
                "mode": "merged",
                "partner_decode_ready_requests": ready,
                "hybrid_min_decode": enter,
                "hybrid_leave": leave,
            }
            if self._stock_mode
            else None
        )
        return self._stock_mode

    def _growth_blocks(self, block_size: int) -> int:
        """Blocks one decoding request grows by during one prefill ticket."""
        if not self._prefill_ticket_ms or not self._decode_step_ms:
            return 1
        steps = self._prefill_ticket_ms / self._decode_step_ms
        return max(1, math.ceil(steps / block_size))

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
            if ticket.lane == "decode" or (
                ticket.lane == "default" and model.async_decode
            ):
                owned = core._hb_decode_fifo.popleft()
                if owned is not ticket.step:
                    raise RuntimeError("per-model decode publication is out of order")
            output = core.finish(ticket.step)
            outputs = core.complete(ticket.step, output)
            scheduled = ticket.step.scheduler_output
            productive = scheduled.total_num_scheduled_tokens > 0
            if productive:
                # A merged step without prefill tokens is a decode period.
                lane = ticket.lane
                if lane == "default":
                    lane = (
                        "decode"
                        if scheduled.total_num_scheduled_tokens
                        == len(scheduled.num_scheduled_tokens)
                        else "merged"
                    )
                self._observe_timing(lane, ticket.dispatched_at)
            core.post_step(model_executed=productive)
            model.publish(outputs)
            if self.record_steps or self.step_observer is not None:
                record = {
                    "model_id": ticket.model_id,
                    "lane": ticket.lane,
                    "execution_mode": "merged" if ticket.lane == "default" else "split",
                    "mode_switch": ticket.mode_switch,
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

        last_mode: dict[str, str] = {}

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
                    self._growth_blocks(sched.block_size),
                    model.prefill_step_tokens,
                )
            if (
                self._residency_cap
                and lane in ("prefill", "default")
                and model.decode_enabled
            ):
                guard = ResidencyCap(sched, self._residency_cap, guard)
            previous_guard = getattr(sched, "_hb_kv_admission_guard", None)
            previous_budget = getattr(sched, "_hb_step_token_budget", None)
            sched._hb_kv_admission_guard = guard
            # Merged batches take the prefill stream's work unit too: the
            # engine's memory profile covers that unit, not a full-budget
            # merged step on the decode stream (131K-token steps OOM'd).
            if lane in ("prefill", "default") and model.prefill_step_tokens:
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
                    else self._merged_record.get(model.model_id)
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
            # Execution mode of this ticket on the generation model: the first
            # productive ticket after a merged<->split change is marked so the
            # switching cost (empty passes, drained pipelines) can be read off
            # the trace. Empty flush passes do not change the mode.
            mode = "merged" if lane == "default" else "split"
            mode_switch = False
            if (
                model.decode_enabled
                and step.scheduler_output.total_num_scheduled_tokens
            ):
                mode_switch = last_mode.get(model.model_id, mode) != mode
                if mode_switch:
                    result.merged_switches += 1
                last_mode[model.model_id] = mode
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
                mode_switch,
            )
            # Merged ("default") batches run on the decode stream and publish
            # through the decode FIFO. With launch-ahead they are pipelined
            # exactly like decode tickets (placeholders, FIFO publication);
            # without it they are synchronous inflight tickets.
            pipelined = lane == "decode" or (lane == "default" and model.async_decode)
            if pipelined:
                model.core._hb_decode_fifo.append(step)
            publish_lane = "decode" if lane == "default" else lane
            if step.scheduler_output.total_num_scheduled_tokens or pipelined:
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
                if not pipelined:
                    complete(ticket)

        merged_models: set[str] = set()

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
                merged = model.model_id in merged_models
                if merged and lane == "prefill":
                    # This model's prefill rides in its merged batches.
                    continue
                key = (model.model_id, "default" if merged else lane)
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
                            self._growth_blocks(sched.block_size),
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
                if key in blocked or not self._ready(model, key[1]):
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
            # Merged-batch mode, decided per pass for each generation model
            # (fixed gate or calibrated choice, see _merged_now). In merged
            # mode the model's next decode-stream ticket is the scheduler's
            # stock mixed batch ("default" lane: resident decodes plus waiting
            # prefill chunks), pipelined to the same depth as decode tickets;
            # its prefill stream stays idle. Switching needs no drain: a
            # request is owned by at most one stream at a time (prefill-stream
            # tickets mark their ids; decode-stream tickets share FIFO order).
            merged_models.clear()
            if not exclusive_pending and exclusive is None:
                for model in self.models:
                    if model.decode_enabled and self._merged_now(model, pending):
                        merged_models.add(model.model_id)
            for lane, capacity in (("decode", self.decode_depth), ("prefill", 1)):
                if exclusive_pending or exclusive is not None:
                    continue
                if len(pending[lane]) >= capacity:
                    continue
                if self.serialized and any(pending.values()):
                    continue
                model = choose(lane)
                if model is not None:
                    if lane == "decode" and model.model_id in merged_models:
                        dispatch(model, "default")
                        result.hybrid_steps += 1
                    else:
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
