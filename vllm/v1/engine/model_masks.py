# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit model/phase placement, owned by the multi-model launch thread."""

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMaskPlan:
    model_id: str
    lane: str
    tpcs: int

    @classmethod
    def parse(cls, spec: str):
        model_id, lane, count = spec.split(":")
        plan = cls(model_id, lane, int(count))
        if not model_id or lane not in ("prefill", "decode") or plan.tpcs <= 0:
            raise ValueError("mask must be MODEL:prefill|decode:POSITIVE_TPC_COUNT")
        if model_id == "*" and lane != "prefill":
            raise ValueError("all-model masks support the shared prefill stream only")
        return plan


class ModelLaneMasks:
    """Restore the shared lane before B launches; never mutate a global mask.

    Prefill can protect its own model's decode on the separate decode stream.
    Decode restriction protects another model's resident or in-flight work.
    Decode uses separate masked/unmasked graph banks because replay-time stream
    masks do not override the mask baked into a graph at capture.
    """

    def __init__(
        self,
        plan,
        models,
        workers,
        streams,
        total_tpcs,
        ctrl,
        indices,
        *,
        activation="resident",
        min_decode_requests=1,
        max_dense_wait_ms=0,
        clock=time.time,
        tiers=None,
    ):
        self.plans = tuple(plan) if isinstance(plan, (tuple, list)) else (plan,)
        if not self.plans or len({p.model_id for p in self.plans}) != len(self.plans):
            raise ValueError("mask plans need distinct model targets")
        if len(self.plans) > 1 and (
            any(p.model_id == "*" for p in self.plans)
            or {p.lane for p in self.plans} != {"prefill"}
        ):
            raise ValueError("multiple mask plans require named prefill models")
        self.plan, self.models, self.workers = self.plans[0], models, workers
        self.streams, self.total_tpcs, self.ctrl = streams, total_tpcs, ctrl
        if isinstance(indices, dict):
            if set(indices) != {p.model_id for p in self.plans}:
                raise ValueError("per-model mask indices must match mask targets")
            self.indices_by_model = {
                name: list(value) for name, value in indices.items()
            }
        elif len(self.plans) == 1:
            self.indices_by_model = {self.plan.model_id: list(indices)}
        else:
            raise ValueError("multiple mask plans require per-model indices")
        for current in self.plans:
            selected = self.indices_by_model[current.model_id]
            if (
                current.model_id not in workers and current.model_id != "*"
            ) or not 0 < current.tpcs < total_tpcs:
                raise ValueError(
                    "mask model must exist and TPC count must be below full GPU"
                )
            if (
                len(selected) != current.tpcs
                or len(set(selected)) != len(selected)
                or any(type(i) is not int or i < 0 or i >= total_tpcs for i in selected)
            ):
                raise ValueError(
                    "explicit mask must contain the requested number of distinct TPCs"
                )
        self.indices = self.indices_by_model[self.plan.model_id]
        # Load-tiered masks for the primary plan: [{"min_decode": n, "indices":
        # [...]}, ...]; at dispatch the tier with the largest min_decode that
        # the partner decode load reaches replaces the plan's single index set.
        self.tiers = []
        for tier in tiers or ():
            sel = [int(i) for i in tier["indices"]]
            if (
                int(tier["min_decode"]) < 0
                or not sel
                or len(set(sel)) != len(sel)
                or any(i < 0 or i >= total_tpcs for i in sel)
            ):
                raise ValueError(
                    "mask tier needs a nonnegative min_decode and distinct TPC indices"
                )
            self.tiers.append({"min_decode": int(tier["min_decode"]), "indices": sel})
        self.tiers.sort(key=lambda t: t["min_decode"])
        # Calibrated policy (mask_calibration.CalibrationTable): per prefill
        # ticket, the candidate with the least estimated cost replaces both
        # the plan's index set and the tiers. ``force_indices`` ([] =
        # unrestricted) is set only by the calibration run itself.
        self.calibration = None
        self.calibrated_model = None  # model whose prefill tickets the table governs
        self.force_indices = None
        self.state = {}
        if activation not in ("resident", "decode"):
            raise ValueError("unknown mask activation policy")
        self.activation = activation
        if min_decode_requests < 1:
            raise ValueError("minimum decode requests must be positive")
        self.min_decode_requests = min_decode_requests
        if not math.isfinite(max_dense_wait_ms) or max_dense_wait_ms < 0:
            raise ValueError("dense wait budget must be nonnegative")
        self.max_dense_wait_ms, self.clock = max_dense_wait_ms, clock

    def before_dispatch(self, model, lane, pending_models, *, pending_tickets=()):
        partners = [
            m
            for m in self.models
            if m.model_id != model.model_id or self.plan.lane == "prefill"
        ]
        partner_ids = {m.model_id for m in partners}
        partner_active = any(
            other.model_id != model.model_id
            and (other.core.scheduler.running or other.model_id in pending_models)
            for other in self.models
        )
        ready_by_model = {
            other.model_id: sum(
                getattr(r, "num_computed_tokens", 0)
                >= getattr(r, "num_prompt_tokens", 1)
                for r in other.core.scheduler.running
            )
            for other in partners
            if getattr(other, "decode_enabled", True)
        }
        decode_ready = sum(ready_by_model.values())
        decode_tickets = [
            t
            for t in pending_tickets
            if t.model_id in partner_ids
            and t.lane == "decode"
            and t.step.scheduler_output.total_num_scheduled_tokens
        ]
        # Pipeline tickets often contain the same requests. Count the largest
        # live batch per model, not the sum of launch-ahead tickets.
        decode_load = sum(
            max(
                ready,
                max(
                    (
                        t.step.scheduler_output.total_num_scheduled_tokens
                        for t in decode_tickets
                        if t.model_id == mid
                    ),
                    default=0,
                ),
            )
            for mid, ready in ready_by_model.items()
        )
        protect = (
            partner_active
            if self.activation == "resident"
            else decode_load >= self.min_decode_requests
        )
        dense_age_ms = 0.0
        if not getattr(model, "decode_enabled", True):
            sched = model.core.scheduler
            now = self.clock()
            dense_age_ms = max(
                (
                    1000 * max(0, now - getattr(r, "arrival_time", now))
                    for r in [
                        *sched.running,
                        *getattr(sched, "waiting", ()),
                        *getattr(sched, "skipped_waiting", ()),
                    ]
                ),
                default=0.0,
            )
        age_override = bool(
            self.max_dense_wait_ms and dense_age_ms >= self.max_dense_wait_ms
        )
        # Preserve the single-plan API's mutable `plan` used by local callers.
        active_plans = (self.plan,) if len(self.plans) == 1 else self.plans
        selected_plan = next(
            (
                p
                for p in active_plans
                if p.model_id == model.model_id or p.model_id == "*"
            ),
            None,
        )
        masked = (
            selected_plan is not None
            and lane == selected_plan.lane
            and protect
            and not age_override
        )
        policy = "gate"
        candidate = None
        estimated_tokens = None
        cost_s = None
        if lane == self.plan.lane:
            stream = self.streams.get((model.model_id, lane), self.streams.get(lane))
            selected_indices = (
                (
                    self.indices
                    if len(active_plans) == 1
                    else self.indices_by_model[selected_plan.model_id]
                )
                if masked
                else ()
            )
            tier_min_decode = None
            if masked and self.tiers and selected_plan.model_id == self.plan.model_id:
                tier = next(
                    (t for t in reversed(self.tiers) if decode_load >= t["min_decode"]),
                    None,
                )
                if tier is not None:
                    selected_indices = tier["indices"]
                    tier_min_decode = tier["min_decode"]
            if self.force_indices is not None:
                # Calibration run: measure exactly this candidate.
                policy = "forced"
                masked = bool(self.force_indices)
                selected_indices = list(self.force_indices)
            elif (
                self.calibration is not None
                and selected_plan is not None
                and selected_plan.model_id == self.plan.model_id
                and model.model_id == (self.calibrated_model or model.model_id)
                and not age_override
            ):
                policy = "calibrated"
                estimated_tokens, waiting_behind = self._estimate_ticket(model)
                candidate, indices, cost_s = self.calibration.choose(
                    estimated_tokens, decode_load, waiting_behind
                )
                masked = indices is not None
                selected_indices = list(indices) if masked else ()
            if self.ctrl is None:
                # Remote workers (TP>1) own the streams: the ticket carries the
                # TPC set and each rank's worker applies it to its stream.
                model.core.scheduler._hb_lane_mask = (
                    list(selected_indices) if masked else None
                )
            else:
                bits = sum(1 << i for i in selected_indices)
                mask = (~bits & ((1 << 128) - 1)) if masked else 0
                if self.state.get(stream.cuda_stream) != mask:
                    # Zero means completely unrestricted, including the upper TPCs.
                    self.ctrl._set_mask(stream, mask, "model-owned mask", log=False)
                    self.state[stream.cuda_stream] = mask
            if lane == "decode" and model.model_id == self.plan.model_id:
                self.workers[model.model_id].model_runner.select_decode_mask(masked)
        return {
            "masked": masked,
            "enabled_tpcs": (
                len(selected_indices)
                if masked and (self.tiers or policy != "gate")
                else selected_plan.tpcs
            )
            if masked
            else self.total_tpcs,
            "tier_min_decode": locals().get("tier_min_decode"),
            "policy": policy,
            "candidate": candidate,
            "estimated_tokens": estimated_tokens,
            "waiting_behind": locals().get("waiting_behind"),
            "shared_prefill_age_guard": getattr(self, "shared_prefill_age_guard", None),
            "cost_s": cost_s,
            "mask_model_id": selected_plan.model_id if masked else None,
            "partner_active": bool(partner_active),
            "activation": self.activation,
            "partner_decode_ready_requests": decode_ready,
            "partner_decode_load": decode_load,
            "min_decode_requests": self.min_decode_requests,
            "oldest_dense_age_ms": dense_age_ms,
            "max_dense_wait_ms": self.max_dense_wait_ms,
            "dense_age_override": age_override,
            "partner_decode_tickets": len(decode_tickets),
            "partner_decode_scheduled_tokens": sum(
                t.step.scheduler_output.total_num_scheduled_tokens
                for t in decode_tickets
            ),
            "graph_bank": "masked" if masked and lane == "decode" else "unmasked",
        }

    @staticmethod
    def _pending_prefills(model):
        sched = model.core.scheduler
        inflight = set(getattr(sched, "_hb_inflight_req_ids", ()))
        # A request can occur in more than one scheduler collection. Completed
        # prompts and already dispatched requests cannot wait behind this ticket.
        pending = {}
        for r in [
            *sched.running,
            *getattr(sched, "waiting", ()),
            *getattr(sched, "skipped_waiting", ()),
        ]:
            rid = getattr(r, "request_id", id(r))
            if rid not in inflight and (
                getattr(r, "num_prompt_tokens", 0)
                > getattr(r, "num_computed_tokens", 0)
            ):
                pending[rid] = r
        return list(pending.values())

    def _shares_prefill(self, first, second):
        a = self.streams.get((first.model_id, "prefill"), self.streams.get("prefill"))
        b = self.streams.get((second.model_id, "prefill"), self.streams.get("prefill"))
        # Remote workers use one broadcast prefill stream per rank.
        return a is b or (
            a is not None and b is not None and a.cuda_stream == b.cuda_stream
        )

    def _estimate_ticket(self, model) -> tuple[int, int]:
        """Budget-capped tokens; all requests delayed on the same CUDA stream."""
        sched = model.core.scheduler
        budget = (
            getattr(sched, "_hb_step_token_budget", None)
            or getattr(sched, "max_num_scheduled_tokens", 0)
            or (1 << 30)
        )
        pending = self._pending_prefills(model)
        tokens = 0
        remaining = 0
        for r in pending:
            work = r.num_prompt_tokens - r.num_computed_tokens
            admitted = min(work, max(0, budget - tokens))
            tokens += admitted
            # A partially served prompt still waits behind this chunk.
            remaining += int(admitted < work)
        remaining += sum(
            len(self._pending_prefills(other))
            for other in self.models
            if other.model_id != model.model_id and self._shares_prefill(model, other)
        )
        return int(tokens or min(budget, 1 << 14)), remaining

    def prefill_priority_indices(self, indices):
        """Bound partner waiting by one predicted generator-prefill service time.

        This is dispatch fairness, separate from mask cost. Existing generation
        priority is retained until a partner has waited that measured duration;
        ready/KV checks still belong to the controller. No fixed age threshold.
        """
        self.shared_prefill_age_guard = None
        if self.calibration is None or self.force_indices is not None:
            return indices
        primary = next(
            (
                m
                for m in self.models
                if m.model_id == (self.calibrated_model or self.plan.model_id)
            ),
            None,
        )
        if primary is None:
            return indices
        tokens, waiting = self._estimate_ticket(primary)
        n = sum(
            getattr(r, "num_computed_tokens", 0) >= getattr(r, "num_prompt_tokens", 1)
            for m in self.models
            if getattr(m, "decode_enabled", True)
            for r in m.core.scheduler.running
        )
        candidate, _, _ = self.calibration.choose(tokens, n, waiting)
        duration = self.calibration.duration(candidate, tokens, n)
        now = self.clock()
        urgent = {}
        for i in indices:
            other = self.models[i]
            if other is primary or not self._shares_prefill(primary, other):
                continue
            age = max(
                (
                    max(0, now - getattr(r, "arrival_time", now))
                    for r in self._pending_prefills(other)
                ),
                default=0,
            )
            if age >= duration:
                urgent[i] = age
        if urgent:
            self.shared_prefill_age_guard = {
                "threshold_s": duration,
                "candidate": candidate,
                "ages_s": {self.models[i].model_id: age for i, age in urgent.items()},
                "rule": "partner age >= predicted next generator prefill duration",
            }
            return sorted(indices, key=lambda i: (i not in urgent, -urgent.get(i, 0)))
        return indices

    def reset(self):
        if self.ctrl is None:
            return
        streams = {
            stream.cuda_stream: stream
            for key, stream in self.streams.items()
            if key == self.plan.lane
            or isinstance(key, tuple)
            and key[1] == self.plan.lane
        }
        for stream in streams.values():
            self.ctrl._set_mask(stream, 0, "unrestricted", log=False)
        if self.plan.lane == "decode":
            self.workers[self.plan.model_id].model_runner.select_decode_mask(False)
        self.state.clear()


def library_provenance(path: Path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
