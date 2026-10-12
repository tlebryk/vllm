# SPDX-License-Identifier: Apache-2.0
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
    ):
        self.plan, self.models, self.workers = plan, models, workers
        self.streams, self.total_tpcs, self.ctrl = streams, total_tpcs, ctrl
        if (
            plan.model_id not in workers and plan.model_id != "*"
        ) or not 0 < plan.tpcs < total_tpcs:
            raise ValueError(
                "mask model must exist and TPC count must be below full GPU"
            )
        if (
            len(indices) != plan.tpcs
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 or i >= total_tpcs for i in indices)
        ):
            raise ValueError(
                "explicit mask must contain the requested number of distinct TPCs"
            )
        self.indices = indices
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
        masked = (
            (model.model_id == self.plan.model_id or self.plan.model_id == "*")
            and lane == self.plan.lane
            and protect
            and not age_override
        )
        if lane == self.plan.lane:
            stream = self.streams.get((model.model_id, lane), self.streams.get(lane))
            if self.state.get(stream.cuda_stream) != masked:
                # Zero means completely unrestricted, including the upper TPCs.
                bits = sum(1 << i for i in self.indices)
                mask = (~bits & ((1 << 128) - 1)) if masked else 0
                self.ctrl._set_mask(stream, mask, "model-owned mask", log=False)
                self.state[stream.cuda_stream] = masked
            if lane == "decode" and model.model_id == self.plan.model_id:
                self.workers[model.model_id].model_runner.select_decode_mask(masked)
        return {
            "masked": masked,
            "enabled_tpcs": self.plan.tpcs if masked else self.total_tpcs,
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

    def reset(self):
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
