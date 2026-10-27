# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup calibration for the prefill-stream mask policy.

The controller chooses, per prefill ticket, which TPC set (or none) the
prefill CUDA stream is pinned to. Instead of a hand-written per-model table,
the choice is driven by measurements taken on the real engine at startup:
for every candidate mask, the prefill rate while ``n`` requests decode and
the slowdown those decodes suffer, at a few batch sizes ``n``. Per ticket the
candidate with the smallest cost in seconds wins::

    cost(m) = D * (1 + W + n * dilation(m, n)),  D = tokens / prefill_rate(m, n)

``D`` is the TTFT the prefilled request pays, ``W * D`` the delay of the W
requests queued behind it on the prefill stream, and ``n * D * dilation`` the
aggregate delay of the resident decodes during that prefill. No tunables.

The merged-batch candidate (``{"name": "merged", "mode": "merged"}``) is not a
mask: the controller runs the scheduler's stock mixed batch on the decode
stream instead of a prefill-stream ticket. Its progress pattern differs (the
resident decodes make one step per merged forward instead of being slowed by
a fraction), so its cost charges each resident decode the merged step's
excess over the decode-alone period, ``n * (S - T_d)`` with ``S = tokens /
merged_rate(n)``, instead of ``n * D * dilation``. Its cell records the
probe's blocked fraction ``1 - alone / during`` as ``dilation`` for reference.

Measurements are cached per deployment key (GPU, TPC topology, candidates,
model, dtype, TP, block size, token budget) and reused across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import random
import statistics
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DECODE_SIZES = (2, 8, 32)
DEFAULT_PROBE_TOKENS = 16384
# Warmers decode at a context comparable to the served workload (long
# prompts), so the measured dilation is that of attention-heavy decode steps.
WARMER_PROMPT_TOKENS = 8192
WARMER_OUTPUT_TOKENS = 4096


def normalize_candidates(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{"name": str, "indices": [int...] | None, "mode"?: "merged"}] with
    distinct index sets; at most one merged-batch candidate."""
    seen: set[Any] = set()
    out = []
    for item in raw:
        mode = item.get("mode", "split")
        if mode not in ("split", "merged"):
            raise ValueError("candidate mode must be split or merged")
        indices = item.get("indices")
        if mode == "merged":
            if indices is not None:
                raise ValueError("the merged-batch candidate takes no TPC set")
            key: Any = "merged"
        else:
            key = None if indices is None else tuple(sorted(int(i) for i in indices))
        if key in seen:
            continue
        seen.add(key)
        if mode == "merged":
            name = str(item.get("name") or "merged")
            out.append({"name": name, "indices": None, "mode": "merged"})
            continue
        name = str(item.get("name") or ("none" if key is None else f"tpc{len(key)}"))
        out.append({"name": name, "indices": None if key is None else list(key)})
    if not out:
        raise ValueError("mask calibration needs at least one candidate")
    if all(c["indices"] is None for c in out):
        raise ValueError("mask calibration needs at least one TPC set candidate")
    if len({c["name"] for c in out}) != len(out):
        raise ValueError("mask calibration candidates need distinct names")
    return out


def is_merged_candidate(cand: dict[str, Any]) -> bool:
    return cand.get("mode") == "merged"


def calibration_key(fields: dict[str, Any]) -> str:
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if not xs:
        raise ValueError("no calibration cells")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class CalibrationTable:
    """Measured prefill rate and decode dilation per candidate and batch size."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.by_name: dict[str, dict[str, Any]] = {}
        for cand in data["candidates"]:
            cells = sorted(cand["cells"], key=lambda c: c["n"])
            self.by_name[cand["name"]] = {
                "indices": cand["indices"],
                "mode": cand.get("mode", "split"),
                "n": [float(c["n"]) for c in cells],
                "rate": [float(c["prefill_tok_s"]) for c in cells],
                "dilation": [float(c["dilation"]) for c in cells],
                "alone_ms": [float(c.get("decode_alone_ms", 0.0)) for c in cells],
            }
        if not self.by_name:
            raise ValueError("empty calibration table")
        self.has_merged = any(c["mode"] == "merged" for c in self.by_name.values())

    def is_merged(self, name: str) -> bool:
        return self.by_name[name]["mode"] == "merged"

    @classmethod
    def load(cls, path: Path) -> CalibrationTable:
        return cls(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(path)

    def cost(self, name: str, tokens: int, n: int, waiting: int = 0) -> float:
        """Seconds of request latency this ticket adds under candidate ``name``:
        its own prefill duration D (the prefilled request's TTFT), D again for
        every request queued behind it on the prefill stream, and D × dilation
        for each of the n resident decodes."""
        cand = self.by_name[name]
        rate = max(1.0, _interp(cand["n"], cand["rate"], n))
        duration = tokens / rate
        if cand["mode"] == "merged":
            # Blocked progress: the resident decodes make one step during the
            # merged step instead of one per decode-alone period, so each
            # loses (step - alone); this depends on the ticket's length, not
            # on the probe's blocked fraction.
            alone = self.decode_alone_ms(n) / 1000
            return duration * (1.0 + waiting) + n * max(0.0, duration - alone)
        dilation = max(0.0, _interp(cand["n"], cand["dilation"], n))
        return duration * (1.0 + waiting + n * dilation)

    def duration(self, name: str, tokens: int, n: int) -> float:
        """Measured service time, without queue or decode-delay weighting."""
        cand = self.by_name[name]
        return tokens / max(1.0, _interp(cand["n"], cand["rate"], n))

    def decode_alone_ms(self, n: float) -> float:
        """Decode step period with n resident requests and no prefill (the
        warmers' baseline, one per batch size, shared by all candidates)."""
        cand = next(iter(self.by_name.values()))
        return max(1e-3, _interp(cand["n"], cand["alone_ms"], n))

    def progress(
        self,
        tokens: int,
        n: int,
        *,
        horizon_s: float | None = None,
        queued_tokens: int = 0,
        queued_requests: int = 0,
        work_unit: int | None = None,
        remaining_output_tokens: float | None = None,
        incoming_output_tokens: float = 0.0,
    ):
        """Paper Section 2: output tokens each candidate produces over one
        common window, for B = n resident decodes and an incoming prefill of
        ``tokens``. Spatial sharing under mask m: B rounds of concurrent decode
        at T_d^par(m) during its prefill T_x^par(m), the new request's first
        token, then (B+1)-request decode at T_d(B+1) for the rest of the
        window. Merged batching: one hybrid step T_m, then (B+1)-request
        decode. Fractional rounds; the window is the slowest candidate's
        prefill unless ``horizon_s`` is given.

        With ``queued_tokens`` / ``queued_requests`` (the prefill work queued
        behind this ticket on the same stream) the window instead covers the
        remaining prefill work R = tokens + queued: the prefill stream stays
        busy for R / rate(m), the k = 1 + queued_requests completing prefills
        each add a first token and grow the batch (mean batch B + (k-1)/2
        while busy, B + k after), and the merged batch serves R in
        ceil(R / work_unit) steps, each advancing every resident decode once.
        With no queue this is exactly the single-ticket rule.

        ``remaining_output_tokens`` (the residents' remaining prescribed output,
        max_tokens minus generated) bounds the decode credit: Section 2 assumes
        no departures, but short outputs finish during a long prefill, after
        which no candidate earns decode progress from them. The incoming
        prefill's requests add ``incoming_output_tokens`` after their first
        token. Returns {name: tokens} plus the per-candidate timings used."""
        alone_now = self.decode_alone_ms(n) / 1000
        k = 1 + max(0, int(queued_requests))
        total = tokens + max(0, int(queued_tokens))
        t_next = self.decode_alone_ms(n + k) / 1000
        unit = max(1, int(work_unit or tokens or 1))
        busy_batch = n + (k - 1) / 2.0
        timings = {}
        for name, cand in self.by_name.items():
            rate = max(1.0, _interp(cand["n"], cand["rate"], n))
            busy_s = total / rate
            if cand["mode"] == "merged":
                timings[name] = {
                    "mode": "merged",
                    "step_s": busy_s,
                    "steps": max(1, math.ceil(total / unit)),
                }
            else:
                dilation = max(0.0, _interp(cand["n"], cand["dilation"], n))
                timings[name] = {
                    "mode": "split",
                    "prefill_s": busy_s,
                    "decode_par_s": alone_now * (1.0 + dilation),
                }
        window = horizon_s
        if window is None:
            window = max(
                t.get("prefill_s", t.get("step_s", 0.0)) for t in timings.values()
            )
        pool_busy = pool_post = math.inf
        if remaining_output_tokens is not None:
            # Decode work that exists: residents' remaining tokens while the
            # prefill runs (plus half the queued requests' outputs as they
            # join), then everything including the incoming requests' outputs.
            pool_busy = remaining_output_tokens + (k - 1) / 2.0 * incoming_output_tokens
            pool_post = remaining_output_tokens + k * incoming_output_tokens
        out = {}
        for name, t in timings.items():
            if t["mode"] == "merged":
                remaining = max(0.0, window - t["step_s"])
                busy = busy_batch * t["steps"]
            else:
                remaining = max(0.0, window - t["prefill_s"])
                busy = busy_batch * t["prefill_s"] / t["decode_par_s"]
            busy = min(busy, pool_busy)
            after = min((n + k) * remaining / t_next, max(0.0, pool_post - busy))
            out[name] = busy + k + after
        return out, {
            "window_s": window,
            "decode_next_s": t_next,
            "queued_tokens": max(0, int(queued_tokens)),
            "queued_requests": max(0, int(queued_requests)),
            "remaining_output_tokens": remaining_output_tokens,
            "incoming_output_tokens": incoming_output_tokens,
            **timings,
        }

    def choose(
        self,
        tokens: int,
        n: int,
        waiting: int = 0,
        *,
        exclude_merged: bool = False,
        rule: str = "queue",
        queued_tokens: int = 0,
        work_unit: int | None = None,
        remaining_output_tokens: float | None = None,
        incoming_output_tokens: float = 0.0,
    ) -> tuple[str, list[int] | None, float]:
        """Best candidate name, its TPC set (None for unrestricted and for the
        merged batch; see ``is_merged``) and its score: seconds of cost under
        the ``queue`` rule (D × (1 + W + n × dilation)), output tokens over
        the common window under the ``progress`` rule (Section 2; W unused)
        or over the remaining-prefill-work window under ``progress_queue``
        (W queued requests holding ``queued_tokens``)."""
        names = [
            name
            for name, cand in self.by_name.items()
            if not (exclude_merged and cand["mode"] == "merged")
        ]
        if rule in ("progress", "progress_queue"):
            gains, timings = self.progress(
                tokens,
                n,
                queued_tokens=queued_tokens if rule == "progress_queue" else 0,
                queued_requests=waiting if rule == "progress_queue" else 0,
                work_unit=work_unit,
                remaining_output_tokens=remaining_output_tokens,
                incoming_output_tokens=incoming_output_tokens,
            )
            # Ties go to the sooner first token.
            best = max(
                names,
                key=lambda name: (
                    gains[name],
                    -timings[name].get("prefill_s", timings[name].get("step_s", 0.0)),
                ),
            )
            return best, self.by_name[best]["indices"], gains[best]
        if rule != "queue":
            raise ValueError("mask policy rule must be queue, progress or progress_queue")
        best = min(names, key=lambda name: self.cost(name, tokens, n, waiting))
        return best, self.by_name[best]["indices"], self.cost(best, tokens, n, waiting)

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.data.get("key"),
            "created": self.data.get("created"),
            "probe_tokens": self.data.get("probe_tokens"),
            "candidates": {
                cand["name"]: [
                    {
                        "n": c["n"],
                        "prefill_tok_s": round(c["prefill_tok_s"]),
                        "decode_alone_ms": round(c["decode_alone_ms"], 2),
                        "decode_during_ms": round(c["decode_during_ms"], 2),
                        "dilation": round(c["dilation"], 3),
                        "progress": c.get("progress", "dilated"),
                    }
                    for c in cand["cells"]
                ]
                for cand in self.data["candidates"]
            },
        }


class _Driver:
    """Runs the controller loop on the backend thread until a condition holds."""

    def __init__(self, backend, controller, model_id: str):
        self.backend, self.controller, self.model_id = backend, controller, model_id
        self.records: list[dict[str, Any]] = []
        self._previous = controller.step_observer

        def observe(record):
            self.records.append(record)
            if self._previous is not None:
                self._previous(record)

        controller.step_observer = observe

    def close(self):
        self.controller.step_observer = self._previous

    def drive(self, until, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while not until():
            self.backend._process_inputs(self.controller)
            for q in self.backend.outputs.values():
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            if self.backend.output_has_headroom() and self.controller.advance():
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"mask calibration timed out while {what}")
            time.sleep(0.0005)

    def decode_ready(self) -> int:
        sched = self.backend.cores[self.model_id].scheduler
        return sum(
            1 for r in sched.running if r.num_computed_tokens >= r.num_prompt_tokens
        )

    def active(self, request_id: str) -> bool:
        with self.backend.lock:
            return (self.model_id, request_id) in self.backend.active

    def decode_completions(self, t0: float, t1: float) -> int:
        # A merged ("default") ticket advances every resident decode by one
        # step too, so it counts as one decode completion.
        return sum(
            1
            for r in self.records
            if r["lane"] in ("decode", "default")
            and r["scheduled_tokens"]
            and t0 <= r["t_end"] <= t1
        )

    def decode_periods(self, t0: float, t1: float) -> list[float]:
        ends = sorted(
            r["t_end"]
            for r in self.records
            if r["lane"] == "decode"
            and r["scheduled_tokens"]
            and t0 <= r["t_end"] <= t1
        )
        return [(b - a) * 1000 for a, b in zip(ends, ends[1:])]


def _make_request(request_id: str, prompt: list[int], max_tokens: int):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens, ignore_eos=True, temperature=0.0
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def run_calibration(
    backend,
    controller,
    lane_masks,
    model_id: str,
    candidates: list[dict[str, Any]],
    *,
    key: str,
    decode_sizes=DEFAULT_DECODE_SIZES,
    probe_tokens: int = DEFAULT_PROBE_TOKENS,
    seed: int = 0,
) -> CalibrationTable:
    """Measure every candidate at every decode batch size on the live engine.

    Per cell: admit ``n`` short-prompt requests and wait until they decode;
    time decode tickets alone; force the candidate mask and submit one
    ``probe_tokens`` prompt; take its prefill rate from the prefill tickets
    and the decode ticket period while they run; cancel the warmers.
    """
    core = backend.cores[model_id]
    vocab = core.vllm_config.model_config.get_vocab_size()
    rng = random.Random(seed)
    low, high = min(1000, vocab // 4), vocab - 1
    driver = _Driver(backend, controller, model_id)
    started = time.time()
    counter = 0

    def prompt(n: int) -> list[int]:
        return [rng.randint(low, high) for _ in range(n)]

    def submit(prompt_tokens: int, max_tokens: int) -> str:
        nonlocal counter
        counter += 1
        rid = f"calib:{key[:8]}:{counter}"
        backend.submit(model_id, _make_request(rid, prompt(prompt_tokens), max_tokens))
        return rid

    # Warmers must stay resident for a whole size: size them from the KV the
    # engine actually has (their prompts + outputs take at most 60% of it at
    # the largest n, leaving room for the probe), capped at the defaults.
    pool = core.scheduler.kv_cache_manager.block_pool
    kv_tokens = pool.get_num_free_blocks() * core.scheduler.block_size

    def warmer_shape(n: int) -> tuple[int, int]:
        """Warmer prompt/output tokens for batch size n: the n warmers take at
        most 60% of KV, so large n means shorter context (the only resident
        state a large batch can have on this KV)."""
        per_warmer = int(0.6 * kv_tokens / n)
        prompt_tokens = max(256, min(WARMER_PROMPT_TOKENS, per_warmer * 2 // 3))
        output_tokens = max(
            128, min(WARMER_OUTPUT_TOKENS, per_warmer - prompt_tokens)
        )
        return prompt_tokens, output_tokens

    for n in decode_sizes:
        logger.info(
            "mask calibration: KV %d tokens -> n=%d warmers %d prompt + %d output",
            kv_tokens,
            n,
            *warmer_shape(n),
        )
    current_warmer = {"prompt": 0, "output": 0}

    def wait_done(rid: str, what: str) -> None:
        driver.drive(lambda: not driver.active(rid), 300, what)

    try:
        results = [
            {"name": c["name"], "indices": c["indices"], "cells": []}
            for c in candidates
        ]

        def force(cand):
            """Pin the controller to exactly this candidate for the probe."""
            if is_merged_candidate(cand):
                lane_masks.force_indices = None
                lane_masks.force_merged = True
            else:
                lane_masks.force_merged = False
                lane_masks.force_indices = (
                    () if cand["indices"] is None else list(cand["indices"])
                )

        def release():
            lane_masks.force_indices = None
            lane_masks.force_merged = None

        def probe_cell(cand, result, n, alone_ms):
            """One candidate against the warmers already decoding at size n."""
            resident = driver.decode_ready()
            if resident < n:
                raise RuntimeError(
                    f"calibration lost warmers before probing {cand['name']} "
                    f"(n={n}, resident={resident}); raise WARMER_OUTPUT_TOKENS"
                )
            merged = is_merged_candidate(cand)
            force(cand)
            probe = submit(probe_tokens, 1)
            wait_done(probe, f"probing {cand['name']} n={n}")
            release()
            lane = "default" if merged else "prefill"
            tickets = [
                r
                for r in driver.records
                if r["lane"] == lane and probe in r["num_scheduled_tokens"]
            ]
            if not tickets:
                raise RuntimeError(f"calibration probe produced no {lane} ticket")
            tokens = sum(r["num_scheduled_tokens"][probe] for r in tickets)
            # Merged tickets also carry the n decode tokens; their duration is
            # the real cost of prefilling in that mode.
            seconds = sum(r["t_end"] - r["t_start"] for r in tickets)
            window_start = min(r["t_start"] for r in tickets)
            window_end = max(r["t_end"] for r in tickets)
            window_ms = (window_end - window_start) * 1000
            completions = driver.decode_completions(window_start, window_end)
            # Mean decode period inside the prefill window; zero completions
            # means decode was starved for the whole window.
            during_ms = window_ms / max(completions, 1)
            if merged:
                # Blocked progress: decodes make one step per merged forward.
                dilation = max(0.0, 1.0 - alone_ms / max(during_ms, 1e-6))
            else:
                dilation = max(0.0, during_ms / alone_ms - 1.0)
            result["cells"].append(
                {
                    "n": n,
                    "prefill_tok_s": tokens / max(seconds, 1e-6),
                    "prefill_tickets": len(tickets),
                    "decode_alone_ms": alone_ms,
                    "decode_during_ms": during_ms,
                    "decode_samples": completions,
                    "window_ms": window_ms,
                    "dilation": dilation,
                    "progress": "blocked" if merged else "dilated",
                    "warmer_prompt_tokens": current_warmer["prompt"],
                    "warmer_output_tokens": current_warmer["output"],
                }
            )
            logger.info(
                "mask calibration %s n=%d: prefill %.0f tok/s, decode %.2f -> %.2f ms",
                cand["name"],
                n,
                result["cells"][-1]["prefill_tok_s"],
                alone_ms,
                during_ms,
            )

        def measure_size(n):
            """Admit n warmers once, time decode alone once, probe every
            candidate against the same warmers, then release them."""
            release()
            warmer_prompt, warmer_output = warmer_shape(n)
            current_warmer.update(prompt=warmer_prompt, output=warmer_output)
            warmers = [submit(warmer_prompt, warmer_output) for _ in range(n)]
            driver.drive(lambda: driver.decode_ready() >= n, 180, f"warming n={n}")
            t0 = time.perf_counter()
            driver.drive(
                lambda: len(driver.decode_periods(t0, time.perf_counter())) >= 12,
                60,
                f"timing decode alone n={n}",
            )
            alone_ms = statistics.median(driver.decode_periods(t0, time.perf_counter()))
            for cand, result in zip(candidates, results):
                probe_cell(cand, result, n, alone_ms)
            backend.cancel(model_id, warmers)
            driver.drive(
                lambda: (
                    not controller.has_work()
                    and not any(driver.active(w) for w in warmers)
                ),
                120,
                "draining warmers",
            )
            driver.records.clear()

        for cand in candidates:
            # Warm-up: the first forward under each candidate pays one-time
            # allocations/lazy init that must not be charged to a cell.
            force(cand)
            wait_done(submit(probe_tokens, 1), "warming up")
            release()
            driver.records.clear()
        for n in decode_sizes:
            measure_size(n)
    finally:
        lane_masks.force_indices = None
        lane_masks.force_merged = None
        driver.close()
    return CalibrationTable(
        {
            "key": key,
            "created": started,
            "elapsed_s": time.time() - started,
            "probe_tokens": probe_tokens,
            "decode_sizes": list(decode_sizes),
            "candidates": [
                {**r, "mode": "merged"} if is_merged_candidate(c) else r
                for c, r in zip(candidates, results)
            ],
        }
    )
