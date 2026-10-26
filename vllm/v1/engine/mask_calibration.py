# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup calibration for the prefill-stream mask policy.

The controller chooses, per prefill ticket, which TPC set (or none) the
prefill CUDA stream is pinned to. Instead of a hand-written per-model table,
the choice is driven by measurements taken on the real engine at startup:
for every candidate mask, the prefill rate while ``n`` requests decode and
the slowdown those decodes suffer, at a few batch sizes ``n``. Per ticket the
candidate with the smallest cost in seconds wins::

    cost(m) = D * (1 + n * dilation(m, n)),  D = tokens / prefill_rate(m, n)

``D`` is the TTFT the prefilled request pays; ``n * D * dilation`` is the
aggregate delay of the resident decodes during that prefill. No tunables.

Measurements are cached per deployment key (GPU, TPC topology, candidates,
model, dtype, TP, block size, token budget) and reused across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
    """[{"name": str, "indices": [int...] | None}] with distinct index sets."""
    seen: set[tuple[int, ...] | None] = set()
    out = []
    for item in raw:
        indices = item.get("indices")
        key = None if indices is None else tuple(sorted(int(i) for i in indices))
        if key in seen:
            continue
        seen.add(key)
        name = str(item.get("name") or ("none" if key is None else f"tpc{len(key)}"))
        out.append({"name": name, "indices": None if key is None else list(key)})
    if not out:
        raise ValueError("mask calibration needs at least one candidate")
    if all(c["indices"] is None for c in out):
        raise ValueError("mask calibration needs at least one TPC set candidate")
    return out


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
                "n": [float(c["n"]) for c in cells],
                "rate": [float(c["prefill_tok_s"]) for c in cells],
                "dilation": [float(c["dilation"]) for c in cells],
            }
        if not self.by_name:
            raise ValueError("empty calibration table")

    @classmethod
    def load(cls, path: Path) -> CalibrationTable:
        return cls(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(path)

    def cost(self, name: str, tokens: int, n: int) -> float:
        cand = self.by_name[name]
        rate = max(1.0, _interp(cand["n"], cand["rate"], n))
        dilation = max(0.0, _interp(cand["n"], cand["dilation"], n))
        duration = tokens / rate
        return duration * (1.0 + n * dilation)

    def choose(self, tokens: int, n: int) -> tuple[str, list[int] | None, float]:
        best = min(self.by_name, key=lambda name: self.cost(name, tokens, n))
        return best, self.by_name[best]["indices"], self.cost(best, tokens, n)

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
        return sum(
            1
            for r in self.records
            if r["lane"] == "decode"
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
    per_warmer = int(0.6 * kv_tokens / max(decode_sizes))
    warmer_prompt = max(256, min(WARMER_PROMPT_TOKENS, per_warmer * 2 // 3))
    warmer_output = max(128, min(WARMER_OUTPUT_TOKENS, per_warmer - warmer_prompt))
    logger.info(
        "mask calibration: KV %d tokens -> warmers %d prompt + %d output tokens",
        kv_tokens,
        warmer_prompt,
        warmer_output,
    )

    def wait_done(rid: str, what: str) -> None:
        driver.drive(lambda: not driver.active(rid), 300, what)

    try:
        results = [
            {"name": c["name"], "indices": c["indices"], "cells": []}
            for c in candidates
        ]

        def probe_cell(cand, result, n, alone_ms):
            """One candidate against the warmers already decoding at size n."""
            resident = driver.decode_ready()
            if resident < n:
                raise RuntimeError(
                    f"calibration lost warmers before probing {cand['name']} "
                    f"(n={n}, resident={resident}); raise WARMER_OUTPUT_TOKENS"
                )
            lane_masks.force_indices = (
                () if cand["indices"] is None else list(cand["indices"])
            )
            probe = submit(probe_tokens, 1)
            wait_done(probe, f"probing {cand['name']} n={n}")
            lane_masks.force_indices = None
            tickets = [
                r
                for r in driver.records
                if r["lane"] == "prefill" and probe in r["num_scheduled_tokens"]
            ]
            if not tickets:
                raise RuntimeError("calibration probe produced no prefill ticket")
            tokens = sum(r["num_scheduled_tokens"][probe] for r in tickets)
            seconds = sum(r["t_end"] - r["t_start"] for r in tickets)
            window_start = min(r["t_start"] for r in tickets)
            window_end = max(r["t_end"] for r in tickets)
            window_ms = (window_end - window_start) * 1000
            completions = driver.decode_completions(window_start, window_end)
            # Mean decode period inside the prefill window; zero completions
            # means decode was starved for the whole window.
            during_ms = window_ms / max(completions, 1)
            result["cells"].append(
                {
                    "n": n,
                    "prefill_tok_s": tokens / max(seconds, 1e-6),
                    "prefill_tickets": len(tickets),
                    "decode_alone_ms": alone_ms,
                    "decode_during_ms": during_ms,
                    "decode_samples": completions,
                    "window_ms": window_ms,
                    "dilation": max(0.0, during_ms / alone_ms - 1.0),
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
            lane_masks.force_indices = None
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
            lane_masks.force_indices = (
                () if cand["indices"] is None else list(cand["indices"])
            )
            wait_done(submit(probe_tokens, 1), "warming up")
            lane_masks.force_indices = None
            driver.records.clear()
        for n in decode_sizes:
            measure_size(n)
    finally:
        lane_masks.force_indices = None
        driver.close()
    return CalibrationTable(
        {
            "key": key,
            "created": started,
            "elapsed_s": time.time() - started,
            "probe_tokens": probe_tokens,
            "decode_sizes": list(decode_sizes),
            "candidates": results,
        }
    )
