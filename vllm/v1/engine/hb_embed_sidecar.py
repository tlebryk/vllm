# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slack Serve two-model SERVE sidecar (Thread C) — shared dense stream.

An in-process Qwen3-Embedding-4B pooling engine that lives inside the serve
EngineCore process and backfills the P2 two-lane controller's slack with a
fixed embedding queue. Active only when HB_P2_EMBED_SIDECAR=1; a strict no-op
otherwise.

ARCHITECTURE (2 streams total; the validated offline-bridge discipline)
-----------------------------------------------------------------------
Decode runs FULL-cudagraph on its own stream, untouched. Prefill and embed
SHARE the prefill lane's CUDA stream, and their eager forwards are
HOST-SERIALIZED by one launch mutex (DENSE_LAUNCH_LOCK), equal priority,
FIFO — exactly the structure the offline 3-model bridge validated ("prefill
and embed host-serialized on one dense stream, decode on the other").

The mutex is the whole mechanism. It simultaneously guarantees:
  * no per-kernel interleaving on the dense stream (two free-running
    enqueuers interleave EVERY kernel — probe 275206 — and that interleaving
    is what produced the FA3 TMA crash in the free-running design);
  * no forward-context race (the module-global is only touched by one eager
    forward at a time; decode's graph-replay path is empirically immune);
  * launch-ahead: the winner enqueues its unit while the loser's previous
    kernels are still executing, so the dense stream never idles between
    units — the stream FIFO is the scheduler.

There is NO overlap dispatch policy: no yield-first, staleness guard, stream
drain, or ticket gate. Optional idle uncapping only changes embed's SM target
after LLM work ends. During overlap, the only contention is a lock handoff on
the executor thread.

Embed's pooling execute has an inline GPU sync (readback), so the embed
thread holds the mutex for its full micro-step; budget 4096 keeps that unit
small. The drain starts only once the LLM is under real load
(HB_P2_EMBED_START_MIN_REQS, default 8) so no embed work escapes the
measured window during harness warmup.

UVA: the embed engine's stock runner must NOT register its staged-buffer
pools in the module-global dirty registry (only the LLM runner fences, on
its own streams — the cross-engine iterate/mutate race of jobs 274982/
275003). The drain thread disables registration for itself via
set_uva_fencing_enabled(False); embed pools then behave exactly like the
ordinary unfenced single-lane runner.

Build happens ONCE, eagerly, during EngineCore.__init__ (before the busy
loop, so no live forward to race and the ~30-60 s load stays out of served
TTFT). The embed engine's core is forced in-process (InprocClient).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# One launch mutex for the shared dense stream: prefill's eager forward
# (SlackServeGPUWorker.execute_model, prefill lanes) and embed micro-steps
# acquire it around their enqueue windows. Plain Lock, equal priority.
DENSE_LAUNCH_LOCK = threading.Lock()

# CUDA event recorded on the dense stream at the end of every prefill launch
# window (single writer, under the lock). The embed thread starts a unit only
# once this has fired: its inline pooling sync then waits only on its OWN
# ~50ms of kernels, never on a prefill unit ahead of it — a lock hold that
# includes a prefill wait blocks the prefill executor worker ~530ms and
# queues decode dispatch behind it (jobs 275449-52: p99 ITL ~500ms). The
# poll gap between prefill-end and embed-launch is ~1-2ms of Python.
PREFILL_TAIL = {"event": None}


def record_prefill_tail(stream: torch.cuda.Stream) -> None:
    """Worker-side (under the lock): mark the end of a prefill launch."""
    event = PREFILL_TAIL["event"]
    if event is None:
        event = torch.cuda.Event()
    event.record(stream)
    PREFILL_TAIL["event"] = event


def prefill_tail_clear() -> bool:
    """True once the last prefill unit's kernels have retired."""
    event = PREFILL_TAIL["event"]
    return event is None or event.query()

# Telemetry (GIL-atomic enough for counters; read at drain report time).
LOCK_STATS = {
    "prefill_wait_ms": 0.0,
    "prefill_waits": 0,
    "prefill_max_wait_ms": 0.0,
    "embed_wait_ms": 0.0,
    "embed_waits": 0,
    "embed_max_wait_ms": 0.0,
}


def _lock_acquire(side: str) -> None:
    t0 = time.perf_counter()
    DENSE_LAUNCH_LOCK.acquire()
    waited = (time.perf_counter() - t0) * 1000
    LOCK_STATS[f"{side}_wait_ms"] += waited
    if waited > 1.0:
        LOCK_STATS[f"{side}_waits"] += 1
        if waited > LOCK_STATS[f"{side}_max_wait_ms"]:
            LOCK_STATS[f"{side}_max_wait_ms"] = waited


# Fairness: threading.Lock has no FIFO handoff, and the embed drain loop
# re-acquires within microseconds of releasing while the prefill executor
# thread is still being woken — prefill then loses 3-5 consecutive handoffs
# (jobs 275475-77: embed unit p50 65ms but prefill wait avg ~250ms, max
# ~525ms). Prefill announces itself here; the embed loop defers while any
# prefill launcher is waiting, bounding prefill's wait to ONE embed unit.
PREFILL_WAITING = {"n": 0}


def prefill_launch_lock_acquire() -> None:
    """Called by SlackServeGPUWorker around a prefill-lane launch."""
    PREFILL_WAITING["n"] += 1
    try:
        _lock_acquire("prefill")
    finally:
        PREFILL_WAITING["n"] -= 1


def prefill_launch_lock_release() -> None:
    DENSE_LAUNCH_LOCK.release()


def _model_runner(engine: Any) -> Any:
    worker = engine.engine_core.engine_core.model_executor.driver_worker
    worker = getattr(worker, "worker", worker)
    return worker.model_runner


def _count_marked_linears(engine: Any) -> int:
    runner = _model_runner(engine)
    model = getattr(runner, "model", None)
    if model is None:
        model = runner.get_model()
    return sum(
        1 for m in model.modules() if getattr(m, "_hb_embed_sm_capped", False)
    )


class HbEmbedSidecar:
    """Owns the in-process embed engine, its queue, and the drain thread."""

    def __init__(self, host_core: Any) -> None:
        self.host_core = host_core
        self.engine: Any = None

        self.embed_model = os.environ.get(
            "HB_P2_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"
        )
        self.embed_gmu = float(os.environ.get("HB_P2_EMBED_GMU", "0.18"))
        # 4096: the embed micro-step is the unit of dense-stream FIFO
        # granularity; a 4K step keeps the worst prefill lock handoff ~40ms.
        self.embed_budget = int(os.environ.get("HB_P2_EMBED_BUDGET", "4096"))
        self.embed_max_model_len = int(os.environ.get("HB_P2_EMBED_MAX_MODEL_LEN", "2048"))
        self.embed_max_num_seqs = int(os.environ.get("HB_P2_EMBED_MAX_NUM_SEQS", "512"))
        self.embed_n = int(os.environ.get("HB_P2_EMBED_N", "512"))
        self.embed_queue_path = os.environ.get(
            "HB_P2_EMBED_QUEUE",
            "/mnt/weka/theo/heterogenious-batching-vllm/data/prompts/embed_rag.jsonl",
        )
        self.embed_sm_target = int(os.environ.get("HB_P2_EMBED_SM_TARGET", "96"))
        # Protect live LLM work, but do not cap an embed-only tail.
        self.embed_uncap_when_llm_idle = (
            os.environ.get("HB_P2_EMBED_UNCAP_IDLE", "0") == "1"
        )
        # Async scheduling pipelines the embed engine's own steps: one
        # get_output() = launch step k+1, then harvest step k (whose forward
        # and D2H already overlapped via AsyncPoolingOutput's copy stream).
        # The lock hold shrinks from launch+full-GPU-sync to launch+residual
        # (~25ms), so back-to-back units keep the dense stream fed and the
        # worst prefill wait behind embed roughly halves. Stock machinery —
        # the sync path was just async_scheduling=False calling
        # AsyncPoolingOutput.get_output() inline.
        self.embed_async_sched = os.environ.get("HB_P2_EMBED_ASYNC_SCHED", "1") == "1"
        # Don't start draining before the LLM is under real load, so no embed
        # work escapes the measured window during harness warmup (the bench's
        # initial single-prompt test would otherwise free-run the queue).
        self.start_min_reqs = int(os.environ.get("HB_P2_EMBED_START_MIN_REQS", "8"))
        # HB_P2_EMBED_PHASE=post_prefill: embed runs ONLY once the LLM's
        # prefill work is exhausted (all prompts admitted and computed), then
        # free-runs overlapped with the decode-only region + tail. Measured
        # price ladder per embed-second: decode-only overlap <<1 (PE regime),
        # tail =1.0, admission-phase overlap 0.99-1.27 (worst) — so never
        # place embed during admission. Default remains "any" (gap-fitting).
        self.phase = os.environ.get("HB_P2_EMBED_PHASE", "any")
        self.prefill_exhausted = False

        self.marked_linears = 0
        self.prompt_count = 0

        self._thread: threading.Thread | None = None
        self._started = False
        self._done = False
        self._pooled_seen: set[str] = set()
        self.embed_tokens_cum = 0
        # Three-stream token accounting: cumulative embed tokens, timestamped.
        toklog = os.environ.get("HB_TOKLOG")
        self._toklog = open(f"{toklog}.embed.jsonl", "a") if toklog else None
        self.pooled: set[str] = set()
        self.norm_samples: list[float] = []
        self.step_records: list[dict[str, Any]] = []
        self.drain_start = 0.0
        self.drain_end = 0.0
        self.idle_uncapped_steps = 0

    # -- build (eager, at EngineCore init) ---------------------------------
    def build(self) -> None:
        from vllm import LLMEngine
        from vllm.engine.arg_utils import EngineArgs
        from vllm.pooling_params import PoolingParams
        from vllm.envs import disable_envs_cache, enable_envs_cache
        from vllm.platforms import current_platform

        t0 = time.perf_counter()
        type(current_platform)._global_graph_pool = None

        pop_keys = (
            "HB_P2_EMBED_SIDECAR",
            "HB_P2_SERVE",
            "HB_LANE_ROUTING",
            "HB_P2_DECODE_GRAPHS_ONLY",
        )
        popped = {k: os.environ.pop(k) for k in pop_keys if k in os.environ}
        saved_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Clamp to >=1 so the Lt hook installs even when the starting target
        # is low (runtime switching needs it). Explicit 0 = hook fully off:
        # smctrl mask arms want plain torch kernels for embed GEMMs.
        if self.embed_sm_target > 0:
            os.environ["HB_EMBED_SM_COUNT_TARGET"] = str(self.embed_sm_target)
        else:
            os.environ.pop("HB_EMBED_SM_COUNT_TARGET", None)
        # Isolate the embed torch.compile cache: Qwen3-Embedding-4B and the
        # serve LLM Qwen3-4B are the SAME qwen3 arch and otherwise share vLLM's
        # compile cache; the embed build's (1, embed_budget) shape range then
        # shadows the LLM's, crashing an LLM prefill of 32768 with
        # 'Shape: 32768 out of considered ranges [(1,4096)]'.
        saved_cache_root = os.environ.get("VLLM_CACHE_ROOT")
        os.environ["VLLM_CACHE_ROOT"] = (
            f"/tmp/hb_embed_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        )
        disable_envs_cache()
        try:
            kwargs = dict(
                model=self.embed_model,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=self.embed_gmu,
                enforce_eager=False,
                enable_prefix_caching=False,
                max_model_len=self.embed_max_model_len,
                max_num_seqs=self.embed_max_num_seqs,
                max_num_batched_tokens=self.embed_budget,
                disable_log_stats=True,
                runner="pooling",
                async_scheduling=self.embed_async_sched,
                compilation_config={
                    "mode": 3,
                    "cudagraph_mode": "NONE",
                    "max_cudagraph_capture_size": 512,
                },
            )
            logger.info("[hb-embed-sidecar] building embed engine model=%s gmu=%.3f "
                        "budget=%d max_model_len=%d sm_target=%d",
                        self.embed_model, self.embed_gmu, self.embed_budget,
                        self.embed_max_model_len, self.embed_sm_target)
            self.engine = LLMEngine.from_engine_args(EngineArgs(**kwargs))
        finally:
            os.environ.pop("HB_EMBED_SM_COUNT_TARGET", None)
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            if saved_mp is not None:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = saved_mp
            os.environ.pop("VLLM_CACHE_ROOT", None)
            if saved_cache_root is not None:
                os.environ["VLLM_CACHE_ROOT"] = saved_cache_root
            for k, v in popped.items():
                os.environ[k] = v
            enable_envs_cache()

        core_name = type(self.engine.engine_core).__name__
        if "Inproc" not in core_name:
            raise RuntimeError(
                f"[hb-embed-sidecar] embed engine is not in-process: {core_name}"
            )
        self.marked_linears = _count_marked_linears(self.engine)
        if self.marked_linears <= 0 and self.embed_sm_target > 0:
            # With an explicit target of 0 the Lt hook is intentionally off
            # (smctrl mask arms use plain torch kernels for embed GEMMs).
            raise RuntimeError("[hb-embed-sidecar] SM hook marked no embed linears")

        # Flush any UVA pools the embed build/warmup registered on THIS
        # (controller) thread, while no LLM traffic exists to race with. The
        # drain thread then opts out of registration entirely.
        from vllm.v1.worker.gpu.buffer_utils import fence_uva_pools
        fence_uva_pools(None)

        rows = [
            json.loads(line)
            for line in Path(self.embed_queue_path).read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"[hb-embed-sidecar] empty embed queue: {self.embed_queue_path}")
        prompts = [rows[i % len(rows)] for i in range(self.embed_n)]
        tokenizer = self.engine.tokenizer
        params = PoolingParams(task="embed")
        self.req_tokens: dict[str, int] = {}
        for i, text in enumerate(prompts):
            ids = tokenizer.encode(text)
            self.req_tokens[f"hbembed_{i}"] = len(ids)
            self.engine.add_request(
                f"hbembed_{i}", {"prompt_token_ids": ids}, params.clone()
            )
        self.prompt_count = self.embed_n
        logger.info(
            "[hb-embed-sidecar] built in %.1fs: marked_linears=%d queued=%d core=%s",
            time.perf_counter() - t0, self.marked_linears, self.prompt_count,
            core_name,
        )

    # -- lazy drain-thread start (from step_two_lane, under real load) ------
    def maybe_start(self) -> None:
        if self.engine is None or self._started:
            return
        try:
            live = self.host_core.scheduler.get_num_unfinished_requests()
        except Exception:  # noqa: BLE001
            live = 0
        if live < self.start_min_reqs:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._drain, name="hb_embed_sidecar", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        from vllm.v1.worker.embed_sm_linear_hook import set_runtime_sm_target
        from vllm.v1.worker.gpu.buffer_utils import set_uva_fencing_enabled
        from vllm.utils.torch_utils import current_stream

        # Embed pools must not enter the module-global UVA fence registry
        # (only the LLM runner fences, on its own streams — cross-engine race).
        set_uva_fencing_enabled(False)

        # SHARED dense stream: the LLM prefill lane's stream, pinned as this
        # thread's ambient stream. Two streams total in the process.
        llm_worker = self.host_core.model_executor.driver_worker
        llm_worker = getattr(llm_worker, "worker", llm_worker)
        dense_stream = llm_worker.lane_streams["prefill"]
        torch.cuda.set_stream(dense_stream)
        runner = _model_runner(self.engine)
        runner.__dict__["main_stream"] = dense_stream
        assert current_stream().cuda_stream == dense_stream.cuda_stream

        engine = self.engine
        self.drain_start = time.perf_counter()
        try:
            while engine.has_unfinished_requests():
                # post_prefill phase: hold until the controller marks all
                # prompt KV computed, then free-run (decode-only + tail).
                if self.phase == "post_prefill" and not self.prefill_exhausted:
                    time.sleep(0.002)
                    continue
                # Fairness handoff: never outrace a waiting prefill launcher.
                if PREFILL_WAITING["n"] > 0:
                    time.sleep(0.0005)
                    continue
                # Launch only onto a quiescent dense stream (see PREFILL_TAIL).
                if not prefill_tail_clear():
                    time.sleep(0.001)
                    continue
                _lock_acquire("embed")
                # Re-validate INSIDE the lock: a prefill unit may have taken
                # the lock first and enqueued after our check above; without
                # this, the step stacks behind ~600ms of still-executing
                # prefill and holds the lock for its whole remainder (trace
                # 275502: 31/40 embed ranges at ~720ms vs ~65ms clean).
                if not prefill_tail_clear():
                    DENSE_LAUNCH_LOCK.release()
                    time.sleep(0.001)
                    continue
                idle_uncapped = (
                    self.embed_uncap_when_llm_idle
                    and self.host_core.scheduler.get_num_unfinished_requests() == 0
                )
                set_runtime_sm_target(0 if idle_uncapped else self.embed_sm_target)
                # smctrl arms: mirror the Lt uncap on the stream TPC mask so
                # an embed-only tail gets the whole GPU back (no-op unless
                # HB_SMCTRL_* is configured).
                from vllm.v1.worker.hb_smctrl import apply_prefill_uncap

                apply_prefill_uncap(dense_stream, idle_uncapped)
                if idle_uncapped:
                    self.idle_uncapped_steps += 1
                step_start = time.perf_counter()
                pooled_before = len(self.pooled)
                torch.cuda.nvtx.range_push("dense/embed")
                try:
                    # Enqueues on the shared dense stream behind whatever unit
                    # is already in flight; the pooling execute's inline sync
                    # completes the step before the lock is released.
                    stats = self._process_step()
                finally:
                    torch.cuda.nvtx.range_pop()
                    set_runtime_sm_target(None)
                    DENSE_LAUNCH_LOCK.release()
                wall = (time.perf_counter() - step_start) * 1000
                # Count embed tokens directly from the pooled requests' prompt
                # lengths (the SchedulerStats attr never resolved -> always 0).
                newly = set(self.pooled) - self._pooled_seen
                self._pooled_seen |= newly
                tokens = sum(
                    self.req_tokens.get(r)
                    or self.req_tokens.get(r.rsplit("-", 1)[0], 0)
                    for r in newly
                )
                self.embed_tokens_cum += tokens
                newly_pooled = len(newly)
                if newly_pooled > 0 or tokens > 0:
                    self.step_records.append(
                        {"t": time.perf_counter(), "wall_ms": wall,
                         "scheduled_tokens": tokens,
                         "embed_tokens_cum": self.embed_tokens_cum,
                         "newly_pooled": newly_pooled, "pooled": len(self.pooled)}
                    )
                    if self._toklog is not None:
                        self._toklog.write(json.dumps(
                            {"t": time.perf_counter(),
                             "embed_tokens_cum": self.embed_tokens_cum}) + "\n")
                        self._toklog.flush()
                    n = len(self.step_records)
                    if n <= 4 or n % 8 == 0:
                        logger.info(
                            "[hb-embed-sidecar] step #%d tokens=%d newly_pooled=%d "
                            "wall_ms=%.1f pooled=%d/%d", n, tokens, newly_pooled,
                            wall, len(self.pooled), self.prompt_count,
                        )
        finally:
            set_runtime_sm_target(None)
            self.drain_end = time.perf_counter()
            self._done = True
        self._report()

    def _process_step(self) -> Any:
        """One embed engine step (offline process_step pattern) + harvest."""
        engine = self.engine
        output = engine.engine_core.get_output()
        for item in output.outputs:
            if getattr(item, "pooling_output", None) is not None:
                self.pooled.add(item.request_id)
                if len(self.norm_samples) < 8:
                    try:
                        data = item.pooling_output.data
                        self.norm_samples.append(
                            float(torch.linalg.vector_norm(data.reshape(-1)).item())
                        )
                    except Exception:  # noqa: BLE001
                        self.norm_samples.append(float("nan"))
        processed = engine.output_processor.process_outputs(
            output.outputs, engine_core_timestamp=output.timestamp,
            iteration_stats=None,
        )
        engine.output_processor.update_scheduler_stats(output.scheduler_stats)
        engine.engine_core.abort_requests(processed.reqs_to_abort)
        return output.scheduler_stats

    def _report(self) -> None:
        wall_ms = (self.drain_end - self.drain_start) * 1000
        n_nan = sum(1 for x in self.norm_samples if x != x)
        total_tokens = sum(r["scheduled_tokens"] for r in self.step_records)
        frac = len(self.pooled) / self.prompt_count if self.prompt_count else 0.0
        logger.info(
            "[hb-embed-sidecar] HB_SIDECAR_DONE pooled=%d expected=%d drained_frac=%.3f "
            "wall_ms=%.1f steps=%d embed_tokens=%d budget=%d "
            "idle_uncapped_steps=%d "
            "lock{prefill_wait_ms=%.1f prefill_waits=%d prefill_max_ms=%.1f "
            "embed_wait_ms=%.1f embed_waits=%d embed_max_ms=%.1f} "
            "norm_samples=%s nan=%d",
            len(self.pooled), self.prompt_count, frac, wall_ms,
            len(self.step_records), total_tokens, self.embed_budget,
            self.idle_uncapped_steps,
            LOCK_STATS["prefill_wait_ms"], LOCK_STATS["prefill_waits"],
            LOCK_STATS["prefill_max_wait_ms"],
            LOCK_STATS["embed_wait_ms"], LOCK_STATS["embed_waits"],
            LOCK_STATS["embed_max_wait_ms"],
            [round(x, 3) for x in self.norm_samples], n_nan,
        )
        if len(self.pooled) != self.prompt_count:
            logger.warning(
                "[hb-embed-sidecar] embed drain incomplete: pooled=%d/%d (frac=%.3f)",
                len(self.pooled), self.prompt_count, frac,
            )
        if n_nan:
            logger.error("[hb-embed-sidecar] CORRECTNESS FAIL %d NaN norms", n_nan)
