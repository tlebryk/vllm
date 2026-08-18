"""Experimental Slack Serve GPU worker.

Keeps lane-specific stream routing out of vLLM's common GPU worker.
"""

import os

import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


def prefill_lane_names() -> list[str]:
    """Prefill lane names for the configured depth (default one lane)."""
    if os.environ.get("HB_P2_PREFILL_LANES") == "2":
        return ["prefill", "prefill1"]
    return ["prefill"]


class SlackServeGPUWorker(GPUWorker):

    def init_device(self) -> None:
        super().init_device()
        self.decode_stream = torch.cuda.Stream(device=self.device)
        # HB_P2_PREFILL_SHARED_STREAM=1 maps every prefill lane onto ONE
        # CUDA stream: the extra lane then only provides a second in-flight
        # ticket context (host prep pipelined behind the running ticket)
        # while the GPU executes prefill tickets strictly in order - no
        # added SM contention, and same-request KV ordering comes free.
        shared = os.environ.get("HB_P2_PREFILL_SHARED_STREAM") == "1"
        self.lane_streams = {}
        for index, lane in enumerate(prefill_lane_names()):
            if shared and index > 0:
                self.lane_streams[lane] = self.lane_streams["prefill"]
            else:
                self.lane_streams[lane] = torch.cuda.Stream(device=self.device)
        # Embed sidecar shares the prefill lane's stream; prefill-lane
        # launches and embed micro-steps are host-serialized by one FIFO
        # mutex (no kernel interleaving on the dense stream, no
        # forward-context race, launch-ahead comes free from the stream
        # FIFO). Runs on the executor thread — the controller never blocks.
        # Optional libsmctrl TPC placement masks (HB_SMCTRL_*). The embed
        # sidecar shares the prefill lane's stream, so the prefill mask
        # covers embed launches too.
        if os.environ.get("HB_SMCTRL_LIB"):
            from vllm.v1.worker.hb_smctrl import apply_env_masks
            apply_env_masks(
                {
                    "prefill": self.lane_streams["prefill"],
                    "decode": self.decode_stream,
                }
            )
        self._hb_dense_lock = None
        self._hb_record_prefill_tail = None
        if os.environ.get("HB_P2_EMBED_SIDECAR") == "1":
            from vllm.v1.engine.hb_embed_sidecar import (
                prefill_launch_lock_acquire,
                prefill_launch_lock_release,
                record_prefill_tail,
            )
            self._hb_dense_lock = (
                prefill_launch_lock_acquire,
                prefill_launch_lock_release,
            )
            self._hb_record_prefill_tail = record_prefill_tail

    # override factory hook to use new model
    def _create_model_runner(self):
        from vllm.v1.worker.gpu.slackserve_model_runner import (
            SlackServeModelRunner,
        )
        return SlackServeModelRunner(self.vllm_config, self.device)

    def _lane_stream(self, lane: str) -> torch.cuda.Stream:
        # ``default`` remains a compatibility alias for decode.
        if lane in ("decode", "default"):
            return self.decode_stream
        stream = self.lane_streams.get(lane)
        if stream is None:
            raise ValueError(f"Unknown execution lane: {lane!r}")
        return stream

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        lane = scheduler_output.execution_lane
        locked = (
            self._hb_dense_lock is not None
            and lane not in ("decode", "default")
        )
        if lane not in ("decode", "default"):
            # Load-conditional prefill mask (HB_SMCTRL_MASK_MIN_RUNNING):
            # re-mask/uncap the prefill stream before this chunk's launches.
            from vllm.v1.worker.hb_smctrl import maybe_mask_prefill_for_load

            maybe_mask_prefill_for_load(self._lane_stream(lane))
        if locked:
            # NVTX spans the lock acquire too, so a trace shows prefill's
            # wait behind an embed unit as the gap from range-start to the
            # first prefill kernel.
            torch.cuda.nvtx.range_push("dense/prefill")
            self._hb_dense_lock[0]()
        try:
            with torch.cuda.stream(self._lane_stream(lane)):
                output = super().execute_model(scheduler_output)
                if (
                    output is None
                    and scheduler_output.total_num_scheduled_tokens > 0
                ):
                    # Enqueue sampling while this ticket's lane-local inputs
                    # and request slots are still current. The returned
                    # AsyncOutput is waited by UniProcExecutor off-thread, so
                    # dispatch remains nonblocking and another lane can launch.
                    output = self.model_runner.sample_tokens(
                        None, lane=lane, return_async=True
                    )
                if locked:
                    self._hb_record_prefill_tail(self._lane_stream(lane))
                return output
        finally:
            if locked:
                self._hb_dense_lock[1]()
                torch.cuda.nvtx.range_pop()

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None", lane="decode"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # Sampling, prompt logprobs, the async D2H copy handoff, and
        # postprocess must all run on the same stream as the lane's forward.
        # AsyncOutput's stream() helper restores ``main_stream`` as the
        # ambient stream on exit, so running this on any other stream would
        # silently launch postprocess unordered w.r.t. the sampler.
        locked = (
            self._hb_dense_lock is not None
            and lane not in ("decode", "default")
        )
        if locked:
            self._hb_dense_lock[0]()
        try:
            with torch.cuda.stream(self._lane_stream(lane)):
                output = self.model_runner.sample_tokens(grammar_output, lane=lane)
                if locked:
                    self._hb_record_prefill_tail(self._lane_stream(lane))
                return output
        finally:
            if locked:
                self._hb_dense_lock[1]()
