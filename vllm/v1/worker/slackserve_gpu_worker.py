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
        self.lane_streams = {
            lane: torch.cuda.Stream(device=self.device)
            for lane in prefill_lane_names()
        }

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
                return self.model_runner.sample_tokens(
                    None, lane=lane, return_async=True
                )
            return output

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None", lane="decode"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # Sampling, prompt logprobs, the async D2H copy handoff, and
        # postprocess must all run on the same stream as the lane's forward.
        # AsyncOutput's stream() helper restores ``main_stream`` as the
        # ambient stream on exit, so running this on any other stream would
        # silently launch postprocess unordered w.r.t. the sampler.
        with torch.cuda.stream(self._lane_stream(lane)):
            return self.model_runner.sample_tokens(grammar_output, lane=lane)
