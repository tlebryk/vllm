"""Experimental Slack Serve GPU worker.

Keeps lane-specific stream routing out of vLLM's common GPU worker.
"""

import os

import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class SlackServeGPUWorker(GPUWorker):

    def init_device(self) -> None:
        super().init_device()
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self.prefill_stream = torch.cuda.Stream(device=self.device)

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
        if lane == "prefill":
            return self.prefill_stream
        raise ValueError(f"Unknown execution lane: {lane!r}")

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        lane = scheduler_output.execution_lane
        with torch.cuda.stream(self._lane_stream(lane)):
            if os.environ.get("HB_LANE_DEBUG") == "1":
                print(
                    "lane=", lane,
                    "current=", torch.cuda.current_stream(self.device).cuda_stream,
                    "decode=", self.decode_stream.cuda_stream,
                    "prefill=", self.prefill_stream.cuda_stream,
                    flush=True,
                )
            return super().execute_model(scheduler_output)

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
