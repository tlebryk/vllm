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

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        lane = scheduler_output.execution_lane
        if lane == "prefill":
            stream_context = torch.cuda.stream(self.prefill_stream)
        elif lane == "decode":
            stream_context = torch.cuda.stream(self.decode_stream)
        elif lane == "default":
            stream_context = torch.cuda.stream(self.decode_stream)
        else:
            raise ValueError(f"Unknown execution lane: {lane!r}")

        with stream_context:
            if os.environ.get("HB_LANE_DEBUG") == "1":
                print(
                    "lane=", lane,
                    "current=", torch.cuda.current_stream(self.device).cuda_stream,
                    "decode=", self.decode_stream.cuda_stream,
                    "prefill=", self.prefill_stream.cuda_stream,
                    flush=True,
                )
            if self.use_v2_model_runner:
                self.model_runner.__dict__["main_stream"] = torch.cuda.current_stream(
                    self.device
                )
            return super().execute_model(scheduler_output)
        
    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None", lane="decode"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output, lane=lane)


