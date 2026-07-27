"""Experimental Slack Serve GPU worker.

Keeps lane-specific stream routing out of vLLM's common GPU worker.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class SlackServeGPUWorker(GPUWorker):

    def init_device(self) -> None:
        super().init_device()
        if (
            self.model_runner.use_async_scheduling
            and os.environ.get("HB_P2_HOST_LAUNCH_THREADS") == "1"
        ):
            raise ValueError(
                "decode async scheduling cannot use experimental host launch threads"
            )
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self.prefill_stream = torch.cuda.Stream(device=self.device)
        self.host_launch_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="SlackServeHostLaunch"
        )
        if os.environ.get("HB_P2_HOST_LAUNCH_THREADS") == "1":
            sys.setswitchinterval(
                float(os.environ.get("HB_P2_GIL_SWITCH_INTERVAL", "0.001"))
            )

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
        if (
            os.environ.get("HB_P2_HOST_LAUNCH_THREADS") == "1"
            and lane in ("decode", "prefill")
        ):
            from vllm.v1.worker.gpu.slackserve_model_runner import PreparedLaneRun

            # Shared scheduler/request/KV preparation stays on the controller
            # thread. Only the expensive model enqueue and sampling move to a
            # lane launcher after every required tensor has been packaged.
            with torch.cuda.stream(self._lane_stream(lane)):
                prepared = self.model_runner.prepare_model(scheduler_output)
            if not isinstance(prepared, PreparedLaneRun):
                return prepared
            return self.host_launch_pool.submit(
                self._launch_and_collect, prepared
            )

        with torch.cuda.stream(self._lane_stream(lane)):
            output = super().execute_model(scheduler_output)
            if (
                output is None
                and scheduler_output.total_num_scheduled_tokens > 0
            ):
                if (
                    self.model_runner.use_async_scheduling
                    and lane in ("decode", "default")
                ):
                    # EngineCore.step_with_batch_queue owns decode sampling.
                    # Keeping that stock split is what permits scheduler
                    # placeholders and the next decode replay to be queued
                    # before the prior output reaches the CPU.
                    return None
                # Enqueue sampling while this ticket's lane-local inputs
                # and request slots are still current. The returned
                # AsyncOutput is waited by UniProcExecutor off-thread, so
                # dispatch remains nonblocking and another lane can launch.
                return self.model_runner.sample_tokens(
                    None, lane=lane, return_async=True
                )
            return output

    @torch.inference_mode()
    def _launch_and_collect(self, prepared):
        lane = prepared.lane
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._lane_stream(lane)),
        ):
            output = self.model_runner.launch_prepared(prepared)
            if output is None:
                output = self.model_runner.sample_tokens(
                    None, lane=lane, return_async=True
                )
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
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

    def shutdown(self) -> None:
        if pool := getattr(self, "host_launch_pool", None):
            pool.shutdown(wait=True)
        super().shutdown()
