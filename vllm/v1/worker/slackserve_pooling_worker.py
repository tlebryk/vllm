# SPDX-License-Identifier: Apache-2.0
"""Existing pooling math with native controller-owned streams and tickets."""

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


class SlackServePoolingRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        if not self.is_pooling_model:
            raise ValueError("pooling adapter requires a pooling model")
        # Output transport is asynchronous independently of scheduling. There
        # is only one dense ticket, so scheduler placeholder logic is unneeded.
        self.async_pooling_output = True
        self.async_output_copy_stream = torch.cuda.Stream(device=device)
        # Every pooling forward is eager with respect to CUDA graphs. The
        # ordinary compiled callable and mature model.pooler are retained.
        self.decode_graphs_only = True


class SlackServePoolingWorker(Worker):
    def init_device(self):
        # Select the mature runner per worker; never change process-wide envs
        # while a generating runner is resident.
        self.use_v2_model_runner = False
        super().init_device()
        self.prefill_stream = torch.cuda.current_stream(self.device)

    def _create_model_runner(self):
        return SlackServePoolingRunner(self.vllm_config, self.device)

    def bind_lane_streams(self, prefill_stream, decode_stream, *, serial=False):
        if prefill_stream.device != self.device or decode_stream.device != self.device:
            raise ValueError("shared lane streams must use the worker's device")
        same = prefill_stream.cuda_stream == decode_stream.cuda_stream
        if same != serial:
            raise ValueError(
                "serial binding requires one stream; concurrent requires two"
            )
        if self.model_runner.execute_model_state is not None:
            raise RuntimeError("cannot rebind a worker with a pending forward")
        prefill_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.prefill_stream = prefill_stream

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        if getattr(scheduler_output, "execution_lane", "default") == "decode":
            raise ValueError("pooling endpoints have no decode lane")
        with torch.cuda.stream(self.prefill_stream):
            return super().execute_model(scheduler_output)
