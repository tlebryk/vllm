# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Existing pooling math with native controller-owned streams and tickets."""

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


class AsyncPoolingChunkOutput(AsyncModelRunnerOutput):
    """Publish a partial pooling chunk only after its forward has completed."""

    def __init__(self, output: ModelRunnerOutput, completion_event: torch.cuda.Event):
        self.output = output
        self.completion_event = completion_event

    def get_output(self) -> ModelRunnerOutput:
        # UniProcExecutor performs this wait on an output thread, leaving
        # generation decode dispatch free to overlap this pooling forward.
        self.completion_event.synchronize()
        return self.output


class SlackServePoolingRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        if not self.is_pooling_model:
            raise ValueError("pooling adapter requires a pooling model")
        from vllm.v1.core.shared_kv import shared_kv_for_config
        from vllm.v1.worker.gpu.buffer_utils import UvaFenceRegistry

        self.shared_kv = shared_kv_for_config(vllm_config)
        self.shared_kv_mapping = None
        self._shared_forward_slots = None
        self.uva_fence_registry = UvaFenceRegistry()
        # Output transport is asynchronous independently of scheduling. There
        # is only one dense ticket, so scheduler placeholder logic is unneeded.
        self.async_pooling_output = True
        self.async_output_copy_stream = torch.cuda.Stream(device=device)
        # Every pooling forward is eager with respect to CUDA graphs. The
        # ordinary compiled callable and mature model.pooler are retained.
        self.decode_graphs_only = True

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        if self.shared_kv is None:
            return super().initialize_kv_cache_tensors(
                kv_cache_config, kernel_block_sizes
            )
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        from vllm.v1.worker.utils import bind_kv_cache

        manager, model_id = self.shared_kv
        if (
            get_kv_cache_layout() != "NHD"
            or any(
                group.backend.get_name() != "FLASH_ATTN"
                for groups in self.attn_groups
                for group in groups
            )
            or kernel_block_sizes != [manager.layouts[model_id].block_size]
        ):
            raise ValueError(
                "shared pooling KV requires NHD FlashAttention with matching block size"
            )
        caches = manager.bind_storage(model_id, kv_cache_config, self.device)
        bind_kv_cache(
            caches, self.compilation_config.static_forward_context, self.kv_caches
        )
        return caches

    def initialize_kv_cache(self, kv_cache_config):
        super().initialize_kv_cache(kv_cache_config)
        if self.shared_kv is not None:
            from vllm.v1.worker.gpu.shared_kv import SharedKVMapping

            manager, model_id = self.shared_kv
            self.shared_kv_mapping = SharedKVMapping(
                self,
                manager,
                model_id,
                lanes=("prefill",),
                max_blocks=self.input_batch.block_table[0]
                .get_device_tensor(1)
                .shape[1],
            )

    def _get_slot_mappings(self, *args, **kwargs):
        groups, layers = super()._get_slot_mappings(*args, **kwargs)
        if self.shared_kv_mapping is not None:
            if not isinstance(layers, dict):
                raise ValueError("shared pooling KV does not support microbatching")
            # The caller retains this dictionary for ForwardContext KV writes.
            # Attention metadata is built next and replaces its logical slots
            # with the same layer-specific physical slots used for KV reads.
            self._shared_forward_slots = layers
        return groups, layers

    def _build_attention_metadata(self, *args, **kwargs):
        metadata, common = super()._build_attention_metadata(*args, **kwargs)
        if self.shared_kv_mapping is not None:
            if not isinstance(metadata, dict):
                raise ValueError("shared pooling KV requires one attention batch")
            metadata, slots = self.shared_kv_mapping.prepare(
                "prefill", metadata, self._shared_forward_slots
            )
            self._shared_forward_slots.clear()
            self._shared_forward_slots.update(slots)
        return metadata, common

    @torch.inference_mode()
    def execute_model(self, scheduler_output, *args, **kwargs):
        try:
            output = super().execute_model(scheduler_output, *args, **kwargs)
            if (
                self.shared_kv is not None
                and scheduler_output.total_num_scheduled_tokens
            ):
                stream = torch.cuda.current_stream(self.device)
                self.uva_fence_registry.fence(stream)
                event = torch.cuda.Event()
                event.record(stream)
                manager, model_id = self.shared_kv
                # Immutable forward fence: scores may be published on a copy
                # stream, but no physical KV page is reused before this event.
                manager.record_use(model_id, "prefill", event)
            return output
        except Exception as error:
            if self.shared_kv is not None:
                self.shared_kv[0].poison(error)
            raise


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
            output = super().execute_model(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens > 0 and isinstance(
                output, ModelRunnerOutput
            ):
                # Unfinished pooling chunks have no scores to copy, so the
                # mature runner returns plain output before the GPU finishes.
                # Native lane tickets still need a full-forward fence before
                # scheduler publication and the next shared dense dispatch.
                completion_event = torch.cuda.Event()
                completion_event.record(self.prefill_stream)
                return AsyncPoolingChunkOutput(output, completion_event)
            return output
