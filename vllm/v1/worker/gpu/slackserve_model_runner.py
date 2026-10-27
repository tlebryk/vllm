# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native streams around the current runner, preserving hybrid-state updates.

This port initially supports single-model TP/EP with PP=DP=1. The upstream
forward/sampling path remains authoritative; only transient storage changes
between streams. Request-indexed state and model/KV storage remain shared.
"""

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.buffer_utils import UvaFenceRegistry, use_uva_fence_registry
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.slackserve_scratch import ModelScratch
from vllm.v1.worker.slackserve_config import SlackServeConfig
from vllm.v1.worker.workspace import WorkspaceManager, use_workspace_manager


@dataclass
class LaneContext:
    fields: dict[str, Any] = field(default_factory=dict)
    execute_model_state: Any = None
    completion_event: Any = None


class SlackServeModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        self.uva_fence_registry = UvaFenceRegistry()
        self.uva_fence_registry.isolate_lanes = True
        with use_uva_fence_registry(self.uva_fence_registry):
            super().__init__(vllm_config, device)
        self.slackserve_config = SlackServeConfig.from_vllm_config(vllm_config)
        cfg = self.slackserve_config
        if (
            self.use_pp
            or self.dp_size != 1
            or self.use_dcp
            or self.pcp_manager is not None
            or self.speculative_config is not None
            or self.lora_config is not None
            or self.scheduler_config.async_scheduling
            or self.is_pooling_model
            or self.model_config.enable_prompt_embeds
            or self.parallel_config.enable_eplb
            or self.parallel_config.enable_dbo
            or self.vllm_config.kv_transfer_config is not None
            or (
                isinstance(self.vllm_config.additional_config, dict)
                and self.vllm_config.additional_config.get("shared_kv") is not None
            )
        ):
            raise ValueError(
                "Modern native port requires single-model TP/EP, PP=DP=1, "
                "no speculative/LoRA/DBO/shared-KV path"
            )
        if cfg.compile_prefill or cfg.prep_streams or cfg.prefill_custom_ops:
            raise ValueError(
                "Modern native port has not validated compiled prefill, "
                "prep streams, or op rebinding"
            )
        if (
            self.compilation_config.mode != CompilationMode.NONE
            and not cfg.compile_decode
        ):
            raise ValueError("Only decode may own the compiled callable")
        self._hb_async_decode = cfg.async_decode_depth > 1
        self.decode_graphs_only = True
        self.lane_workspaces = {
            lane: WorkspaceManager(self.device)
            for lane in ["decode", *cfg.prefill_lane_names()]
        }
        self.contexts = {}
        for lane in self.lane_workspaces:
            self.contexts[lane] = LaneContext(
                fields={
                    "input_buffers": self.input_buffers
                    if lane == "decode"
                    else InputBuffers(
                        self.max_num_reqs, self.max_num_tokens, self.device
                    ),
                    "output_copy_stream": self.output_copy_stream
                    if lane == "decode"
                    else torch.cuda.Stream(self.device),
                },
                completion_event=torch.cuda.Event(),
            )
        self._dummy_lane = "decode"
        self._active_lane = None
        self.shared_kv = None
        self.model_scratch = {}

    def load_model(self, *args, **kwargs):
        with use_uva_fence_registry(self.uva_fence_registry):
            result = super().load_model(*args, **kwargs)
        if type(self.model_state) not in (DefaultModelState, MambaHybridModelState):
            raise ValueError(
                "Unvalidated stream-local model state: "
                f"{type(self.model_state).__name__}"
            )
        if getattr(self.model_state, "recoverssm", None) is not None:
            raise ValueError("RecoverSSM stream ownership has not been validated")
        for lane in self.slackserve_config.prefill_lane_names():
            self.model_scratch[lane] = ModelScratch(self.model)
        return result

    def initialize_kv_cache(self, *args, **kwargs):
        with use_uva_fence_registry(self.uva_fence_registry):
            result = super().initialize_kv_cache(*args, **kwargs)
            decode = self.contexts["decode"]
            for key in (
                "block_tables",
                "attn_groups",
                "model_state",
                "cudagraph_manager",
            ):
                decode.fields[key] = getattr(self, key)
            for lane, context in self.contexts.items():
                if lane == "decode":
                    continue
                # Share logical per-request tables, but never the gathered
                # batch tables or slot mappings captured by decode graphs.
                tables = copy(self.block_tables)
                tables.input_block_tables = [
                    torch.empty_like(t) for t in self.block_tables.input_block_tables
                ]
                tables.input_block_table_ptrs = tables._make_ptr_tensor(
                    tables.input_block_tables
                )
                tables.slot_mappings = torch.empty_like(self.block_tables.slot_mappings)
                config = copy(self.vllm_config)
                config.compilation_config = copy(self.compilation_config)
                config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                with self.model_scratch[lane].use():
                    groups, _, block_sizes = init_attn_backend(
                        self.kv_cache_config, config, self.device
                    )
                if block_sizes != self.kernel_block_sizes:
                    raise ValueError(
                        "Prefill and decode selected incompatible KV block layouts"
                    )
                state = copy(cast(DefaultModelState, self.model_state))
                if state.rope_state is not None:
                    state.rope_state = copy(state.rope_state)
                    state.rope_state.positions = torch.empty_like(
                        state.rope_state.positions
                    )
                if state.supports_mm_inputs:
                    state.encoder_runner = copy(state.encoder_runner)
                    state.encoder_runner.inputs_embeds = torch.empty_like(
                        state.encoder_runner.inputs_embeds
                    )
                if isinstance(state, MambaHybridModelState) and state._align_mode:
                    # State indices are persistent per request and shared.
                    # Source columns/offsets and aligned batch indices are scratch.
                    state._mamba_src_col_gpu = torch.empty_like(
                        state._mamba_src_col_gpu
                    )
                    state._mamba_src_off_gpu = torch.empty_like(
                        state._mamba_src_off_gpu
                    )
                    state.initialize_kv_cache(self.kv_cache_config, tables)
                context.fields.update(
                    block_tables=tables,
                    attn_groups=groups,
                    model_state=state,
                    cudagraph_manager=None,
                )
        return result

    @contextmanager
    def _use_lane(self, lane):
        if self._active_lane is not None:
            if self._active_lane != lane:
                raise RuntimeError("Nested execution switched streams")
            yield self.contexts[lane]
            return
        context = self.contexts[lane]
        previous = {key: getattr(self, key) for key in context.fields}
        previous_stream = self.main_stream
        self._active_lane = lane
        for key, value in context.fields.items():
            setattr(self, key, value)
        self.main_stream = torch.cuda.current_stream(self.device)
        try:
            with (
                use_uva_fence_registry(self.uva_fence_registry),
                self.uva_fence_registry.use_lane(lane),
                use_workspace_manager(self.lane_workspaces[lane]),
                self.model_scratch[lane].use(self.main_stream)
                if lane in self.model_scratch
                else nullcontext(),
            ):
                yield context
        finally:
            for key, value in previous.items():
                setattr(self, key, value)
            self.main_stream = previous_stream
            self._active_lane = None

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None, **kwargs):
        dummy = kwargs.get("dummy_run", False)
        lane = self._dummy_lane if dummy else scheduler_output.execution_lane
        lane = "decode" if lane == "default" else lane
        if scheduler_output.scheduled_encoder_inputs or any(
            r.mm_features for r in scheduler_output.scheduled_new_reqs
        ):
            raise ValueError("Modern native overlap currently accepts text-only inputs")
        from vllm.v1.attention.backends.mla.slackserve_plan import use_mla_sm_budget

        indices = getattr(scheduler_output, "execution_mask", None)
        sms = 2 * len(indices) if lane != "decode" and indices and not dummy else None
        with self._use_lane(lane) as context, use_mla_sm_budget(sms):
            if not dummy and context.execute_model_state is not None:
                raise RuntimeError(
                    "Sampling must be enqueued before reusing stream inputs"
                )
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors,
                skip_compiled=lane != "decode",
                **kwargs,
            )
            if not dummy:
                context.execute_model_state = self.execute_model_state
                self.execute_model_state = None
            return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output, lane="decode", return_async=False):
        lane = "decode" if lane == "default" else lane
        with self._use_lane(lane) as context:
            self.execute_model_state = context.execute_model_state
            context.execute_model_state = None
            output = super().sample_tokens(grammar_output)
            # Upstream copies tokens before recurrent-state postprocessing.
            # Ticket retirement must also cover that tail and UVA readers.
            self.uva_fence_registry.fence(self.main_stream)
            tail = torch.cuda.Event()
            tail.record(self.main_stream)
            context.completion_event = tail
            if isinstance(output, AsyncOutput):
                output.completion_event = tail
                if not return_async:
                    return output.get_output()
            return output

    def _dummy_run(self, num_tokens, *args, **kwargs):
        lane = kwargs.pop("workspace_lane", None) or (
            "decode"
            if kwargs.get("uniform_decode", False) or num_tokens <= self.max_num_reqs
            else "prefill"
        )
        budget = self.slackserve_config.prefill_step_tokens
        if lane == "prefill" and budget:
            num_tokens = min(num_tokens, budget)
        previous = self._dummy_lane
        self._dummy_lane = lane
        try:
            with self._use_lane(lane):
                result = super()._dummy_run(num_tokens, *args, **kwargs)
                self.uva_fence_registry.fence(torch.cuda.current_stream(self.device))
                return result
        finally:
            self._dummy_lane = previous
            self.execute_model_state = None

    def profile_run(self, *args, **kwargs):
        super().profile_run(*args, **kwargs)
        # Reserve both streams' MoE workspace before sizing the shared KV pool.
        self._dummy_run(
            self.max_num_reqs,
            uniform_decode=True,
            skip_attn=True,
            is_profile=True,
            workspace_lane="decode",
        )
        torch.accelerator.synchronize()

    def profile_cudagraph_memory(self):
        try:
            return super().profile_cudagraph_memory()
        finally:
            # Upstream tears down its throwaway KV/graph state. Drop our
            # stream-local references too, before real cache allocation.
            for context in self.contexts.values():
                for key in (
                    "block_tables",
                    "attn_groups",
                    "model_state",
                    "cudagraph_manager",
                ):
                    context.fields.pop(key, None)
                context.execute_model_state = None

    def capture_model(self, **kwargs):
        with self._use_lane("decode"):
            return super().capture_model(**kwargs)
