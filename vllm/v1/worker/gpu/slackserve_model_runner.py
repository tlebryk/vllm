import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.attention.backends.fa_utils import flash_attn_scheduler_sm_margin
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

# add imports
from vllm.v1.worker.gpu.block_table import (
    BlockTables,
    _compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import (
    AsyncOutput,
    ExecuteModelState,
    GPUModelRunner,
    pp_broadcast,
    pp_receive,
)


class SlackServeModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        unsupported = (
            self.lora_config is not None
            or self.speculative_config is not None
            or self.supports_mm_inputs
            or self.use_pp
            or self.parallel_config.tensor_parallel_size != 1
            or self.dp_size != 1
            or self.use_dcp
            or self.use_async_scheduling
        )
        if unsupported:
            raise ValueError(
                "SlackServeModelRunner requires TP=1 text generation without "
                "LoRA, speculative decoding, MM, PP/DP/DCP, or async scheduling"
            )
        if (
            os.environ.get("HB_P2_DECODE_GRAPHS_ONLY") == "1"
            and self.compilation_config.mode != CompilationMode.NONE
        ):
            raise ValueError(
                "decode-only P2 graphs require compilation mode NONE; "
                "vLLM's compiled callable has shared runtime buffers"
            )
        # RequestState, sampler, and the logical BlockTables remain the base
        # runner's single source of truth. Tensors overwritten while a forward
        # is prepared are per lane. Decode deliberately owns the base runner's
        # buffers: CUDA graphs are captured against those exact addresses.
        # Prefill gets separate scratch and always runs eager when
        # HB_P2_DECODE_GRAPHS_ONLY=1.
        from vllm.v1.worker.slackserve_gpu_worker import prefill_lane_names

        # Per-lane FlashAttention-3 tile-scheduler SM margin (SMs left free).
        # Default 0 preserves today's behavior bit-identically.
        self.fa_sm_margin = {
            "decode": int(os.environ.get("HB_P2_FA_SM_MARGIN_DECODE", "0") or "0"),
            "prefill": int(os.environ.get("HB_P2_FA_SM_MARGIN_PREFILL", "0") or "0"),
        }

        self.contexts = {
            "decode": LaneContext(
                input_buffers=self.input_buffers,
                completion_event=torch.cuda.Event(),
                copy_stream=torch.cuda.Stream(self.device),
                copy_event=torch.cuda.Event(),
                staging_event=torch.cuda.Event(),
            ),
        }
        # One fully private context per prefill lane (HB_P2_PREFILL_LANES=2
        # adds a second). Output D2H copies must not share a stream/event
        # across lanes: two in-flight tickets would re-record one event and
        # race their copies. The same holds for input preparation buffers.
        for lane in prefill_lane_names():
            self.contexts[lane] = LaneContext(
                input_buffers=InputBuffers(
                    max_num_reqs=self.max_num_reqs,
                    max_num_tokens=self.max_num_tokens,
                    device=self.device,
                ),
                completion_event=torch.cuda.Event(),
                copy_stream=torch.cuda.Stream(self.device),
                copy_event=torch.cuda.Event(),
                staging_event=torch.cuda.Event(),
            )

    def initialize_kv_cache(self, kv_cache_config) -> None:
        super().initialize_kv_cache(kv_cache_config)
        # Decode uses BlockTables' persistent forward buffers, whose addresses
        # are baked into captured CUDA graphs. Only prefill lanes need
        # independent attention scratch.
        for lane, context in self.contexts.items():
            if lane != "decode":
                context.attn_scratch = LaneAttentionScratch(self.block_tables)

    @contextmanager
    def _use_input_buffers(self, context: "LaneContext"):
        """Bind only lane-local preparation scratch for one synchronous setup."""
        previous = self.input_buffers
        self.input_buffers = context.input_buffers
        try:
            yield
        finally:
            self.input_buffers = previous

    @contextmanager
    def _use_eager_attention_metadata(self, lane: str):
        """Keep eager prefill from overwriting decode graph metadata.

        FA3's metadata builder owns one persistent scheduler-metadata tensor
        whenever FULL graphs are enabled. A normal eager build also writes
        that tensor, so an overlapping prefill can corrupt a decode graph
        already reading it. Temporarily disabling the builder's full-graph
        path makes prefill use its own ordinary metadata allocation.
        """
        if not (
            lane.startswith("prefill")
            and os.environ.get("HB_P2_DECODE_GRAPHS_ONLY") == "1"
        ):
            yield
            return

        builders = []
        seen: set[int] = set()
        for group in self.attn_groups:
            for attention_group in group:
                builder = attention_group.get_metadata_builder(0)
                if id(builder) in seen or not hasattr(
                    builder, "use_full_cuda_graph"
                ):
                    continue
                seen.add(id(builder))
                builders.append((builder, builder.use_full_cuda_graph))
                builder.use_full_cuda_graph = False
        try:
            yield
        finally:
            for builder, previous in builders:
                builder.use_full_cuda_graph = previous

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        lane = scheduler_output.execution_lane
        # ``default`` remains a compatibility alias for decode.
        if lane == "default":
            lane = "decode"
        context = self.contexts[lane]
        if not dummy_run:
            # The staged-write scatter kernels read shared, persistent UVA
            # staging buffers (req_states/block_tables/sampler). Before this
            # lane's CPU code overwrites that staging memory, every other
            # lane's already-enqueued scatter kernels must have consumed it.
            for other_lane, other in self.contexts.items():
                if other_lane != lane and other.staging_event is not None:
                    other.staging_event.synchronize()
            # Update the request states.
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            assert context.staging_event is not None
            context.staging_event.record(torch.cuda.current_stream(self.device))
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        batch_desc = self.cudagraph_manager.dispatch(
            num_reqs, num_toks, uniform_tok_count
        )
        num_tokens_across_dp = None

        skip_compiled = False
        if (
            lane.startswith("prefill")
            and os.environ.get("HB_P2_DECODE_GRAPHS_ONLY") == "1"
        ):
            # A captured graph owns fixed input/attention/output addresses.
            # Replaying that one graph concurrently from both lanes races
            # those buffers and causes illegal memory accesses. Keep the
            # high-frequency decode lane graphed. P2 uses compilation mode
            # NONE because vLLM's compiled callable owns reusable activation
            # buffers that are not safe to share across the two streams.
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_toks,
                num_reqs=num_reqs,
            )
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            # Override batch_desc to NONE.
            skip_compiled = True
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_toks,
                num_reqs=num_reqs,
            )

        if self.dp_size > 1:
            batch_desc, num_tokens_across_dp = sync_cudagraph_and_dp_padding(
                self.cudagraph_manager,
                batch_desc,
                num_toks,
                num_reqs,
                uniform_tok_count,
                self.dp_size,
                self.dp_rank,
            )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            with self._use_input_buffers(context):
                input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self._prepare_attn(context, input_batch)

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            with self._use_input_buffers(context):
                input_batch = InputBatch.make_dummy(
                    batch_desc.num_reqs or num_reqs,
                    batch_desc.num_tokens,
                    self.input_buffers,
                )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                block_tables = None
                slot_mappings = None
            # FIXME(woosuk): Fix warmup for LoRA.

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            margin = self.fa_sm_margin["decode" if lane == "decode" else "prefill"]
            with (
                flash_attn_scheduler_sm_margin(margin),
                self._use_eager_attention_metadata(lane),
            ):
                attn_metadata = self.model_state.prepare_attn(
                    input_batch,
                    batch_desc.cg_mode,
                    block_tables,
                    slot_mappings,
                    self.attn_groups,
                    self.kv_cache_config,
                )

        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # Run MM encoder (if needed) and get multimodal embeddings.
            # Only first PP rank prepares multimodal embeddings.
            # NOTE(woosuk): We must call get_mm_embeddings even during dummy runs
            # to obtain inputs_embeds, because the compiled model expects this input.
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
                self.req_states,
            )

        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None
            assert intermediate_tensors is not None

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.model(**model_inputs)
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    aux_hidden_states = None

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)

        # Keep post-forward state per lane. The inherited profiling warmup
        # reads ``self.execute_model_state`` directly, so mirror only dummy
        # work there for compatibility; real lane work must not share it.
        execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            kv_connector_output=kv_connector_output,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        self.contexts[lane].execute_model_state = execute_model_state
        assert context.completion_event is not None
        context.main_stream = torch.cuda.current_stream(self.device)
        context.completion_event.record(context.main_stream)
        if dummy_run:
            self.execute_model_state = execute_model_state

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            assert isinstance(hidden_states, IntermediateTensors)
            hidden_states.kv_connector_output = kv_connector_output
            return hidden_states
        # Last rank (or no PP): hidden_states is a tensor for sampling.
        assert isinstance(hidden_states, torch.Tensor)
        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
        lane: str = "decode",
        return_async: bool = False,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        # ``default`` remains a compatibility alias for decode.
        if lane == "default":
            lane = "decode"
        context = self.contexts[lane]
        # AsyncOutput's stream() helper restores ``main_stream`` as the
        # ambient stream on exit, and postprocess launches on whatever stream
        # is then current. Sampling must therefore run on the same stream the
        # lane's forward used; the Slack Serve worker guarantees this.
        assert context.main_stream is None or (
            torch.cuda.current_stream(self.device) == context.main_stream
        ), "sample_tokens must run on the lane's stream"
        if context.completion_event is not None:
            torch.cuda.current_stream(self.device).wait_event(context.completion_event)
        execute_model_state = context.execute_model_state
        if execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = execute_model_state.input_batch
        attn_metadata = execute_model_state.attn_metadata
        slot_mappings_by_layer = execute_model_state.slot_mappings_by_layer
        hidden_states = execute_model_state.hidden_states
        aux_hidden_states = execute_model_state.aux_hidden_states
        kv_connector_output = execute_model_state.kv_connector_output
        num_tokens_across_dp = execute_model_state.num_tokens_across_dp
        # reset execute model state
        context.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs, max_sample_len=self.num_speculative_steps + 1
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # Last rank: sample tokens
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.use_pp:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=context.main_stream or self.main_stream,
            copy_stream=context.copy_stream or self.output_copy_stream,
            copy_event=context.copy_event or self.output_copy_event,
        )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )
        if self.speculator is not None:
            assert self.sampler is not None
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.req_states.draft_logits,
                num_tokens_across_dp=num_tokens_across_dp,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        # Publish lane readiness only after sampling and postprocess have been
        # enqueued. execute_model recorded this same event after the forward;
        # re-recording it here makes controller queries cover the full ticket.
        assert context.completion_event is not None
        context.completion_event.record(torch.cuda.current_stream(self.device))

        if return_async or self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def _prepare_attn(
        self, state: "LaneContext", input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if state is self.contexts["decode"]:
            # These persistent buffers are the ones captured by CUDA graphs.
            return self.prepare_attn(input_batch)
        assert state.attn_scratch is not None
        return state.attn_scratch.prepare(input_batch)


@dataclass
class LaneContext:
    input_buffers: InputBuffers
    completion_event: torch.cuda.Event | None = None
    main_stream: torch.cuda.Stream | None = None
    attn_scratch: "LaneAttentionScratch | None" = None
    execute_model_state: ExecuteModelState | None = None
    copy_stream: torch.cuda.Stream | None = None
    copy_event: torch.cuda.Event | None = None
    staging_event: torch.cuda.Event | None = None


class LaneAttentionScratch:
    """Per-lane forward outputs backed by one canonical BlockTables store."""

    def __init__(self, logical: BlockTables):
        self.logical = logical
        self.input_block_tables = [
            torch.zeros_like(table.gpu) for table in logical.block_tables
        ]
        self.input_block_table_ptrs = logical._make_ptr_tensor(self.input_block_tables)
        self.slot_mappings = torch.zeros(
            logical.num_kv_cache_groups,
            logical.max_num_batched_tokens,
            dtype=torch.int64,
            device=logical.device,
        )

    def prepare(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        logical = self.logical
        num_reqs = input_batch.idx_mapping.shape[0]
        _gather_block_tables_kernel[
            (logical.num_kv_cache_groups, input_batch.num_reqs_after_padding)
        ](
            input_batch.idx_mapping,
            logical.block_table_ptrs,
            self.input_block_table_ptrs,
            logical.block_table_strides,
            logical.num_blocks.gpu,
            logical.num_blocks.gpu.stride(0),
            num_reqs,
            self.input_block_tables[0].shape[1],
            BLOCK_SIZE=1024,
        )
        _compute_slot_mappings_kernel[(logical.num_kv_cache_groups, num_reqs + 1)](
            logical.max_num_batched_tokens,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            logical.block_table_ptrs,
            logical.block_table_strides,
            logical.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            logical.cp_rank,
            CP_SIZE=logical.cp_size,
            CP_INTERLEAVE=logical.cp_interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=1024,
        )
        return (
            tuple(
                table[: input_batch.num_reqs_after_padding]
                for table in self.input_block_tables
            ),
            self.slot_mappings[:, : input_batch.num_tokens_after_padding],
        )
