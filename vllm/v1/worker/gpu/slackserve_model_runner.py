# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import vllm.forward_context as forward_context
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.fa_utils import flash_attn_scheduler_sm_margin
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
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
from vllm.v1.worker.gpu.pp_utils import pp_finish_ticket
from vllm.v1.worker.slackserve_config import SlackServeConfig
from vllm.v1.worker.workspace import WorkspaceManager, use_workspace_manager


class SlackServeModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.slackserve_config = SlackServeConfig.from_vllm_config(vllm_config)
        # Expert scratch may be reused sequentially within a lane, never
        # between an eager prefill and a concurrently replaying decode graph.
        self.lane_workspaces: dict[str, WorkspaceManager] = {}
        unsupported = (
            self.lora_config is not None
            or self.speculative_config is not None
            or self.supports_mm_inputs
            or self.dp_size != 1
            or self.use_dcp
            or self.use_async_scheduling
        )
        if unsupported:
            raise ValueError(
                "SlackServeModelRunner requires text generation without "
                "LoRA, speculative decoding, MM, DP/DCP, or async scheduling"
            )
        # HB_P2_COMPILE_DECODE=1 opts the DECODE lane into vLLM's inductor
        # (VLLM_COMPILE) compiled callable while the prefill lane(s) stay on the
        # ORIGINAL eager forward. Safe by construction: the decode lane only ever
        # *replays* a FULL cudagraph captured over the compiled forward (its
        # inductor-planned intermediate buffers live at fixed, decode-private
        # addresses), and the prefill lane is forced onto self.forward via
        # skip_compiled=True (see execute_model) so it never invokes the compiled
        # callable and never touches those buffers. Without this flag the two
        # lanes sharing one compiled callable would race its reused runtime
        # buffers, hence the guard below.
        # HB_P2_COMPILE_PREFILL=1 is the MIRROR IMAGE: the PREFILL lane invokes
        # the compiled callable live while the DECODE lane runs the ORIGINAL
        # eager forward everywhere -- including during decode's FULL cudagraph
        # capture, so decode graphs are captured over eager exactly as in mode 0
        # and their captured buffers have nothing to do with inductor's runtime
        # buffers. Same safety argument, mirrored: exactly one live user of the
        # compiled callable (prefill), no cross-lane buffer sharing.
        self._compile_decode = self.slackserve_config.compile_decode
        self._compile_prefill = self.slackserve_config.compile_prefill
        self.decode_graphs_only = self.slackserve_config.decode_graphs_only
        if self._compile_decode and self._compile_prefill:
            raise ValueError(
                "HB_P2_COMPILE_DECODE and HB_P2_COMPILE_PREFILL are mutually "
                "exclusive: only one lane may own the single compiled callable."
            )
        self._compile_any = self._compile_decode or self._compile_prefill
        if (
            self.decode_graphs_only
            and self.compilation_config.mode != CompilationMode.NONE
            and not self._compile_any
        ):
            raise ValueError(
                "decode-only P2 graphs require compilation mode NONE; "
                "vLLM's compiled callable has shared runtime buffers. "
                "Set HB_P2_COMPILE_DECODE=1 (compile decode, prefill eager) or "
                "HB_P2_COMPILE_PREFILL=1 (compile prefill, decode eager) to give "
                "the compiled callable a single lane."
            )
        if self._compile_any:
            which = (
                "HB_P2_COMPILE_DECODE"
                if self._compile_decode
                else ("HB_P2_COMPILE_PREFILL")
            )
            if not self.decode_graphs_only:
                raise ValueError(
                    f"{which}=1 requires HB_P2_DECODE_GRAPHS_ONLY=1: the "
                    "non-compiled lane must be pinned to eager so it cannot "
                    "enter the compiled callable."
                )
            if self.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                raise ValueError(
                    f"{which}=1 requires compilation mode 3 (VLLM_COMPILE); "
                    f"got mode={self.compilation_config.mode}."
                )
        # RequestState, sampler, and the logical BlockTables remain the base
        # runner's single source of truth. Tensors overwritten while a forward
        # is prepared are per lane. Decode deliberately owns the base runner's
        # buffers: CUDA graphs are captured against those exact addresses.
        # Prefill gets separate scratch and always runs eager when
        # HB_P2_DECODE_GRAPHS_ONLY=1.
        # Per-lane FlashAttention-3 tile-scheduler SM margin (SMs left free).
        # Default 0 preserves today's behavior bit-identically.
        self.fa_sm_margin = {
            "decode": self.slackserve_config.fa_sm_margin_decode,
            "prefill": self.slackserve_config.fa_sm_margin_prefill,
        }
        # HB_P2_PREP_STREAM=1: enqueue ticket-head staging scatters and input
        # preparation on a dedicated PER-LANE prep stream, and make the
        # lane's execution stream wait on a prep-done event. Prep kernels
        # then never queue behind a busy execution stream: the other lanes'
        # staging events complete promptly (no ticket-head serialization)
        # and a second same-stream ticket's prep cannot stall decode
        # dispatch. The prep stream must be per lane: prep-time allocations
        # (attention metadata) are freed into the allocating stream's pool
        # as soon as host refs drop, and a shared prep stream would let one
        # lane's next prep reuse memory another lane's forward is still
        # reading. One in-flight ticket per lane makes same-lane reuse safe.
        self.use_prep_streams = self.slackserve_config.prep_streams
        self.lane_staging = self.slackserve_config.lane_staging
        self.uva_fence_registry.isolate_lanes = self.lane_staging
        self.prep_streams: dict[str, torch.cuda.Stream] = {}
        # The controller may keep multiple decode tickets
        # in flight (see EngineCore.step_two_lane). All host prep is enqueued
        # from the engine thread in dispatch order, so lane-context reuse
        # stays serialized; the only per-ticket resource is the output
        # copy_event (fresh per decode sample below).
        async_decode_depth = self.slackserve_config.async_decode_depth
        if not 1 <= async_decode_depth <= 4:
            raise ValueError(
                "HB_P2_ASYNC_DECODE_DEPTH must be between 1 and 4; "
                f"got {async_decode_depth}"
            )
        self._hb_async_decode = async_decode_depth > 1

        self.contexts = {
            "decode": LaneContext(
                input_buffers=self.input_buffers,
                completion_event=torch.cuda.Event(),
                copy_stream=torch.cuda.Stream(self.device),
                copy_event=torch.cuda.Event(),
                staging_event=torch.cuda.Event(),
                prep_event=torch.cuda.Event(),
            ),
        }
        # One fully private context per prefill lane (HB_P2_PREFILL_LANES=2
        # adds a second). Output D2H copies must not share a stream/event
        # across lanes: two in-flight tickets would re-record one event and
        # race their copies. The same holds for input preparation buffers.
        for lane in self.slackserve_config.prefill_lane_names():
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
                prep_event=torch.cuda.Event(),
            )

    def initialize_kv_cache(self, kv_cache_config) -> None:
        super().initialize_kv_cache(kv_cache_config)
        # Decode uses BlockTables' persistent forward buffers, whose addresses
        # are baked into captured CUDA graphs. Only prefill lanes need
        # independent attention scratch.
        for lane, context in self.contexts.items():
            if lane != "decode":
                context.attn_scratch = LaneAttentionScratch(self.block_tables)
        self.shared_kv_mapping = None
        if self.shared_kv is not None:
            from vllm.v1.worker.gpu.shared_kv import SharedKVMapping

            shared, model_id = self.shared_kv
            self.shared_kv_mapping = SharedKVMapping(self, shared, model_id)

    def _apply_prefill_custom_ops(self) -> None:
        """HB_P2_PREFILL_CUSTOM_OPS: per-lane custom-op dispatch.

        Run the server with custom_ops ["none"] so the decode lane's FULL
        cudagraphs are captured over the inductor-fused (stock-kernel) forward,
        then -- after capture, when the decode lane only ever *replays* graphs
        and never re-enters Python module forwards -- rebind the listed
        CustomOps to forward_cuda. The only remaining caller of module
        forwards is the eager prefill lane (skip_compiled=True), which needs
        the hand-fused CUDA kernels to avoid the unfused-native launch storm.
        Measured motivation: the shared custom_ops set taxes uncontended
        decode steps +6-9% at batch<=16 (the lambda=0.87 loss), while
        custom_ops none globally blows up prefill (TTFT +192ms, decode stalls
        p99 150-170ms during prefill chunks).
        """
        spec = self.slackserve_config.prefill_custom_ops
        if not spec:
            return
        names = {s.strip().lstrip("+") for s in spec.split(",") if s.strip()}
        from vllm.model_executor.custom_op import CustomOp

        flipped = 0
        for mod in self.model.modules():
            if isinstance(mod, CustomOp) and getattr(mod, "name", None) in names:
                mod._forward_method = mod.forward_cuda
                flipped += 1
        if flipped == 0:
            raise RuntimeError(
                f"HB_P2_PREFILL_CUSTOM_OPS matched no modules: {sorted(names)}"
            )
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "HB_P2_PREFILL_CUSTOM_OPS: rebound %d modules to forward_cuda "
            "for the eager prefill lane (%s); decode graphs keep the "
            "captured inductor-fused kernels",
            flipped,
            sorted(names),
        )

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int, *args, **kwargs):
        workspace_lane = kwargs.pop("workspace_lane", None)
        if (
            workspace_lane is None
            and num_tokens > self.max_num_reqs
            and not kwargs.get("uniform_decode", False)
        ):
            budget = self.slackserve_config.prefill_step_tokens
            if budget:
                num_tokens = min(num_tokens, budget)
        lane = workspace_lane or (
            "prefill" if num_tokens > self.max_num_reqs else "decode"
        )
        previous_lane = getattr(self, "_dummy_workspace_lane", None)
        self._dummy_workspace_lane = workspace_lane
        try:
            return super()._dummy_run(num_tokens, *args, **kwargs)
        finally:
            # The base profiling helper clears its mirrored state, but
            # SlackServe's lane-local mirror must also cease being pending.
            self.contexts[lane].execute_model_state = None
            self.execute_model_state = None
            self._dummy_workspace_lane = previous_lane

    @torch.inference_mode()
    def profile_run(self) -> None:
        # The controller enforces this work unit; engine capacity remains
        # unchanged. Profile the largest executable prefill, not a forbidden
        # larger batch that would unnecessarily consume the KV budget.
        self._profiling_prefill = True
        try:
            super().profile_run()
        finally:
            self._profiling_prefill = False
        if self.model_config.is_moe:
            # The main profile exercises the largest prefill work unit.
            # Reserve decode scratch as well before calculating KV headroom.
            # vLLM's post-KV warmup uses the default/decode lane for a
            # two-token prompt per request. Exercise that maximum shape
            # before KV sizing and locking graph-owned scratch pointers.
            # Runtime decode remains one token per request; this is only
            # an initialization reservation, not a scheduling limit.
            self._dummy_run(
                2 * self.max_num_reqs, skip_attn=True, workspace_lane="decode"
            )
            # Merged batches (stock hybrid steps) also run on the decode lane
            # and carry up to the prefill work unit of prompt tokens; their
            # expert workspace must exist before the lock (a 16K-token merged
            # probe needed 1.4 GB against a 352 MB decode workspace).
            merged_unit = min(
                self.slackserve_config.prefill_step_tokens or self.max_num_tokens,
                self.max_num_tokens,
            )
            if merged_unit > 2 * self.max_num_reqs:
                self._dummy_run(merged_unit, skip_attn=True, workspace_lane="decode")
            torch.accelerator.synchronize()
            for workspace in self.lane_workspaces.values():
                workspace.lock()

    def capture_model(self) -> int:
        workspace = self.lane_workspaces.setdefault(
            "decode", WorkspaceManager(self.device)
        )
        # Graph capture invokes the model directly, bypassing execute_model.
        # Freeze pointers to the same runner-owned decode scratch it profiled.
        with use_workspace_manager(workspace):
            if not self._compile_prefill:
                n = super().capture_model()
                self._apply_prefill_custom_ops()
                return n
            # HB_P2_COMPILE_PREFILL=1: decode graphs must be captured over the
            # ORIGINAL eager forward (decode never uses the compiled callable).
            # support_torch_compile's __call__ returns self.forward whenever
            # do_not_compile is True (decorators.py: `if self.do_not_compile ...`),
            # so toggling it forces the capture warmup + capture passes eager. The
            # decorator is applied to the compiled BACKBONE submodule (e.g.
            # Qwen3Model), not the top-level model, so toggle every submodule that
            # carries the attribute. The prefill lane keeps its compiled callable
            # for runtime; it compiles lazily on the first real prefill forward
            # (absorbed by warmup).
            toggled: list = []
            for mod in self.model.modules():
                if "do_not_compile" in vars(mod):
                    toggled.append((mod, mod.do_not_compile))
                    mod.do_not_compile = True
            if not toggled:
                raise RuntimeError(
                    "HB_P2_COMPILE_PREFILL: no submodule exposes do_not_compile; "
                    "expected a support_torch_compile wrapper in mode 3."
                )
            try:
                n = super().capture_model()
            finally:
                for mod, prev in toggled:
                    mod.do_not_compile = prev
            self._apply_prefill_custom_ops()
            return n

    def capture_decode_mask(self, ctrl, disable_mask: int) -> int:
        """Capture a second, model-owned graph bank with a fixed TPC mask.

        Both banks use this runner's decode inputs and output destination, but
        independent graph allocation pools. Runtime never replays them concurrently
        because this model has one FIFO decode stream.
        """
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        if self._compile_any or not self.decode_graphs_only:
            raise ValueError(
                "decode mask variants require eager capture and decode-only graphs"
            )
        original = self.cudagraph_manager
        if not original._graphs_captured or not original.graphs:
            raise ValueError("decode masking requires captured FULL graphs")
        masked = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            original.cudagraph_mode,
            original.decode_query_len,
        )
        masked.pool = torch.cuda.graph_pool_handle()
        masked.hidden_states = original.hidden_states
        masked.aux_hidden_states = original.aux_hidden_states
        masked.capture_mask_ctrl = ctrl
        masked.capture_disable_mask = disable_mask
        self.cudagraph_manager = masked
        try:
            workspace = self.lane_workspaces.setdefault(
                "decode", WorkspaceManager(self.device)
            )
            with use_workspace_manager(workspace):
                size = super().capture_model()
        finally:
            self.cudagraph_manager = original
        self._decode_mask_banks = {False: original, True: masked}
        return size

    def select_decode_mask(self, masked: bool) -> None:
        self.cudagraph_manager = self._decode_mask_banks[masked]

    def _end_prep(
        self,
        context: "LaneContext",
        exec_stream: torch.cuda.Stream | None,
        *,
        fence: bool = True,
    ) -> None:
        """Return to the lane's execution stream, ordered after this prep."""
        # The legacy staging guard ends at prep. Lane-owned snapshots may
        # also feed sampling, so productive tickets defer their fence to tail.
        if fence:
            self.uva_fence_registry.fence(torch.cuda.current_stream(self.device))
        if exec_stream is None:
            return
        assert context.prep_event is not None
        context.prep_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.set_stream(exec_stream)
        exec_stream.wait_event(context.prep_event)

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
        if not (lane.startswith("prefill") and self.decode_graphs_only):
            yield
            return

        changed = []
        seen: set[int] = set()
        for group in self.attn_groups:
            for attention_group in group:
                builder = attention_group.get_metadata_builder(0)
                if id(builder) in seen:
                    continue
                seen.add(id(builder))
                if hasattr(builder, "use_full_cuda_graph"):
                    changed.append(
                        (builder, "use_full_cuda_graph", builder.use_full_cuda_graph)
                    )
                    builder.use_full_cuda_graph = False
                # Triton's small-query attention writes persistent softmax
                # intermediates. Even a final one-token prefill must never
                # write the scratch captured by an in-flight decode graph.
                fields = (
                    "softmax_segm_output",
                    "softmax_segm_max",
                    "softmax_segm_expsum",
                )
                if all(hasattr(builder, name) for name in fields):
                    banks = getattr(builder, "_slackserve_lane_scratch", None)
                    if banks is None:
                        banks = {}
                        builder._slackserve_lane_scratch = banks
                    scratch = banks.get(lane)
                    if scratch is None:
                        scratch = tuple(
                            torch.empty_like(getattr(builder, name)) for name in fields
                        )
                        banks[lane] = scratch
                    for name, private in zip(fields, scratch):
                        changed.append((builder, name, getattr(builder, name)))
                        setattr(builder, name, private)
        try:
            yield
        finally:
            for builder, name, original in reversed(changed):
                setattr(builder, name, original)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        from vllm.v1.worker.embed_sm_linear_hook import model_sm_phase

        lane = scheduler_output.execution_lane
        if lane == "default":
            lane = "decode"
            if dummy_run:
                lane = getattr(self, "_dummy_workspace_lane", None) or (
                    "prefill"
                    if scheduler_output.total_num_scheduled_tokens > self.max_num_reqs
                    else "decode"
                )
                scheduler_output.execution_lane = lane
        workspace = self.lane_workspaces.setdefault(lane, WorkspaceManager(self.device))
        try:
            with (
                use_workspace_manager(workspace),
                model_sm_phase(self.model, lane.startswith("prefill")),
            ):
                return self._execute_model(
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                )
        except Exception as error:
            shared_kv = getattr(self, "shared_kv", None)
            if shared_kv is not None:
                shared_kv[0].poison(error)
            raise

    def _execute_model(
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
        # With prep streams enabled, everything from the staged-write
        # scatters through input/attention preparation is enqueued on this
        # lane's prep stream; ``_end_prep`` restores the lane's execution
        # stream and orders it after a prep-done event before the forward.
        prep_exec_stream = None
        if self.use_prep_streams and not dummy_run:
            prep_stream = self.prep_streams.get(lane)
            if prep_stream is None:
                prep_stream = torch.cuda.Stream(self.device)
                self.prep_streams[lane] = prep_stream
            prep_exec_stream = torch.cuda.current_stream(self.device)
            torch.cuda.set_stream(prep_stream)
        if getattr(self, "shared_kv_mapping", None) is not None:
            # Lane-local translated tables/slots are captured graph inputs.
            # Prep-stream writes must follow the previous forward's reads.
            torch.cuda.current_stream(self.device).wait_event(context.completion_event)
        if not dummy_run:
            # The legacy slots are shared across lanes. Private lane banks
            # instead guard reuse of the specific buffer, including consumers
            # after prep, so unrelated decode staging need not block prefill.
            if not self.lane_staging:
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
                self._end_prep(context, prep_exec_stream)
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
        if lane.startswith("prefill") and self.decode_graphs_only:
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
            if self._compile_decode:
                # HB_P2_COMPILE_DECODE=1: the decode lane owns the inductor
                # compiled callable. Force the prefill lane onto the ORIGINAL
                # eager forward (support_torch_compile's __call__ returns
                # self.forward as soon as skip_compiled is set, BEFORE the
                # compiled branch). This is the structural guarantee that the
                # prefill lane cannot enter the compiled callable and therefore
                # cannot touch the decode graph's inductor-planned buffers. It
                # also keeps prefill on UnquantizedLinearMethod.apply so the
                # cuBLASLt SM-cap hook still fires for prefill GEMMs.
                skip_compiled = True
        if self._compile_prefill and lane == "decode":
            # HB_P2_COMPILE_PREFILL=1 (mirror image): only the prefill lane owns
            # the compiled callable. The decode lane runs the ORIGINAL eager
            # forward everywhere -- its FULL graphs were captured over eager
            # (see capture_model), so replay touches no inductor buffers. This
            # skip_compiled=True covers the decode eager-fallback (batch >
            # capture size) and dummy runs; decode FULL-graph replay does not
            # consult skip_compiled. Prefill keeps skip_compiled=False and thus
            # enters the compiled callable.
            skip_compiled = True
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
            self._end_prep(context, prep_exec_stream)
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
            if getattr(self, "shared_kv_mapping", None) is not None:
                attn_metadata, slot_mappings_by_layer = self.shared_kv_mapping.prepare(
                    lane, attn_metadata, slot_mappings_by_layer
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

        self._end_prep(
            context, prep_exec_stream, fence=not self.lane_staging or dummy_run
        )

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
            model_inputs["intermediate_tensors"] = (
                self.cudagraph_manager.prepare_pp_inputs(
                    batch_desc, intermediate_tensors
                )
            )
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(
                batch_desc, intermediate_tensors
            )
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

            if self._compile_decode and not skip_compiled:
                # Reaching the compiled callable (skip_compiled False) is only
                # legal for the decode lane. Any prefill lane must have taken
                # the skip_compiled=True branch above; assert it here so a
                # regression that let prefill share the compiled buffers fails
                # loudly instead of racing them.
                assert lane == "decode", (
                    f"HB_P2_COMPILE_DECODE: lane={lane!r} would enter the "
                    "compiled callable; only decode may."
                )
            if self._compile_prefill and not skip_compiled:
                # Mirror image: only prefill lanes may enter the compiled
                # callable. The decode lane must have taken skip_compiled=True
                # above (and its FULL graphs run via replay, never here).
                assert lane.startswith("prefill"), (
                    f"HB_P2_COMPILE_PREFILL: lane={lane!r} would enter the "
                    "compiled callable; only prefill may."
                )
            with forward_context.set_forward_context(
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
        if not dummy_run and self.shared_kv is not None:
            # A fresh event is immutable after publication to the allocator;
            # the ordinary per-lane event is re-recorded during sampling.
            event = torch.cuda.Event()
            event.record(context.main_stream)
            shared, model_id = self.shared_kv
            shared.record_use(model_id, lane, event)
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
            if self.lane_staging:
                self.uva_fence_registry.fence(torch.cuda.current_stream(self.device))
            pp_finish_ticket()
            assert context.completion_event is not None
            context.completion_event.record(torch.cuda.current_stream(self.device))
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
        # HB_P2_ASYNC_DECODE=1: two decode tickets can be in flight, and each
        # AsyncOutput's waiter thread synchronizes its copy_event on its own
        # schedule. Re-recording one shared per-lane event from ticket N+1
        # would couple (and needlessly extend) ticket N's wait, so async
        # decode uses a fresh event per ticket instead.
        copy_event = context.copy_event or self.output_copy_event
        if lane == "decode" and self._hb_async_decode:
            copy_event = torch.cuda.Event()
        ticket_tail_event = torch.cuda.Event()
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=context.main_stream or self.main_stream,
            copy_stream=context.copy_stream or self.output_copy_stream,
            copy_event=copy_event,
            completion_event=ticket_tail_event,
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
        if self.lane_staging:
            # Sampling also consumes UVA parameter snapshots. Record after
            # every consumer, including reads of snapshots without fresh copies.
            self.uva_fence_registry.fence(torch.cuda.current_stream(self.device))
        if self.use_pp:
            pp_finish_ticket()
        assert context.completion_event is not None
        context.completion_event.record(torch.cuda.current_stream(self.device))
        ticket_tail_event.record(torch.cuda.current_stream(self.device))

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
    prep_event: torch.cuda.Event | None = None


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
