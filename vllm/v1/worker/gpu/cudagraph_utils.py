# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    graph_capture,
    is_global_first_rank,
    get_pp_group,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = (
            self.compilation_config.get_cudagraph_pool() if cudagraph_mode else None
        )

        self._graphs_captured = False
        self.capture_stream: torch.cuda.Stream | None = None
        self._candidates: list[list[BatchExecutionDescriptor]] = []
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        self._init_candidates()

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()

        descs_by_token_count = defaultdict(list)
        descs_by_mode = defaultdict(list)

        for num_tokens in capture_sizes:
            # Capture uniform decode specfifc graphs if required
            #  (i.e. separate decode routine)
            if (
                separate_decode_routine
                and decode_mode
                and self.decode_query_len <= num_tokens <= max_decode_tokens
            ):
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_tokens // self.decode_query_len,
                    uniform_token_count=self.decode_query_len,
                )
                descs_by_mode[decode_mode].append(desc)
                descs_by_token_count[num_tokens].append(desc)

            if mixed_mode:
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed
                # so we leave it as None
                num_reqs = (
                    min(num_tokens, self.max_num_reqs)
                    if mixed_mode == CUDAGraphMode.FULL
                    else None
                )
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_count[num_tokens].append(desc)

        if not descs_by_token_count:
            return

        sorted_padded = sorted(descs_by_token_count.keys())
        self._candidates = [[] for _ in range(sorted_padded[-1] + 1)]

        current_range_start = 0
        for cg_size in sorted_padded:
            for i in range(current_range_start, cg_size + 1):
                self._candidates[i] = descs_by_token_count[cg_size]
            current_range_start = cg_size + 1

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: Callable[
            [BatchExecutionDescriptor], Callable[[CUDAGraphMode], None]
        ],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a function that runs forward with a given CUDAGraphMode.
        """
        with graph_capture(device=self.device) as capture_context:
            # Keep this manager's capture stream explicit. torch.cuda.graph's
            # implicit side stream is process-global: independent graph pools
            # captured there can still bind the same cuBLAS workspace. Those
            # graphs race that workspace when models replay on separate streams.
            self.capture_stream = capture_context.stream
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        mask_ctrl = getattr(self, "capture_mask_ctrl", None)
                        capture_stream = self.capture_stream
                        if mask_ctrl is not None:
                            mask_ctrl._set_mask(
                                capture_stream,
                                self.capture_disable_mask,
                                "decode graph capture",
                                log=False,
                            )
                        try:
                            with torch.cuda.graph(
                                graph, self.pool, stream=capture_stream
                            ):
                                forward_fn(CUDAGraphMode.NONE)
                                # Join offloader's copy stream after forward.
                                get_offloader().join_after_forward()
                        finally:
                            if mask_ctrl is not None:
                                mask_ctrl._set_mask(
                                    capture_stream, 0, "unrestricted", log=False
                                )
                        self.graphs[desc] = graph
        self._graphs_captured = True

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""
        if self._graphs_captured and 0 < num_tokens < len(self._candidates):
            for desc in self._candidates[num_tokens]:
                if _is_compatible(desc, num_reqs, num_tokens, uniform_token_count):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)
        self.pp_inputs = {}
        self.pp_outputs = {}
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
        shared_kv_mapping=None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
            )
            if shared_kv_mapping is not None:
                attn_metadata, slot_mappings = shared_kv_mapping.prepare(
                    "decode", attn_metadata, slot_mappings
                )

            pp_input = None
            if not get_pp_group().is_first_rank:
                pp_input = model.make_empty_intermediate_tensors(
                    num_tokens,
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )
                self.pp_inputs[desc] = pp_input

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = (
                    BatchDescriptor(num_tokens=num_tokens)
                    if cg_mode == CUDAGraphMode.PIECEWISE
                    else None
                )
                with set_forward_context(
                    attn_metadata if cg_mode != CUDAGraphMode.PIECEWISE else None,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                ):
                    model_inputs = {
                        "input_ids": input_buffers.input_ids[:num_tokens],
                        "positions": input_buffers.positions[:num_tokens],
                        "intermediate_tensors": pp_input,
                        **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
                    }
                    if pp_input is not None:
                        model_inputs["input_ids"] = None
                        model_inputs["inputs_embeds"] = None
                    model_output = model(**model_inputs)
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if isinstance(hidden_states, IntermediateTensors):
                        if desc not in self.pp_outputs:
                            self.pp_outputs[desc] = IntermediateTensors(
                                {
                                    k: torch.empty_like(v)
                                    for k, v in hidden_states.items()
                                }
                            )
                        for k, v in hidden_states.items():
                            self.pp_outputs[desc][k].copy_(v)
                        return
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    self.hidden_states[:num_tokens] = hidden_states
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc)

    def prepare_pp_inputs(
        self,
        desc: BatchExecutionDescriptor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        """Use the same activation addresses for capture and either replay mode."""
        if desc in self.pp_inputs:
            assert intermediate_tensors is not None
            target = self.pp_inputs[desc]
            for name, buffer in target.items():
                value = intermediate_tensors[name]  # waits for asynchronous recv
                count = value.shape[0]
                if count > buffer.shape[0]:
                    raise ValueError("PP activations exceed captured input size")
                buffer[:count].copy_(value)
                buffer[count:].zero_()
            return target
        return intermediate_tensors

    def run_fullgraph(
        self,
        desc: BatchExecutionDescriptor,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Copy received PP activations into graph-owned inputs, then replay."""
        self.prepare_pp_inputs(desc, intermediate_tensors)
        super().run_fullgraph(desc)
        if desc in self.pp_outputs:
            return self.pp_outputs[desc]
        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=True,
    )
    return attn_metadata, slot_mappings_by_layer
