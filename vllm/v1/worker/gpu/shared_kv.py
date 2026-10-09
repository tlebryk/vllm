# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent lane-private mappings for shared physical KV pages."""

from dataclasses import replace

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


class SharedKVMapping:
    def __init__(self, runner, manager, model_id):
        self.manager, self.model_id = manager, model_id
        self.layout = manager.layouts[model_id]
        self.layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        self.heads = self.layout.num_kv_heads if manager.mode == "head" else 1
        self.units = manager.units[model_id]
        self.pages = StagedWriteTensor(
            (manager.logical_capacity(model_id), self.units),
            torch.int32,
            runner.device,
            fence_registry=runner.uva_fence_registry,
        )
        manager.allocation_listeners[model_id] = self.stage_lease
        self.pending = {}
        self.read_events = {}
        self.scratch = {}
        self.device = runner.device
        self.max_pack_bytes = (
            4
            * runner.max_num_tokens
            * runner.model_config.get_num_attention_heads(runner.parallel_config)
            * self.layout.head_size
            if manager.mode == "head" and hasattr(runner, "model_config")
            else 0
        )
        self.max_blocks = runner.block_tables.input_block_tables[0].shape[1]
        for lane in runner.contexts:
            # Captured decode always uses these fixed addresses. Larger eager
            # batches get temporary scratch without replacing captured storage.
            tokens = (
                min(
                    runner.max_num_tokens,
                    runner.compilation_config.max_cudagraph_capture_size or 1,
                )
                if lane == "decode"
                else runner.max_num_tokens
            )
            reqs = min(runner.max_num_reqs, tokens)
            self.scratch[lane] = self._allocate(reqs, tokens)

    def stage_lease(self, lease):
        self.pending[lease.block_id] = lease.pages

    def _allocate(self, reqs, tokens):
        return (
            torch.zeros(
                (self.layout.num_layers, self.heads * reqs, self.max_blocks),
                dtype=torch.int32,
                device=self.device,
            ),
            torch.full((self.units, tokens), -1, dtype=torch.int64, device=self.device),
            torch.zeros(self.heads * reqs + 1, dtype=torch.int32, device=self.device),
            torch.zeros(self.heads * reqs, dtype=torch.int32, device=self.device),
        )

    def prepare(self, lane, metadata, slots):
        stream = torch.cuda.current_stream(self.device)
        # The shared mapping is consumed only during preparation, never by a
        # captured attention forward. Order writes after other lanes' lookups.
        for event in self.read_events.values():
            stream.wait_event(event)
        for block_id, pages in self.pending.items():
            self.pages.stage_write(block_id, 0, pages)
        self.pending.clear()
        self.pages.apply_write()
        names = self.layer_names
        first = metadata[names[0]]
        reqs = first.seq_lens.numel()
        tokens = slots[names[0]].numel()
        scratch = self.scratch[lane]
        if reqs * self.heads > scratch[0].shape[1] or tokens > scratch[1].shape[1]:
            scratch = self._allocate(reqs, tokens)
        tables, mapped_slots, query_start, seq_lens = scratch
        if self.manager.mode == "head":
            _expand_sequences[(triton.cdiv(self.heads * reqs + 1, 256),)](
                first.query_start_loc,
                first.seq_lens,
                query_start,
                seq_lens,
                reqs,
                self.heads,
                BLOCK=256,
            )
        result, result_slots = {}, {}
        # One uniform attention group has one logical table/slot vector. Fuse
        # translation across layers/heads rather than launching per-layer work.
        _map_tables[(self.units, triton.cdiv(reqs * self.max_blocks, 256))](
            first.block_table,
            first.block_table.stride(0),
            self.pages.gpu,
            tables,
            tables.stride(0),
            self.units,
            self.heads,
            reqs,
            self.max_blocks,
            BLOCK=256,
        )
        _map_slots[(self.units, triton.cdiv(tokens, 256))](
            slots[names[0]],
            self.pages.gpu,
            mapped_slots,
            mapped_slots.stride(0),
            self.units,
            self.layout.block_size,
            tokens,
            BLOCK=256,
        )
        for layer, name in enumerate(names):
            source = metadata[name]
            table = tables[layer, : self.heads * reqs]
            layer_slots = mapped_slots[
                layer * self.heads : (layer + 1) * self.heads, :tokens
            ]
            result_slots[name] = (
                layer_slots
                if self.heads > 1 or (self.manager.mode == "head")
                else layer_slots[0]
            )
            if self.manager.mode == "head":
                result[name] = replace(
                    source,
                    block_table=table,
                    query_start_loc=query_start[: self.heads * reqs + 1],
                    seq_lens=seq_lens[: self.heads * reqs],
                    scheduler_metadata=None,
                    slot_mapping=layer_slots,
                    shared_kv_heads=self.heads,
                    shared_kv_num_tokens=tokens,
                )
            else:
                result[name] = replace(
                    source, block_table=table, slot_mapping=layer_slots[0]
                )
        event = torch.cuda.Event()
        event.record(stream)
        self.read_events[lane] = event
        return result, result_slots

    def metadata_bytes(self):
        return {
            "logical_mapping": self.pages.gpu.numel() * self.pages.gpu.element_size(),
            "lane_scratch": {
                lane: sum(t.numel() * t.element_size() for t in tensors)
                for lane, tensors in self.scratch.items()
            },
            "head_pack_pair_max_bytes_per_layer": self.max_pack_bytes,
        }


@triton.jit
def _expand_sequences(q_src, seq_src, q_dst, seq_dst, reqs, heads, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = reqs * heads
    total = tl.load(q_src + reqs)
    h, r = i // reqs, i % reqs
    start = tl.load(q_src + r, mask=i < count, other=0)
    tl.store(q_dst + i, h * total + start, mask=i < count)
    lens = tl.load(seq_src + r, mask=i < count, other=0)
    tl.store(seq_dst + i, lens, mask=i < count)
    if tl.program_id(0) == 0:
        tl.store(q_dst + count, total * heads)


@triton.jit
def _map_tables(
    src,
    src_stride,
    pages,
    dst,
    layer_stride,
    units,
    heads,
    reqs,
    width,
    BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    layer, head = unit // heads, unit % heads
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i < reqs * width
    logical = tl.load(src + (i // width) * src_stride + i % width, mask=mask, other=0)
    page = tl.load(pages + logical * units + unit, mask=mask, other=0)
    tl.store(dst + layer * layer_stride + head * reqs * width + i, page, mask=mask)


@triton.jit
def _map_slots(
    src,
    pages,
    dst,
    dst_stride,
    units,
    block_size,
    tokens,
    BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(src + i, mask=i < tokens, other=-1)
    page = tl.load(
        pages + (slot // block_size) * units + unit,
        mask=(i < tokens) & (slot >= 0),
        other=0,
    )
    tl.store(
        dst + unit * dst_stride + i,
        tl.where(slot >= 0, page.to(tl.int64) * block_size + slot % block_size, -1),
        mask=i < tokens,
    )


def write_head_cache(key, value, kv_cache, slot_mapping):
    heads, tokens = slot_mapping.shape
    _write_head_cache[(tokens, heads)](
        key,
        value,
        kv_cache,
        kv_cache.stride(0),
        slot_mapping,
        slot_mapping.stride(0),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key.shape[-1],
        BLOCK_D=triton.next_power_of_2(key.shape[-1]),
    )


@triton.jit
def _write_head_cache(
    key,
    value,
    cache,
    plane_stride,
    slots,
    slot_stride,
    key_token_stride,
    key_head_stride,
    value_token_stride,
    value_head_stride,
    dim,
    BLOCK_D: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    # Slot tensors are head-major; grid is token-major.
    slot = tl.load(slots + head * slot_stride + token)
    d = tl.arange(0, BLOCK_D)
    mask = (d < dim) & (slot >= 0)
    k = tl.load(
        key + token * key_token_stride + head * key_head_stride + d, mask=mask, other=0
    )
    v = tl.load(
        value + token * value_token_stride + head * value_head_stride + d,
        mask=mask,
        other=0,
    )
    tl.store(cache + slot * dim + d, k, mask=mask)
    tl.store(cache + plane_stride + slot * dim + d, v, mask=mask)


def forward_head_attention(impl, query, kv_cache, metadata, output):
    """Treat each KV head as an independent sequence, preserving GQA groups.

    One FlashAttention launch per layer, with the per-head block table. GPU
    packing uses the actual query-token count from cu_seqlens, so graph bucket
    padding cannot create gaps between heads or change captured addresses.
    """
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    heads = metadata.shared_kv_heads
    tokens = metadata.shared_kv_num_tokens
    groups = impl.num_heads // heads
    dim = query.shape[-1]
    packed = torch.empty(
        (heads * tokens, groups, dim), dtype=query.dtype, device=query.device
    )
    packed_output = torch.empty_like(packed)
    _pack_query[(triton.cdiv(tokens * impl.num_heads * dim, 256),)](
        query,
        packed,
        metadata.query_start_loc,
        metadata.seq_lens.numel(),
        tokens,
        heads,
        groups,
        dim,
        query.stride(0),
        query.stride(1),
        UNPACK=False,
        BLOCK=256,
    )
    key_cache, value_cache = kv_cache.unbind(0)
    flash_attn_varlen_func(
        q=packed,
        k=key_cache,
        v=value_cache,
        out=packed_output,
        cu_seqlens_q=metadata.query_start_loc,
        max_seqlen_q=metadata.max_query_len,
        seqused_k=metadata.seq_lens,
        max_seqlen_k=metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=metadata.causal,
        block_table=metadata.block_table,
        softcap=impl.logits_soft_cap,
        fa_version=impl.vllm_flash_attn_version,
        num_splits=metadata.max_num_splits,
    )
    _pack_query[(triton.cdiv(tokens * impl.num_heads * dim, 256),)](
        output,
        packed_output,
        metadata.query_start_loc,
        metadata.seq_lens.numel(),
        tokens,
        heads,
        groups,
        dim,
        output.stride(0),
        output.stride(1),
        UNPACK=True,
        BLOCK=256,
    )
    return output


@triton.jit
def _pack_query(
    standard,
    packed,
    starts,
    reqs,
    tokens,
    heads,
    groups,
    dim,
    token_stride,
    head_stride,
    UNPACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % dim
    qhead = (i // dim) % (heads * groups)
    token = i // (dim * heads * groups)
    actual = tl.load(starts + reqs) // heads
    mask = token < actual
    src = standard + token * token_stride + qhead * head_stride + d
    dst = (
        packed
        + ((qhead // groups) * actual + token) * groups * dim
        + (qhead % groups) * dim
        + d
    )
    if UNPACK:
        x = tl.load(dst, mask=mask, other=0)
        tl.store(src, x, mask=token < tokens)
    else:
        x = tl.load(src, mask=mask, other=0)
        tl.store(dst, x, mask=mask)
