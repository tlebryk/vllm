# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


DUAL_MODEL_CONFIG_KEY = "dual_model"
DEFAULT_DECODE_MODEL_ID = "decode"
DEFAULT_EMBED_MODEL_ID = "embed"


@dataclass(frozen=True)
class DualModelConfig:
    embed_model: str
    decode_model_id: str = DEFAULT_DECODE_MODEL_ID
    embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    embed_tokenizer: str | None = None
    embed_dtype: str | None = None
    embed_max_model_len: int | None = None
    embed_enforce_eager: bool | None = None
    decode_running_reserve: int | None = None
    max_embed_running_reqs: int | None = None
    embed_waiting_min_batch_reqs: int | None = None
    embed_release_running_decode_threshold: int | None = None
    embed_release_when_decode_waiting_drained: bool = False

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "DualModelConfig | None":
        additional_config = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional_config, dict):
            return None
        raw_cfg = additional_config.get(DUAL_MODEL_CONFIG_KEY)
        if not isinstance(raw_cfg, dict):
            return None

        embed_model = raw_cfg.get("embed_model")
        if not embed_model:
            raise ValueError(
                "additional_config['dual_model']['embed_model'] is required."
            )

        return cls(
            embed_model=embed_model,
            decode_model_id=raw_cfg.get(
                "decode_model_id", DEFAULT_DECODE_MODEL_ID
            ),
            embed_model_id=raw_cfg.get("embed_model_id", DEFAULT_EMBED_MODEL_ID),
            embed_tokenizer=raw_cfg.get("embed_tokenizer"),
            embed_dtype=raw_cfg.get("embed_dtype"),
            embed_max_model_len=raw_cfg.get("embed_max_model_len"),
            embed_enforce_eager=raw_cfg.get("embed_enforce_eager"),
            decode_running_reserve=raw_cfg.get("decode_running_reserve"),
            max_embed_running_reqs=raw_cfg.get("max_embed_running_reqs"),
            embed_waiting_min_batch_reqs=raw_cfg.get(
                "embed_waiting_min_batch_reqs"
            ),
            embed_release_running_decode_threshold=raw_cfg.get(
                "embed_release_running_decode_threshold"
            ),
            embed_release_when_decode_waiting_drained=bool(
                raw_cfg.get("embed_release_when_decode_waiting_drained", False)
            ),
        )


@dataclass
class SplitSchedulerOutputs:
    decode: SchedulerOutput
    embed: SchedulerOutput
    scheduled_req_order: list[str]


def _filter_cached_request_data(
    cached: CachedRequestData,
    keep_req_ids: set[str],
    group_indices: tuple[int, ...],
) -> CachedRequestData:
    if not keep_req_ids:
        return CachedRequestData.make_empty()

    kept_indices = [
        idx for idx, req_id in enumerate(cached.req_ids) if req_id in keep_req_ids
    ]
    req_ids = [cached.req_ids[idx] for idx in kept_indices]

    if len(cached.new_token_ids) == len(cached.req_ids):
        new_token_ids = [cached.new_token_ids[idx] for idx in kept_indices]
    else:
        # In the normal non-PP path, new_token_ids is intentionally empty even
        # when req_ids is populated. Preserve that contract instead of assuming
        # every CachedRequestData list is req-aligned.
        assert not cached.new_token_ids, (
            "CachedRequestData.new_token_ids should either be empty or match "
            "req_ids length."
        )
        new_token_ids = []

    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids={
            req_id for req_id in cached.resumed_req_ids if req_id in keep_req_ids
        },
        new_token_ids=new_token_ids,
        all_token_ids={
            req_id: token_ids
            for req_id, token_ids in cached.all_token_ids.items()
            if req_id in keep_req_ids
        },
        new_block_ids=[
            _project_block_ids(cached.new_block_ids[idx], group_indices)
            for idx in kept_indices
        ],
        num_computed_tokens=[
            cached.num_computed_tokens[idx] for idx in kept_indices
        ],
        num_output_tokens=[cached.num_output_tokens[idx] for idx in kept_indices],
    )


def _project_block_ids(
    block_ids: tuple[list[int], ...] | None,
    group_indices: tuple[int, ...],
) -> tuple[list[int], ...] | None:
    if block_ids is None:
        return None
    return tuple(block_ids[group_idx] for group_idx in group_indices)


def _project_new_request_data(
    req: NewRequestData,
    group_indices: tuple[int, ...],
) -> NewRequestData:
    return NewRequestData(
        req_id=req.req_id,
        model_id=req.model_id,
        prompt_token_ids=req.prompt_token_ids,
        mm_features=req.mm_features,
        sampling_params=req.sampling_params,
        pooling_params=req.pooling_params,
        block_ids=_project_block_ids(req.block_ids, group_indices) or (),
        num_computed_tokens=req.num_computed_tokens,
        lora_request=req.lora_request,
        prompt_embeds=req.prompt_embeds,
        prefill_token_ids=req.prefill_token_ids,
    )


def _project_num_common_prefix_blocks(
    num_common_prefix_blocks: list[int],
    group_indices: tuple[int, ...],
) -> list[int]:
    if not num_common_prefix_blocks:
        return []
    return [num_common_prefix_blocks[group_idx] for group_idx in group_indices]


def split_scheduler_output_by_model(
    scheduler_output: SchedulerOutput,
    req_id_to_model_id: dict[str, str],
    decode_model_id: str,
    embed_model_id: str,
    decode_kv_group_indices: tuple[int, ...],
    embed_kv_group_indices: tuple[int, ...],
) -> SplitSchedulerOutputs:
    decode_new_reqs = [
        _project_new_request_data(req, decode_kv_group_indices)
        for req in scheduler_output.scheduled_new_reqs
        if req.model_id == decode_model_id
    ]
    embed_new_reqs = [
        _project_new_request_data(req, embed_kv_group_indices)
        for req in scheduler_output.scheduled_new_reqs
        if req.model_id == embed_model_id
    ]

    for req in scheduler_output.scheduled_new_reqs:
        req_id_to_model_id[req.req_id] = req.model_id

    scheduled_req_order = list(scheduler_output.num_scheduled_tokens.keys())
    decode_req_ids = {
        req_id
        for req_id in scheduled_req_order
        if req_id_to_model_id.get(req_id) == decode_model_id
    }
    embed_req_ids = {
        req_id
        for req_id in scheduled_req_order
        if req_id_to_model_id.get(req_id) == embed_model_id
    }

    decode_finished = {
        req_id
        for req_id in scheduler_output.finished_req_ids
        if req_id_to_model_id.get(req_id) == decode_model_id
    }
    embed_finished = {
        req_id
        for req_id in scheduler_output.finished_req_ids
        if req_id_to_model_id.get(req_id) == embed_model_id
    }

    decode_preempted = {
        req_id
        for req_id in (scheduler_output.preempted_req_ids or set())
        if req_id_to_model_id.get(req_id) == decode_model_id
    }
    embed_preempted = {
        req_id
        for req_id in (scheduler_output.preempted_req_ids or set())
        if req_id_to_model_id.get(req_id) == embed_model_id
    }

    decode_output = SchedulerOutput(
        scheduled_new_reqs=decode_new_reqs,
        scheduled_cached_reqs=_filter_cached_request_data(
            scheduler_output.scheduled_cached_reqs,
            decode_req_ids,
            decode_kv_group_indices,
        ),
        num_scheduled_tokens={
            req_id: count
            for req_id, count in scheduler_output.num_scheduled_tokens.items()
            if req_id in decode_req_ids
        },
        total_num_scheduled_tokens=sum(
            count
            for req_id, count in scheduler_output.num_scheduled_tokens.items()
            if req_id in decode_req_ids
        ),
        scheduled_spec_decode_tokens={
            req_id: token_ids
            for req_id, token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()
            )
            if req_id in decode_req_ids
        },
        scheduled_encoder_inputs={
            req_id: input_ids
            for req_id, input_ids in scheduler_output.scheduled_encoder_inputs.items()
            if req_id in decode_req_ids
        },
        num_common_prefix_blocks=_project_num_common_prefix_blocks(
            scheduler_output.num_common_prefix_blocks,
            decode_kv_group_indices,
        ),
        finished_req_ids=decode_finished,
        free_encoder_mm_hashes=scheduler_output.free_encoder_mm_hashes,
        preempted_req_ids=decode_preempted,
        has_structured_output_requests=(
            scheduler_output.has_structured_output_requests and bool(decode_req_ids)
        ),
        pending_structured_output_tokens=(
            scheduler_output.pending_structured_output_tokens and bool(decode_req_ids)
        ),
        num_invalid_spec_tokens=(
            {
                req_id: count
                for req_id, count in (
                    scheduler_output.num_invalid_spec_tokens or {}
                ).items()
                if req_id in decode_req_ids
            }
            or None
        ),
        kv_connector_metadata=scheduler_output.kv_connector_metadata,
        ec_connector_metadata=scheduler_output.ec_connector_metadata,
        new_block_ids_to_zero=scheduler_output.new_block_ids_to_zero,
    )
    embed_output = SchedulerOutput(
        scheduled_new_reqs=embed_new_reqs,
        scheduled_cached_reqs=_filter_cached_request_data(
            scheduler_output.scheduled_cached_reqs,
            embed_req_ids,
            embed_kv_group_indices,
        ),
        num_scheduled_tokens={
            req_id: count
            for req_id, count in scheduler_output.num_scheduled_tokens.items()
            if req_id in embed_req_ids
        },
        total_num_scheduled_tokens=sum(
            count
            for req_id, count in scheduler_output.num_scheduled_tokens.items()
            if req_id in embed_req_ids
        ),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=_project_num_common_prefix_blocks(
            scheduler_output.num_common_prefix_blocks,
            embed_kv_group_indices,
        ),
        finished_req_ids=embed_finished,
        free_encoder_mm_hashes=[],
        preempted_req_ids=embed_preempted,
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        num_invalid_spec_tokens=None,
        kv_connector_metadata=None,
        ec_connector_metadata=None,
        new_block_ids_to_zero=scheduler_output.new_block_ids_to_zero,
    )

    for req_id in decode_finished | embed_finished:
        req_id_to_model_id.pop(req_id, None)

    return SplitSchedulerOutputs(
        decode=decode_output,
        embed=embed_output,
        scheduled_req_order=scheduled_req_order,
    )


def merge_model_runner_outputs(
    scheduled_req_order: list[str],
    decode_output: ModelRunnerOutput | None,
    embed_output: ModelRunnerOutput | None,
) -> ModelRunnerOutput:
    req_id_to_index = {
        req_id: index for index, req_id in enumerate(scheduled_req_order)
    }
    merged = ModelRunnerOutput(
        req_ids=scheduled_req_order,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=[[] for _ in scheduled_req_order],
        pooler_output=[None for _ in scheduled_req_order],
    )

    if decode_output is not None:
        for req_id in decode_output.req_ids:
            merged_index = req_id_to_index[req_id]
            decode_index = decode_output.req_id_to_index[req_id]
            if decode_output.sampled_token_ids:
                merged.sampled_token_ids[merged_index] = (
                    decode_output.sampled_token_ids[decode_index]
                )
        merged.prompt_logprobs_dict.update(decode_output.prompt_logprobs_dict)
        merged.kv_connector_output = decode_output.kv_connector_output
        merged.num_nans_in_logits = decode_output.num_nans_in_logits
        merged.cudagraph_stats = decode_output.cudagraph_stats
        if embed_output is None:
            merged.logprobs = decode_output.logprobs

    if embed_output is not None:
        for req_id in embed_output.req_ids:
            merged_index = req_id_to_index[req_id]
            embed_index = embed_output.req_id_to_index[req_id]
            if embed_output.pooler_output is not None:
                merged.pooler_output[merged_index] = (
                    embed_output.pooler_output[embed_index]
                )
        merged.prompt_logprobs_dict.update(embed_output.prompt_logprobs_dict)
        merged.kv_connector_output = _merge_kv_connector_output(
            merged.kv_connector_output,
            embed_output.kv_connector_output,
        )

    return merged


def _merge_kv_connector_output(
    lhs: KVConnectorOutput | None,
    rhs: KVConnectorOutput | None,
) -> KVConnectorOutput | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return KVConnectorOutput.merge(lhs, rhs)
