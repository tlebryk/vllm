# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import inspect
from dataclasses import is_dataclass
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.dual_model_helpers import (
    DualModelConfig,
    merge_model_runner_outputs,
    split_scheduler_output_by_model,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class DualModelRunner:
    """Experimental wrapper that hosts one decode and one embed runner."""

    EMBED_MODEL_PREFIX = "embed_model"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if not envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("DualModelRunner currently requires V2 model runner.")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "DualModelRunner currently does not support async scheduling."
            )
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "DualModelRunner currently requires pipeline_parallel_size=1."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DualModelRunner currently does not support speculative decoding."
            )
        # add two streams (hardcoded for now)
        self.decode_stream = torch.cuda.default_stream(device)
        self.embed_stream = torch.cuda.Stream(device)

        dual_cfg = DualModelConfig.from_vllm_config(vllm_config)
        if dual_cfg is None:
            raise ValueError(
                "DualModelRunner requires additional_config['dual_model']."
            )

        self.vllm_config = vllm_config
        self.dual_cfg = dual_cfg
        self.device = device
        self.decode_runner = GPUModelRunner(vllm_config, device)
        self.embed_vllm_config = self._build_embed_vllm_config(vllm_config, dual_cfg)
        self.embed_runner = GPUModelRunner(self.embed_vllm_config, device)

        self.model_memory_usage = 0
        self.is_pooling_model = False
        self.model: nn.Module | None = None
        self.req_id_to_model_id: dict[str, str] = {}
        self.pending_embed_output: ModelRunnerOutput | None = None
        self.pending_embed_needs_pool = False
        self.pending_req_order: list[str] = []
        self.decode_kv_group_indices: tuple[int, ...] = ()
        self.embed_kv_group_indices: tuple[int, ...] = ()
        self.decode_kv_cache_spec: dict[str, Any] = {}
        self.embed_kv_cache_spec: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        decode_runner = self.__dict__.get("decode_runner")
        if decode_runner is not None:
            return getattr(decode_runner, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @staticmethod
    def _replace_dataclass(instance: Any, **updates: Any) -> Any:
        cls = type(instance)
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass.")
        init_kwargs: dict[str, Any] = {}
        for name, parameter in inspect.signature(cls).parameters.items():
            if name == "self":
                continue
            if name in updates:
                init_kwargs[name] = updates.pop(name)
            elif hasattr(instance, name):
                init_kwargs[name] = getattr(instance, name)
            elif parameter.default is inspect._empty:
                raise ValueError(
                    f"Missing required constructor argument {name!r} for "
                    f"{cls.__name__}."
                )
        init_kwargs.update(updates)
        return cls(**init_kwargs)

    def _build_embed_vllm_config(
        self,
        vllm_config: VllmConfig,
        dual_cfg: DualModelConfig,
    ) -> VllmConfig:
        # The embed runner must own an independent config tree so model-load
        # side effects such as static_forward_context registration do not alias
        # the decode runner's state.
        embed_vllm_config = copy.deepcopy(vllm_config)
        embed_model_config = self._replace_dataclass(
            embed_vllm_config.model_config,
            model=dual_cfg.embed_model,
            tokenizer=dual_cfg.embed_tokenizer or dual_cfg.embed_model,
            runner="pooling",
            served_model_name=dual_cfg.embed_model,
            dtype=dual_cfg.embed_dtype or embed_vllm_config.model_config.dtype,
            max_model_len=(
                dual_cfg.embed_max_model_len
                or embed_vllm_config.model_config.max_model_len
            ),
            enforce_eager=(
                dual_cfg.embed_enforce_eager
                if dual_cfg.embed_enforce_eager is not None
                else embed_vllm_config.model_config.enforce_eager
            ),
            model_weights="",
            hf_config_path=None,
        )
        embed_scheduler_config = self._replace_dataclass(
            embed_vllm_config.scheduler_config,
            max_model_len=embed_model_config.max_model_len,
            is_encoder_decoder=embed_model_config.is_encoder_decoder,
            runner_type="pooling",
        )
        embed_vllm_config.model_config = embed_model_config
        embed_vllm_config.scheduler_config = embed_scheduler_config
        embed_vllm_config.additional_config = {}
        embed_vllm_config.compilation_config.static_forward_context.clear()
        embed_vllm_config.compilation_config.static_all_moe_layers.clear()
        return embed_vllm_config

    def update_max_model_len(self, max_model_len: int) -> None:
        self.decode_runner.update_max_model_len(max_model_len)
        self.embed_runner.update_max_model_len(max_model_len)

    @staticmethod
    def _project_kv_cache_config(
        kv_cache_config: KVCacheConfig,
        runner_kv_cache_spec: dict[str, Any],
    ) -> tuple[KVCacheConfig, tuple[int, ...]]:
        runner_layer_names = set(runner_kv_cache_spec)
        projected_groups: list[KVCacheGroupSpec] = []
        group_indices: list[int] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            runner_group_layers = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in runner_layer_names
            ]
            if not runner_group_layers:
                continue
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={
                        layer_name: group_spec.kv_cache_specs[layer_name]
                        for layer_name in runner_group_layers
                    },
                )
            projected_groups.append(
                KVCacheGroupSpec(
                    layer_names=runner_group_layers,
                    kv_cache_spec=group_spec,
                )
            )
            group_indices.append(group_idx)

        projected_tensors = [
            KVCacheTensor(
                size=kv_cache_tensor.size,
                shared_by=[
                    layer_name
                    for layer_name in kv_cache_tensor.shared_by
                    if layer_name in runner_layer_names
                ],
            )
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
            if any(
                layer_name in runner_layer_names
                for layer_name in kv_cache_tensor.shared_by
            )
        ]
        return (
            KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=projected_tensors,
                kv_cache_groups=projected_groups,
            ),
            tuple(group_indices),
        )

    def get_supported_tasks(self) -> tuple[str, ...]:
        tasks = list(self.decode_runner.get_supported_tasks())
        for task in self.embed_runner.get_supported_tasks():
            if task not in tasks:
                tasks.append(task)
        return tuple(tasks)

    def load_model(self, *args, **kwargs) -> None:
        self.decode_runner.load_model(*args, **kwargs)
        self.embed_runner.load_model(
            *args,
            prefix=self.EMBED_MODEL_PREFIX,
            **kwargs,
        )
        self.decode_kv_cache_spec = self.decode_runner.get_kv_cache_spec()
        self.embed_kv_cache_spec = self.embed_runner.get_kv_cache_spec()
        overlap = set(self.decode_kv_cache_spec) & set(self.embed_kv_cache_spec)
        if overlap:
            raise ValueError(
                "Decode/embed KV layer names must be disjoint; overlapping names: "
                f"{sorted(overlap)[:4]}"
            )
        self.model_memory_usage = (
            self.decode_runner.model_memory_usage
            + self.embed_runner.model_memory_usage
        )
        self.model = self.decode_runner.model

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self):
        return {
            **self.decode_kv_cache_spec,
            **self.embed_kv_cache_spec,
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        decode_kv_cache_config, self.decode_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.decode_kv_cache_spec)
        )
        embed_kv_cache_config, self.embed_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.embed_kv_cache_spec)
        )
        self.decode_runner.initialize_kv_cache(decode_kv_cache_config)
        self.embed_runner.initialize_kv_cache(embed_kv_cache_config)

    def profile_run(self) -> None:
        self.decode_runner.profile_run()
        self.embed_runner.profile_run()

    def reset_mm_cache(self) -> None:
        self.decode_runner.reset_mm_cache()
        self.embed_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.decode_runner.reset_encoder_cache()
        self.embed_runner.reset_encoder_cache()

    def profile_cudagraph_memory(self) -> int:
        return (
            self.decode_runner.profile_cudagraph_memory()
            + self.embed_runner.profile_cudagraph_memory()
        )

    def capture_model(self) -> int:
        return (
            self.decode_runner.capture_model()
            + self.embed_runner.capture_model()
        )

    def _dummy_run(self, *args, **kwargs):
        return self.decode_runner._dummy_run(*args, **kwargs)

    def _dummy_sampler_run(self, *args, **kwargs) -> None:
        self.decode_runner._dummy_sampler_run(*args, **kwargs)

    def _dummy_pooler_run(self, *args, **kwargs) -> None:
        self.embed_runner._dummy_pooler_run(*args, **kwargs)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ):
        if dummy_run:
            self.decode_runner.execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            )
            self.embed_runner.profile_run()
            return None

        split_outputs = split_scheduler_output_by_model(
            scheduler_output=scheduler_output,
            req_id_to_model_id=self.req_id_to_model_id,
            decode_model_id=self.dual_cfg.decode_model_id,
            embed_model_id=self.dual_cfg.embed_model_id,
            decode_kv_group_indices=self.decode_kv_group_indices,
            embed_kv_group_indices=self.embed_kv_group_indices,
        )
        self.pending_req_order = split_outputs.scheduled_req_order

        decode_output = None
        if split_outputs.decode.num_scheduled_tokens or split_outputs.decode.finished_req_ids:
            with torch.cuda.stream(self.decode_stream):
                decode_output = self.decode_runner.execute_model(
                    split_outputs.decode,
                    intermediate_tensors=intermediate_tensors,
                )

        embed_exec_output = None
        embed_output: ModelRunnerOutput | None = None
        if split_outputs.embed.num_scheduled_tokens or split_outputs.embed.finished_req_ids:
            with torch.cuda.stream(self.embed_stream):
                embed_exec_output = self.embed_runner.execute_model(
                    split_outputs.embed,
                )

        decode_needs_sample = (
            decode_output is None and bool(split_outputs.decode.num_scheduled_tokens)
        )
        if decode_needs_sample:
            self.pending_embed_output = embed_exec_output
            self.pending_embed_needs_pool = (
                split_outputs.embed.num_scheduled_tokens
                or split_outputs.embed.finished_req_ids
            ) and embed_exec_output is None
            return None

        if split_outputs.embed.num_scheduled_tokens or split_outputs.embed.finished_req_ids:
            embed_output = (
                self.embed_runner.pool()
                if embed_exec_output is None
                else embed_exec_output
            )
            if embed_output is None:
                raise RuntimeError("Embed runner failed to produce pooling output.")

        self.pending_embed_output = None
        self.pending_embed_needs_pool = False
        return merge_model_runner_outputs(
            scheduled_req_order=self.pending_req_order,
            decode_output=decode_output,
            embed_output=embed_output,
        )

    def sample_tokens(self, grammar_output: GrammarOutput | None):
        decode_output = self.decode_runner.sample_tokens(grammar_output)
        embed_output = self.pending_embed_output
        if self.pending_embed_needs_pool:
            embed_output = self.embed_runner.pool()
            if embed_output is None:
                raise RuntimeError("Embed runner failed to produce pooling output.")
        merged = merge_model_runner_outputs(
            scheduled_req_order=self.pending_req_order,
            decode_output=decode_output,
            embed_output=embed_output,
        )
        self.pending_embed_output = None
        self.pending_embed_needs_pool = False
        self.pending_req_order = []
        return merged

    def take_draft_token_ids(self):
        return self.decode_runner.take_draft_token_ids()
