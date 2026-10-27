# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-known one/two-model configuration for the native lane backend."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model: str
    kv_cache_memory_bytes: int | None
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float = 0.45
    revision: str | None = None
    prefill_reserve_blocks: int = 0
    prefill_sm_target: int = 0
    compile_decode: bool = False
    lane_staging: bool = False
    dtype: str = "float16"
    block_size: int = 16
    runner: str = "generate"
    pooling_task: str | None = None
    hf_overrides: dict | None = None
    tokenizer: str | None = None
    trust_remote_code: bool = False
    compilation: dict | None = None
    prefill_custom_ops: str = ""
    prefill_reserve_fraction: float = 0.0
    # Policy work unit, separate from the engine's scheduler capacity.
    prefill_step_tokens: int = 0
    # None preserves backend auto-selection. A100 replication can pin FA2.
    flash_attn_version: int | None = None
    attention_backend: str = "FLASH_ATTN"
    # None preserves upstream behavior; FA-only replication can skip FI tuning.
    enable_flashinfer_autotune: bool | None = None
    # Dispatch this model only after other GPU tickets have completed. Other
    # models keep their normal P/D overlap when this model is not executing.
    exclusive_execution: bool = False
    # Opt-in session reuse, currently supported only with private KV pools.
    enable_prefix_caching: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False

    def __post_init__(self) -> None:
        if type(self.enable_prefix_caching) is not bool:
            raise ValueError("enable_prefix_caching must be a boolean")
        if not self.model_id or not self.model:
            raise ValueError("each model needs an ID and checkpoint")
        if (
            min(
                self.max_model_len,
                self.max_num_seqs,
                self.max_num_batched_tokens,
                self.block_size,
            )
            <= 0
        ):
            raise ValueError("KV bytes and model/scheduler capacities must be positive")
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes <= 0:
            raise ValueError(
                "KV bytes must be positive or None for private auto sizing"
            )
        if self.runner not in ("generate", "pooling"):
            raise ValueError("runner must be generate or pooling")
        if self.runner == "pooling" and self.pooling_task not in (
            "embed",
            "classify",
            "score",
        ):
            raise ValueError("pooling runner requires an embed/classify/score task")
        if self.runner == "generate" and self.pooling_task is not None:
            raise ValueError("generation runner cannot specify a pooling task")
        if type(self.exclusive_execution) is not bool:
            raise ValueError("exclusive_execution must be a boolean")
        if self.exclusive_execution and self.runner != "pooling":
            raise ValueError("exclusive_execution currently requires a pooling model")
        if self.runner == "pooling" and self.lane_staging:
            raise ValueError("lane_staging currently requires the generation runner")
        if not 0 <= self.prefill_reserve_fraction < 1:
            raise ValueError("prefill reserve fraction must be in [0, 1)")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if (
            type(self.pipeline_parallel_size) is not int
            or self.pipeline_parallel_size < 1
        ):
            raise ValueError("pipeline_parallel_size must be a positive integer")
        if type(self.enable_expert_parallel) is not bool:
            raise ValueError("enable_expert_parallel must be a boolean")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("GPU memory utilization must be in (0, 1]")
        if min(self.prefill_reserve_blocks, self.prefill_sm_target) < 0:
            raise ValueError("KV reserve and SM target must be nonnegative")
        if not 0 <= self.prefill_step_tokens <= self.max_num_batched_tokens:
            raise ValueError("prefill step tokens must fit the engine token capacity")
        if self.dtype not in ("float16", "bfloat16", "auto"):
            raise ValueError("dense runners support FP16/BF16 or auto in private mode")
        if self.flash_attn_version not in (None, 2, 3, 4):
            raise ValueError(
                "native attention supports automatic selection, FA2, FA3 or FA4"
            )
        if (
            self.enable_flashinfer_autotune is not None
            and type(self.enable_flashinfer_autotune) is not bool
        ):
            raise ValueError("FlashInfer autotune must be a boolean or None")

    def engine_kwargs(self, *, eager: bool, decode_depth: int) -> dict[str, Any]:
        if eager and self.compile_decode:
            raise ValueError("compile_decode requires graph mode")
        pooling = self.runner == "pooling"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "revision": self.revision,
            "runner": self.runner,
            "dtype": self.dtype,
            "seed": 0,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": self.enable_prefix_caching,
            "block_size": self.block_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "async_scheduling": False,
            "enforce_eager": eager,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "enable_expert_parallel": self.enable_expert_parallel,
            "distributed_executor_backend": (
                "mp"
                if self.tensor_parallel_size * self.pipeline_parallel_size > 1
                else None
            ),
            "worker_cls": (
                "vllm.v1.worker.slackserve_pooling_worker.SlackServePoolingWorker"
                if pooling
                else "vllm.v1.worker.slackserve_gpu_worker.SlackServeGPUWorker"
            ),
            "scheduler_cls": (
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
                if decode_depth > 1 and not pooling
                else "vllm.v1.core.sched.scheduler.Scheduler"
            ),
            # Dense FP16/BF16 Llama + FlashAttention avoids global FlashInfer,
            # MLA and MoE workspaces; those backends require a separate audit.
            "attention_config": {"backend": self.attention_backend},
            "compilation_config": {
                "mode": 3 if self.compile_decode or pooling else 0,
                "cudagraph_mode": "NONE" if eager or pooling else "FULL",
                "max_cudagraph_capture_size": min(self.max_num_seqs * 2, 200),
                "custom_ops": ["all"],
                "pass_config": {
                    "fuse_norm_quant": False,
                    "fuse_act_quant": False,
                    "fuse_attn_quant": False,
                },
            },
            "additional_config": {
                "slackserve": {
                    "model_id": self.model_id,
                    "decode_graphs_only": True,
                    "async_decode_depth": decode_depth if not pooling else 1,
                    "compile_decode": self.compile_decode,
                    "lane_staging": self.lane_staging,
                    "prefill_sm_target": self.prefill_sm_target,
                    "prefill_step_tokens": self.prefill_step_tokens,
                    "prefill_custom_ops": self.prefill_custom_ops,
                }
            },
        }
        if self.compilation is not None:
            kwargs["compilation_config"] = self.compilation
        if self.hf_overrides is not None:
            kwargs["hf_overrides"] = self.hf_overrides
        if self.tokenizer is not None:
            kwargs["tokenizer"] = self.tokenizer
        if self.flash_attn_version is not None:
            kwargs["attention_config"]["flash_attn_version"] = self.flash_attn_version
        if self.enable_flashinfer_autotune is not None:
            kwargs["kernel_config"] = {
                "enable_flashinfer_autotune": self.enable_flashinfer_autotune
            }
        if pooling:
            kwargs["convert"] = "embed" if self.pooling_task == "embed" else "classify"
        return kwargs


def load_specs(
    path: Path | list[ModelSpec] | tuple[ModelSpec, ...], n: int, warmup: int
) -> list[ModelSpec]:
    specs = (
        list(path)
        if isinstance(path, (list, tuple))
        else [ModelSpec(**row) for row in json.loads(path.read_text())]
    )
    if not 1 <= len(specs) <= 2 or len({s.model_id for s in specs}) != len(specs):
        raise ValueError("native serving requires one or two unique model IDs")
    if any(s.max_num_seqs < max(n, warmup) for s in specs):
        raise ValueError("max_num_seqs must cover each complete fixed queue")
    if sum(s.gpu_memory_utilization for s in specs) > 0.95:
        raise ValueError("combined initialization utilization fractions exceed 0.95")
    return specs
