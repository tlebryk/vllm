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
    # Optional activation-profile ceiling, allowing smaller runtime work units
    # without increasing KV allocation or changing the measured capacity.
    prefill_profile_tokens: int | None = None
    attention_backend: str = "FLASH_ATTN"
    flash_attn_version: int | None = None

    def __post_init__(self) -> None:
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
        if self.runner == "pooling" and self.lane_staging:
            raise ValueError("lane_staging currently requires the generation runner")
        if not 0 <= self.prefill_reserve_fraction < 1:
            raise ValueError("prefill reserve fraction must be in [0, 1)")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("GPU memory utilization must be in (0, 1]")
        if min(self.prefill_reserve_blocks, self.prefill_sm_target) < 0:
            raise ValueError("KV reserve and SM target must be nonnegative")
        if not 0 <= self.prefill_step_tokens <= self.max_num_batched_tokens:
            raise ValueError("prefill step tokens must fit the engine token capacity")
        if self.prefill_profile_tokens is not None and not (
            (self.prefill_step_tokens or self.max_num_batched_tokens)
            <= self.prefill_profile_tokens <= self.max_num_batched_tokens
        ):
            raise ValueError("prefill profile tokens must cover the policy work unit")
        if self.dtype not in ("float16", "bfloat16", "auto"):
            raise ValueError("dense runners support FP16/BF16 or auto in private mode")

    def engine_kwargs(self, *, eager: bool, decode_depth: int) -> dict[str, Any]:
        if eager and self.compile_decode:
            raise ValueError("compile_decode requires graph mode")
        pooling = self.runner == "pooling"
        kwargs = {
            "model": self.model,
            "revision": self.revision,
            "runner": self.runner,
            "dtype": self.dtype,
            "seed": 0,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": False,
            "block_size": self.block_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "async_scheduling": False,
            "enforce_eager": eager,
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
            "attention_config": {"backend": self.attention_backend, "flash_attn_version": self.flash_attn_version},
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
                    "prefill_custom_ops": self.prefill_custom_ops,
                    "prefill_step_tokens": self.prefill_profile_tokens
                    if self.prefill_profile_tokens is not None
                    else self.prefill_step_tokens,
                }
            },
        }
        if self.compilation is not None:
            kwargs["compilation_config"] = self.compilation
        if self.hf_overrides is not None:
            kwargs["hf_overrides"] = self.hf_overrides
        if self.tokenizer is not None:
            kwargs["tokenizer"] = self.tokenizer
        if pooling:
            kwargs["convert"] = "embed" if self.pooling_task == "embed" else "classify"
        return kwargs


def load_specs(path: Path, n: int, warmup: int) -> list[ModelSpec]:
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
