"""Startup-known dense model configuration for the shared native backend."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model: str
    kv_cache_memory_bytes: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float = 0.45
    revision: str | None = None
    prefill_reserve_blocks: int = 0
    prefill_sm_target: int = 0
    compile_decode: bool = False
    dtype: str = "float16"
    block_size: int = 16

    def __post_init__(self) -> None:
        if not self.model_id or not self.model:
            raise ValueError("each model needs an ID and checkpoint")
        if (
            min(
                self.kv_cache_memory_bytes,
                self.max_model_len,
                self.max_num_seqs,
                self.max_num_batched_tokens,
                self.block_size,
            )
            <= 0
        ):
            raise ValueError("KV bytes and model/scheduler capacities must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("GPU memory utilization must be in (0, 1]")
        if min(self.prefill_reserve_blocks, self.prefill_sm_target) < 0:
            raise ValueError("KV reserve and SM target must be nonnegative")
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("initial dense Llama path supports FP16/BF16 only")

    def engine_kwargs(self, *, eager: bool, decode_depth: int) -> dict[str, Any]:
        if eager and self.compile_decode:
            raise ValueError("compile_decode requires graph mode")
        return {
            "model": self.model,
            "revision": self.revision,
            "runner": "generate",
            "dtype": self.dtype,
            "seed": 0,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "block_size": self.block_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "async_scheduling": False,
            "enforce_eager": eager,
            "worker_cls": "vllm.v1.worker.slackserve_gpu_worker.SlackServeGPUWorker",
            "scheduler_cls": (
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
                if decode_depth > 1
                else "vllm.v1.core.sched.scheduler.Scheduler"
            ),
            # Dense FP16/BF16 Llama + FlashAttention avoids global FlashInfer,
            # MLA and MoE workspaces; those backends require a separate audit.
            "attention_config": {"backend": "FLASH_ATTN"},
            "compilation_config": {
                "mode": 3 if self.compile_decode else 0,
                "cudagraph_mode": "NONE" if eager else "FULL",
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
                    "async_decode_depth": decode_depth,
                    "compile_decode": self.compile_decode,
                    "prefill_sm_target": self.prefill_sm_target,
                }
            },
        }


def load_specs(path: Path, n: int, warmup: int) -> list[ModelSpec]:
    specs = [ModelSpec(**row) for row in json.loads(path.read_text())]
    if len(specs) != 2 or len({s.model_id for s in specs}) != 2:
        raise ValueError("this verifier requires exactly two unique model IDs")
    if any(s.max_num_seqs < max(n, warmup) for s in specs):
        raise ValueError("max_num_seqs must cover each complete fixed queue")
    if sum(s.gpu_memory_utilization for s in specs) > 0.95:
        raise ValueError("combined initialization utilization fractions exceed 0.95")
    return specs
