# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated server configuration for Slack Serve's dense sidecar."""

import json
import os
from argparse import Namespace
from dataclasses import fields

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.utils import is_init_field


def configure_dense_sidecar(args: Namespace, *, native: bool = False) -> None:
    """Apply the two-stream preset before ``AsyncLLM`` starts EngineCore."""
    model = getattr(args, "slackserve_dense_model", None)
    if model is None:
        return
    if native and getattr(args, "slackserve_models", None):
        raise ValueError("choose --slackserve-models or --slackserve-dense-model")

    unsupported_parallel = {
        name: getattr(args, name, 1)
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
        )
        if getattr(args, name, 1) != 1
    }
    if unsupported_parallel:
        raise ValueError(
            "--slackserve-dense-model currently requires one GPU "
            f"(TP=PP=DP=PCP=1); got {unsupported_parallel}"
        )

    dense_task = args.slackserve_dense_task
    if dense_task not in ("embed", "score", "classify"):
        raise ValueError(
            "--slackserve-dense-task must be embed, score, or classify; "
            f"got {dense_task!r}"
        )
    hf_overrides = args.slackserve_dense_hf_overrides
    if hf_overrides is not None:
        try:
            parsed_overrides = json.loads(hf_overrides)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--slackserve-dense-hf-overrides must be valid JSON"
            ) from exc
        if not isinstance(parsed_overrides, dict):
            raise ValueError("--slackserve-dense-hf-overrides must decode to an object")

    worker = "vllm.v1.worker.slackserve_gpu_worker.SlackServeGPUWorker"
    if args.worker_cls == "auto":
        args.worker_cls = worker
    elif args.worker_cls != worker:
        raise ValueError(
            "--slackserve-dense-model requires SlackServeGPUWorker; "
            f"got --worker-cls={args.worker_cls!r}"
        )
    if args.scheduler_cls is None:
        args.scheduler_cls = "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
    if args.async_scheduling is None:
        args.async_scheduling = False

    # Reproducible H100 long-context defaults. Ordinary vLLM primary-engine
    # flags and the dense-sidecar flags remain explicit overrides.
    if args.gpu_memory_utilization == 0.9:
        args.gpu_memory_utilization = 0.65
    if args.max_model_len is None:
        args.max_model_len = 11264
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = 65536
    if args.max_num_seqs is None:
        args.max_num_seqs = 192
    args.enable_prefix_caching = False

    compilation = args.compilation_config
    if (
        compilation.mode is None
        and compilation.cudagraph_mode is None
        and compilation.max_cudagraph_capture_size is None
        and not compilation.custom_ops
    ):
        compilation.mode = CompilationMode.VLLM_COMPILE
        compilation.cudagraph_mode = CUDAGraphMode.FULL
        compilation.max_cudagraph_capture_size = 200
        compilation.custom_ops = ["none"]
        compilation.pass_config.fuse_norm_quant = False
        compilation.pass_config.fuse_act_quant = False
        compilation.pass_config.fuse_attn_quant = False

    dense_gmu = args.slackserve_dense_gpu_memory_utilization
    if not 0 < dense_gmu < 1:
        raise ValueError(
            "--slackserve-dense-gpu-memory-utilization must be between 0 and 1"
        )
    if args.gpu_memory_utilization + dense_gmu > 0.9:
        raise ValueError(
            "primary and dense GPU-memory utilization must sum to at most 0.90; "
            f"got {args.gpu_memory_utilization + dense_gmu:.3f}"
        )

    if native:
        from vllm.v1.engine.multi_model_config import ModelSpec

        names = args.served_model_name or [args.model]
        if len(names) != 1:
            raise ValueError("native dense serving requires one primary model name")
        args.slackserve_models = [
            ModelSpec(
                model_id=names[0],
                model=args.model,
                kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                revision=args.revision,
                tokenizer=args.tokenizer,
                trust_remote_code=args.trust_remote_code,
                hf_overrides=args.hf_overrides,
                dtype=args.dtype,
                compile_decode=True,
                compilation={
                    field.name: getattr(compilation, field.name)
                    for field in fields(compilation)
                    if is_init_field(type(compilation), field.name)
                },
                prefill_custom_ops=os.environ.get(
                    "HB_P2_PREFILL_CUSTOM_OPS",
                    "+rms_norm,+silu_and_mul,+rotary_embedding",
                ),
                prefill_reserve_fraction=float(
                    os.environ.get("HB_P2_PREFILL_KV_WATERMARK", "0.02")
                ),
                prefill_sm_target=int(
                    os.environ.get("HB_PREFILL_SM_COUNT_TARGET", "96")
                ),
            ),
            ModelSpec(
                model_id=args.slackserve_dense_served_model_name,
                model=model,
                runner="pooling",
                pooling_task=dense_task,
                kv_cache_memory_bytes=args.slackserve_dense_kv_memory_bytes,
                max_model_len=args.slackserve_dense_max_model_len,
                max_num_seqs=args.slackserve_dense_max_num_seqs,
                max_num_batched_tokens=args.slackserve_dense_max_num_batched_tokens,
                gpu_memory_utilization=dense_gmu,
                dtype="bfloat16",
                trust_remote_code=args.trust_remote_code,
                hf_overrides=json.loads(hf_overrides) if hf_overrides else None,
                compilation={
                    "mode": 3,
                    "cudagraph_mode": "NONE",
                    "max_cudagraph_capture_size": 512,
                },
            ),
        ]
        args.slackserve_shared_kv = "private"
        if mask := os.environ.get("HB_SMCTRL_PREFILL_TPCS"):
            if not mask.startswith("spread:"):
                raise ValueError("native dense masking requires a spread:TPCS mask")
            if args.slackserve_smctrl_mask is not None:
                raise ValueError(
                    "choose the native mask flag or HB_SMCTRL_PREFILL_TPCS"
                )
            args.slackserve_smctrl_mask = f"*:prefill:{mask.split(':', 1)[1]}"
            args.slackserve_smctrl_lib = os.environ.get("HB_SMCTRL_LIB")
        return

    os.environ.update(
        {
            "HB_LANE_ROUTING": "1",
            "HB_P2_SERVE": "1",
            "HB_P2_EMBED_SIDECAR": "1",
            "HB_P2_DENSE_LIVE": "1",
            "HB_P2_PREFILL_LANES": "1",
            "HB_P2_PREFILL_SHARED_STREAM": "1",
            "HB_P2_EMBED_MODEL": model,
            "HB_P2_DENSE_SERVED_MODEL": args.slackserve_dense_served_model_name,
            "HB_P2_DENSE_TASK": dense_task,
            "HB_P2_EMBED_GMU": str(dense_gmu),
            "HB_P2_EMBED_BUDGET": str(args.slackserve_dense_max_num_batched_tokens),
            "HB_P2_EMBED_MAX_MODEL_LEN": str(args.slackserve_dense_max_model_len),
            "HB_P2_EMBED_MAX_NUM_SEQS": str(args.slackserve_dense_max_num_seqs),
            # Topology-B correctness invariants. Override stale experiment
            # variables rather than silently running a different topology.
            "HB_P2_ASYNC_DECODE": "1",
            "HB_P2_DENSE_LOCK": "0",
            "HB_P2_DECODE_GRAPHS_ONLY": "1",
            "HB_P2_COMPILE_DECODE": "1",
            "HB_P2_EMBED_ASYNC_SCHED": "0",
        }
    )
    if hf_overrides is None:
        os.environ.pop("HB_P2_DENSE_HF_OVERRIDES", None)
    else:
        os.environ["HB_P2_DENSE_HF_OVERRIDES"] = hf_overrides
    for name, value in {
        "HB_PREFILL_SM_COUNT_TARGET": "96",
        # Small KV admission reserve; async-ticket refcounts provide the
        # correctness boundary when the physical KV pool becomes tight.
        "HB_P2_PREFILL_KV_WATERMARK": "0.02",
        "HB_LT_ALGO_SELECT": "prefer:d1,18,17",
        # Eager prefill needs these fused kernels when decode owns compilation.
        "HB_P2_PREFILL_CUSTOM_OPS": "+rms_norm,+silu_and_mul,+rotary_embedding",
        "HB_P2_EMBED_SM_TARGET": "0",
        "HB_P2_EMBED_UNCAP_IDLE": "1",
    }.items():
        os.environ.setdefault(name, value)
