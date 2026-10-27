# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early, CUDA-free validation for the initial colocated TP pooling scope."""


def validate_colocated_models(specs, mode, topology):
    sizes = [getattr(s, "tensor_parallel_size", 1) for s in specs]
    pp_sizes = [getattr(s, "pipeline_parallel_size", 1) for s in specs]
    if not any(t * p > 1 for t, p in zip(sizes, pp_sizes)):
        return False
    if mode != "private":
        raise ValueError("TP native serving currently requires private KV pools")
    if len(set(sizes)) != 1:
        raise ValueError("colocated models require identical TP sizes")
    if len(specs) == 1:
        if topology == "single-stream":
            raise ValueError("single-model TP serial mode is not implemented")
        return False
    if any(p > 1 for p in pp_sizes):
        raise ValueError("colocated pooling does not yet support pipeline parallelism")
    if len(specs) != 2 or [s.runner for s in specs] != ["generate", "pooling"]:
        raise ValueError(
            "colocated TP requires generator first, then one pooling model"
        )
    if topology not in ("single-stream", "three-stream-overlap"):
        raise ValueError("colocated TP supports serial or generator/pooling overlap")
    if any(s.kv_cache_memory_bytes is None for s in specs):
        raise ValueError("colocated TP initially requires explicit private KV budgets")
    return True
