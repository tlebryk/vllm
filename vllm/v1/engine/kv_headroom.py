# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predict native admission demand with the real hybrid-cache allocator."""

from collections.abc import Iterable

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


def additional_blocks(cache: KVCacheManager, request: Request, tokens: int) -> int:
    tokens = min(tokens, cache.max_model_len)
    return cache.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=tokens,
        new_computed_blocks=cache.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=request.num_computed_tokens,
        num_local_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=tokens,
        # Prediction must not mutate Mamba's per-step checkpoint decision.
        # The admission cap also accounts for recycled SWA/state storage.
        apply_admission_cap=True,
        prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
    )


class CacheHeadroom:
    """Reserve measured decode growth and charge new prefills across all groups."""

    def __init__(
        self,
        cache: KVCacheManager,
        running: Iterable[Request],
        growth_tokens: int,
        step_tokens: int,
    ):
        self.cache = cache
        self.free = cache.block_pool.get_num_free_blocks()
        self.step_tokens = step_tokens
        self.headroom = sum(
            additional_blocks(
                cache,
                request,
                max(request.num_tokens, request.num_computed_tokens) + growth_tokens,
            )
            for request in running
            if request.num_computed_tokens >= request.num_prompt_tokens
        )

    def _need(self, request: Request) -> int:
        tokens = min(
            request.num_prompt_tokens, request.num_computed_tokens + self.step_tokens
        )
        return additional_blocks(self.cache, request, tokens)

    def can_admit(self, request: Request) -> bool:
        return self.free - self._need(request) >= self.headroom

    def __call__(self, request: Request) -> bool:
        need = self._need(request)
        if self.free - need < self.headroom:
            return False
        self.free -= need
        return True
