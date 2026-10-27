# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predict native admission demand with the real hybrid-cache allocator."""

from collections.abc import Iterable

from vllm.sampling_params import SamplingParams
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


def calibration_warmer_shape(
    cache: KVCacheManager,
    count: int,
    probe_tokens: int,
    prompt_limit: int,
    output_limit: int,
) -> tuple[int, int]:
    """Fit warmers plus a probe using physical group demand, without allocation.

    Keep the existing 60% warmer reservation and 2:1 prompt/output preference.
    Recurrent state consumes blocks per request, not per scheduler token block.
    """
    if probe_tokens + 1 > cache.max_model_len:
        raise ValueError("Calibration probe exceeds model context length")

    def demand(prompt_tokens: int, output_tokens: int) -> int:
        request = Request(
            "__native_calibration_capacity_probe__",
            [1] * prompt_tokens,
            SamplingParams(max_tokens=output_tokens),
            None,
        )
        if (
            request.request_id
            in cache.coordinator.single_type_managers[0].req_to_blocks
        ):
            raise RuntimeError("Calibration capacity probe request ID is already live")
        return additional_blocks(cache, request, prompt_tokens + output_tokens)

    free = cache.block_pool.get_num_free_blocks()
    budget = min(int(0.6 * free), free - demand(probe_tokens, 1))

    def shape(total: int) -> tuple[int, int]:
        prompt = min(prompt_limit, total * 2 // 3)
        return prompt, min(output_limit, total - prompt)

    low, high = 384, min(prompt_limit + output_limit, cache.max_model_len)
    if count <= 0 or high < low or count * demand(*shape(low)) > budget:
        raise ValueError("Calibration warmers and probe exceed physical cache capacity")
    while low < high:
        middle = (low + high + 1) // 2
        if count * demand(*shape(middle)) <= budget:
            low = middle
        else:
            high = middle - 1
    return shape(low)


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
