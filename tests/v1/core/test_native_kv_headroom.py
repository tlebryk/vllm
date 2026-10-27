# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native admission must count every physical cache group, including state."""

from math import lcm

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.kv_headroom import (
    CacheHeadroom,
    additional_blocks,
    calibration_warmer_shape,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def cache(specs, blocks):
    return KVCacheManager(
        KVCacheConfig(
            blocks,
            [],
            [KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)],
        ),
        max_model_len=1024,
        scheduler_block_size=lcm(*(spec.block_size for spec in specs)),
        hash_block_size=16,
        enable_caching=False,
    )


def request(name, tokens):
    return Request(name, [1] * tokens, SamplingParams(max_tokens=64), None)


def attention():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def test_two_attention_groups_cannot_be_admitted_as_one():
    manager = cache([attention(), attention()], 3)  # null block plus two free
    budget = CacheHeadroom(manager, [], 16, 16)
    first, second = request("first", 16), request("second", 16)
    assert budget(first)
    assert not budget.can_admit(second)
    assert manager.allocate_slots(first, 16) is not None
    assert manager.allocate_slots(second, 16) is None


def test_mamba_state_allocation_is_not_linear_in_prompt_length():
    state = MambaSpec(
        # Non-caching recurrent state uses a request-sized logical block.
        block_size=1024,
        shapes=((32,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    manager = cache([attention(), state], 64)
    req = request("hybrid", 64)
    predicted = additional_blocks(manager, req, 64)
    free = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(req, 64) is not None
    assert predicted == free - manager.block_pool.get_num_free_blocks() == 5


def test_decode_growth_counts_groups_and_existing_allocations():
    manager = cache([attention(), attention()], 10)
    resident = request("resident", 16)
    assert manager.allocate_slots(resident, 16) is not None
    resident.num_computed_tokens = 16
    budget = CacheHeadroom(manager, [resident], 16, 16)
    assert budget.headroom == 2
    assert budget.can_admit(request("new", 16))
    predicted = additional_blocks(manager, resident, 32)
    free = manager.block_pool.get_num_free_blocks()
    assert manager.allocate_slots(resident, 16) is not None
    assert free - manager.block_pool.get_num_free_blocks() == predicted == 2


def test_calibration_warmers_fit_all_cache_groups_with_probe_headroom():
    manager = cache([attention(), attention()], 400)
    free = manager.block_pool.get_num_free_blocks()
    prompt, output = calibration_warmer_shape(manager, 4, 128, 8192, 4096)
    assert manager.block_pool.get_num_free_blocks() == free
    for i in range(4):
        warmer = request(f"warmer{i}", prompt)
        assert manager.allocate_slots(warmer, prompt + output) is not None
    used = free - manager.block_pool.get_num_free_blocks()
    assert used <= int(0.6 * free)
    assert manager.allocate_slots(request("probe", 128), 129) is not None
