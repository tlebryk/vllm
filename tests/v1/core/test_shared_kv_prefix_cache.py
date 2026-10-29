# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prefix caching inside the SlackServe shared KV pool (CPU only)."""

import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.shared_kv import (
    SharedKVLayout,
    SharedKVManager,
    install_shared_block_pool,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK = 16
PAGE_BYTES = 2 * BLOCK * 2 * 64 * 2  # layer mode, 2 KV heads, head size 64


class Event:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


@pytest.fixture(autouse=True)
def _hash_and_flag(monkeypatch):
    init_none_hash(sha256)
    monkeypatch.setenv("SLACKSERVE_SHARED_PREFIX_CACHE", "1")


def make_manager(num_pages, layers=(1, 1)):
    layouts = [
        SharedKVLayout(mid, n, 2, 64, BLOCK, "float16") for mid, n in zip("ab", layers)
    ]
    return SharedKVManager(layouts, num_pages * PAGE_BYTES, "layer")


def make_kv(shared, model_id, caching):
    kv = KVCacheManager(
        KVCacheConfig(
            num_blocks=shared.logical_capacity(model_id),
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK,
                        num_kv_heads=2,
                        head_size=64,
                        dtype=torch.float16,
                    ),
                )
            ],
        ),
        max_model_len=4096,
        enable_caching=caching,
        hash_block_size=BLOCK,
    )
    install_shared_block_pool(kv, shared, model_id)
    return kv


def make_request(request_id, tokens):
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def admit(kv, request):
    computed, hit = kv.get_computed_blocks(request)
    blocks = kv.allocate_slots(request, request.num_tokens - hit, hit, computed)
    return None if blocks is None else hit


def prefix(seed, blocks, tail=5):
    return [seed * 1000 + i // BLOCK for i in range(blocks * BLOCK)] + [7] * tail


def check_invariants(shared):
    pages = [p for x in shared._leases.values() for p in x.pages]
    pending = [p for x, _, _ in shared._pending.values() for p in x.pages]
    everything = pages + pending + list(shared._free)
    assert len(everything) == len(set(everything)) == shared.num_pages - 1
    assert 0 not in everything
    for mid, pool in shared.cache_pools.items():
        assert set(pool.shared_leases) == {k[1] for k in shared._leases if k[0] == mid}
        queued = {b.block_id for b in pool.free_block_queue.get_all_free_blocks()}
        cached = set(pool.cached_free)
        # Free queue IDs are lease-less; cached free IDs hold leases.
        assert not queued & set(pool.shared_leases)
        assert cached <= set(pool.shared_leases)
        assert all(pool.blocks[i].ref_cnt == 0 for i in queued | cached)
        assert all(pool.blocks[i].block_hash is not None for i in cached)
    shared.snapshot()


def test_flag_unset_keeps_refusal(monkeypatch):
    monkeypatch.delenv("SLACKSERVE_SHARED_PREFIX_CACHE")
    shared = make_manager(12)
    with pytest.raises(ValueError, match="non-prefix-cached"):
        make_kv(shared, "a", True)
    kv = make_kv(shared, "b", False)
    assert type(kv.block_pool).__name__ == "SharedBlockPool"
    assert not shared.cache_pools
    assert "cached_pages" not in shared.snapshot()


def test_cached_blocks_keep_leases_and_hit():
    shared = make_manager(12)
    kv = make_kv(shared, "a", True)
    pool = kv.block_pool
    first = make_request("q1", prefix(1, 3))
    assert admit(kv, first) == 0
    leases = dict(pool.shared_leases)
    kv.free(first)
    # Three full blocks stay leased and cached; the partial block retires.
    assert len(pool.cached_free) == 3
    assert shared.snapshot()["owned_pages"]["a"] == 3
    assert shared.snapshot()["cached_pages"]["a"] == 3
    assert pool.get_num_free_blocks() == 3 + 8
    allocations = shared.stats["allocations"]
    second = make_request("q2", prefix(1, 3, tail=9))
    assert admit(kv, second) == 3 * BLOCK
    assert not pool.cached_free
    assert shared.stats["allocations"] == allocations + 1
    for block_id in kv.get_block_ids("q2")[0][:3]:
        assert pool.shared_leases[block_id] == leases[block_id]
    check_invariants(shared)


def test_eviction_reuses_lease_when_pages_run_out():
    shared = make_manager(8)  # seven usable pages
    kv = make_kv(shared, "a", True)
    pool = kv.block_pool
    old = make_request("old", prefix(1, 3, tail=0))
    admit(kv, old)
    kv.free(old)
    cached = dict(pool.shared_leases)
    assert set(cached) == set(pool.cached_free)
    assert pool.get_num_free_blocks() == 7
    # Six blocks: four free pages, then the two least recently used cached.
    new = make_request("new", prefix(2, 6, tail=0))
    allocations = shared.stats["allocations"]
    assert admit(kv, new) == 0
    assert shared.stats["allocations"] == allocations + 4
    reused = set(kv.get_block_ids("new")[0]) & set(cached)
    assert len(reused) == 2
    for block_id in reused:
        assert pool.shared_leases[block_id] == cached[block_id]
    assert len(pool.cached_free) == 1
    assert pool.get_num_free_blocks() == 1
    check_invariants(shared)
    # The surviving cached block is the most recently freed: old's first.
    assert pool.blocks[next(iter(pool.cached_free))].block_hash is not None
    assert admit(kv, make_request("more", prefix(3, 2, tail=0))) is None
    check_invariants(shared)


def test_other_model_reclaims_cached_pages_after_fence():
    shared = make_manager(9)  # eight usable pages
    verifier = make_kv(shared, "a", True)
    generator = make_kv(shared, "b", False)
    vpool, gpool = verifier.block_pool, generator.block_pool
    event = Event(ready=False)
    shared.record_use("a", "prefill", event)
    cached = make_request("v", prefix(1, 6, tail=0))
    admit(verifier, cached)
    verifier.free(cached)
    assert len(vpool.cached_free) == 6
    # The verifier's forward is still in flight: its pages are not free now.
    assert gpool.get_num_free_blocks() == 2
    request = make_request("g", prefix(9, 4, tail=0))
    assert admit(generator, request) is None
    assert generator.allocation_pending_blocks == 4
    # Reclamation evicted exactly the shortfall, LRU first, behind the fence.
    assert len(vpool.cached_free) == 4
    assert shared.snapshot()["pending_pages"] == 2
    assert shared.snapshot()["cached_pages"]["a"] == 4
    check_invariants(shared)
    event.ready = True
    assert gpool.get_num_free_blocks() == 8
    assert admit(generator, request) == 0
    # Collected pages covered the request; the remaining cache is intact.
    assert len(vpool.cached_free) == 4
    assert shared.snapshot()["owned_pages"] == {"a": 4, "b": 4}
    # With the fence passed, eviction is synchronous inside allocation.
    assert gpool.get_num_free_blocks() == 4
    assert admit(generator, make_request("g2", prefix(8, 2, tail=0))) == 0
    assert len(vpool.cached_free) == 2
    assert shared.snapshot()["owned_pages"] == {"a": 2, "b": 6}
    assert vpool.get_num_free_blocks() == 2
    # Tail blocks are evicted first; the two leading blocks still hit.
    _, hit = verifier.get_computed_blocks(make_request("v2", prefix(1, 6, tail=1)))
    assert hit == 2 * BLOCK
    check_invariants(shared)


def test_unequal_units_and_random_operations():
    rng = random.Random(0)
    shared = make_manager(64, layers=(2, 3))
    models = {"a": make_kv(shared, "a", True), "b": make_kv(shared, "b", True)}
    outstanding = []
    live = {mid: [] for mid in models}
    for step in range(600):
        mid = rng.choice("ab")
        kv = models[mid]
        if rng.random() < 0.5:
            event = Event(ready=False)
            shared.record_use(mid, "prefill", event)
            outstanding.append(event)
        for event in outstanding:
            event.ready = event.ready or rng.random() < 0.3
        if live[mid] and rng.random() < 0.45:
            kv.free(live[mid].pop(rng.randrange(len(live[mid]))))
        else:
            request = make_request(
                f"{mid}{step}", prefix(rng.randrange(4), rng.randrange(1, 6))
            )
            if admit(kv, request) is not None:
                live[mid].append(request)
        shared.collect()
        check_invariants(shared)
    for mid, requests in live.items():
        for request in requests:
            models[mid].free(request)
    for event in outstanding:
        event.ready = True
    shared.collect()
    check_invariants(shared)
    state = shared.snapshot()
    assert state["pending_pages"] == 0
    assert state["owned_pages"] == state["cached_pages"]
