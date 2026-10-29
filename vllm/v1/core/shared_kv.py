# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process shared KV capacity; CPU ownership is independent of GPU layout.

Only startup binding uses a ContextVar. Runners and schedulers retain explicit
manager references afterwards. There is no process-global allocator singleton.
"""

import os
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import get_ident
from typing import Any


@dataclass(frozen=True)
class SharedKVLayout:
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: str

    def __post_init__(self):
        if (
            not self.model_id
            or min(self.num_layers, self.num_kv_heads, self.head_size, self.block_size)
            <= 0
        ):
            raise ValueError("shared KV requires positive dimensions and model IDs")
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("shared KV v1/v1.1 supports FP16/BF16 only")


@dataclass(frozen=True)
class SharedKVLease:
    model_id: str
    block_id: int
    generation: int
    pages: tuple[int, ...]


_startup_manager: ContextVar[Any] = ContextVar("slackserve_shared_kv", default=None)


def shared_kv_for_config(config):
    options = config.additional_config.get("shared_kv")
    if options is None:
        return None
    manager = _startup_manager.get()
    if manager is None or manager.pool_id != options.get("pool_id"):
        raise RuntimeError("shared KV requires its explicit in-process startup scope")
    model_id = options["model_id"]
    if model_id not in manager.layouts:
        raise ValueError("model not registered with shared KV manager")
    return manager, model_id


class SharedKVManager:
    """One fixed backing allocation with event-delayed cross-model reuse.

    Logical IDs remain model-private vLLM IDs. Each lease maps one logical
    token block to layer/head pages (head mode) or layer pages (layer mode).
    A generation makes stale or duplicate releases an error. All mutations
    run on the scheduler/launch thread; completion threads never allocate.
    """

    def __init__(self, layouts, budget_bytes: int, mode: str, pool_id="dual_llm"):
        layouts = list(layouts)
        if len(layouts) != 2 or len({x.model_id for x in layouts}) != 2:
            raise ValueError("shared KV requires exactly two distinct startup models")
        if mode not in ("head", "layer") or budget_bytes <= 0:
            raise ValueError("shared KV requires head/layer mode and a positive budget")
        compatibility = {
            (x.num_kv_heads, x.head_size, x.block_size, x.dtype) for x in layouts
        }
        if len(compatibility) != 1:
            raise ValueError("v1/v1.1 requires matching heads, dimensions, dtype and N")
        self.layouts = {x.model_id: x for x in layouts}
        self.mode, self.pool_id = mode, pool_id
        self.budget_bytes = budget_bytes
        first = layouts[0]
        self.storage_heads = first.num_kv_heads if mode == "layer" else 1
        self.page_bytes = (
            2 * first.block_size * self.storage_heads * first.head_size * 2
        )
        self.num_pages = budget_bytes // self.page_bytes
        self.units = {
            x.model_id: x.num_layers * (x.num_kv_heads if mode == "head" else 1)
            for x in layouts
        }
        # Page zero is a shared dummy/null page, never leased to requests.
        if self.num_pages - 1 < max(self.units.values()):
            raise ValueError("shared KV budget cannot hold one block of each layout")
        self._free = deque(range(1, self.num_pages))
        self._leases: dict[tuple[str, int], SharedKVLease] = {}
        self._pending: dict[tuple[str, int], tuple[SharedKVLease, tuple, Any]] = {}
        self._generation = 0
        self._owned_pages = {x.model_id: 0 for x in layouts}
        self._pending_pages = 0
        self._pending_blocks = {x.model_id: 0 for x in layouts}
        self._events: dict[str, dict[str, Any]] = {x.model_id: {} for x in layouts}
        self.allocation_listeners = {}
        # Opt-in prefix caching: model_id -> pool holding leased free blocks.
        self.cache_pools = {}
        self._thread_id = get_ident()
        self.storage = None
        self.device = None
        self.failure = None
        self.stats = {"allocations": 0, "releases": 0, "allocation_failures": 0}

    def _check_thread(self):
        if get_ident() != self._thread_id:
            raise RuntimeError("shared KV mutations must run on the launch thread")

    @contextmanager
    def startup_scope(self):
        self._check_thread()
        token = _startup_manager.set(self)
        try:
            yield self
        finally:
            _startup_manager.reset(token)

    def logical_capacity(self, model_id):
        # Includes logical null block zero; usable capacity is capacity - 1.
        return (self.num_pages - 1) // self.units[model_id] + 1

    def collect(self):
        self._check_thread()
        if self.failure is not None:
            return
        for key, (lease, events, callback) in list(self._pending.items()):
            if all(event.query() for event in events):
                self._free.extend(lease.pages)
                del self._pending[key]
                self._pending_pages -= len(lease.pages)
                self._pending_blocks[lease.model_id] -= 1
                self.stats["releases"] += 1
                if callback is not None:
                    callback()

    def free_blocks(self, model_id):
        self.collect()
        return len(self._free) // self.units[model_id]

    def can_allocate_after_reclamation(self, model_id, count, free_logical):
        """Would existing releases suffice, without evicting another request?

        Other models' releases can supply physical pages, but only this model's
        releases return its logical IDs. Neither is reusable before its fence.
        """
        if self.failure is not None:
            return False
        physical = (len(self._free) + self._pending_pages) // self.units[model_id]
        logical = free_logical + self._pending_blocks[model_id]
        return count <= min(physical, logical)

    def allocate(self, model_id, block_ids):
        self._check_thread()
        self.check_healthy()
        self.collect()
        block_ids = tuple(block_ids)
        keys = [(model_id, block_id) for block_id in block_ids]
        if len(set(keys)) != len(keys) or any(
            block_id <= 0
            or block_id >= self.logical_capacity(model_id)
            or key in self._leases
            or key in self._pending
            for key, block_id in zip(keys, block_ids)
        ):
            raise ValueError("duplicate, live, pending or invalid shared KV logical ID")
        count = len(block_ids) * self.units[model_id]
        if count > len(self._free):
            self.stats["allocation_failures"] += 1
            raise ValueError("insufficient shared KV pages")
        # Validate the entire batch before taking any physical pages.
        result = []
        for key, block_id in zip(keys, block_ids):
            self._generation += 1
            lease = SharedKVLease(
                model_id,
                block_id,
                self._generation,
                tuple(self._free.popleft() for _ in range(self.units[model_id])),
            )
            self._leases[key] = lease
            result.append(lease)
        listener = self.allocation_listeners.get(model_id)
        try:
            if listener is not None:
                for lease in result:
                    listener(lease)
        except Exception:
            for lease in reversed(result):
                del self._leases[(model_id, lease.block_id)]
                self._free.extendleft(reversed(lease.pages))
            self.stats["allocation_failures"] += 1
            raise
        self.stats["allocations"] += len(result)
        self._owned_pages[model_id] += count
        return result

    def retire(self, lease, callback=None, *, collect=True):
        self._check_thread()
        self.check_healthy()
        key = (lease.model_id, lease.block_id)
        if self._leases.get(key) != lease:
            raise ValueError("stale or duplicate shared KV release")
        del self._leases[key]
        # Fresh immutable events: never snapshot a re-recorded lane event.
        events = tuple(self._events[lease.model_id].values())
        self._pending[key] = (lease, events, callback)
        self._owned_pages[lease.model_id] -= len(lease.pages)
        self._pending_pages += len(lease.pages)
        self._pending_blocks[lease.model_id] += 1
        if collect:
            self.collect()

    def reclaimable_pages(self, model_id, *, fenced=True):
        """Pages other models' free cached blocks would release on eviction.

        With ``fenced``, only models whose last recorded uses have completed,
        i.e. pages that eviction makes free immediately.
        """
        return sum(
            len(pool.cached_free) * self.units[other]
            for other, pool in self.cache_pools.items()
            if other != model_id
            and pool.cached_free
            and (not fenced or all(e.query() for e in self._events[other].values()))
        )

    def reclaim(self, model_id, pages):
        """Evict other models' LRU free cached blocks covering ``pages``."""
        self._check_thread()
        for other, pool in self.cache_pools.items():
            if other != model_id and pages > 0:
                count = -(-pages // self.units[other])
                pages -= pool.evict_cached(count) * self.units[other]
        self.collect()

    def record_use(self, model_id, lane, event):
        self._check_thread()
        self._events[model_id][lane] = event

    def poison(self, reason):
        """A partial failed GPU launch cannot safely reuse this serving session."""
        self._check_thread()
        self.failure = str(reason)

    def check_healthy(self):
        if self.failure is not None:
            raise RuntimeError(
                "shared KV session failed; restart before reusing storage"
            )

    def leases(self, model_id):
        return [x for x in self._leases.values() if x.model_id == model_id]

    def snapshot(self):
        self.collect()
        owned = dict(self._owned_pages)
        pending = self._pending_pages
        assert sum(owned.values()) + pending + len(self._free) + 1 == self.num_pages
        cached = {}
        if self.cache_pools:
            # Owned pages of free cached blocks; evictable on demand.
            cached = {
                "cached_pages": {
                    mid: len(pool.cached_free) * self.units[mid]
                    for mid, pool in self.cache_pools.items()
                }
            }
        return {
            "mode": self.mode,
            "budget_bytes": self.budget_bytes,
            "actual_storage_bytes": self.num_pages * self.page_bytes,
            "unallocated_budget_bytes": self.budget_bytes % self.page_bytes,
            "page_bytes": self.page_bytes,
            "num_pages": self.num_pages,
            "null_pages": 1,
            "free_pages": len(self._free),
            "pending_pages": pending,
            "owned_pages": owned,
            "failure": self.failure,
            **self.stats,
            **cached,
        }

    def validate_config(self, model_id, kv_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        layout = self.layouts[model_id]
        if len(kv_config.kv_cache_groups) != 1:
            raise ValueError("shared KV requires one uniform full-attention group")
        group = kv_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or (
                spec.attention_chunk_size is not None
                or spec.page_size_padded is not None
            )
        ):
            raise ValueError("shared KV requires unpadded full attention")
        actual = SharedKVLayout(
            model_id,
            len(group.layer_names),
            spec.num_kv_heads,
            spec.head_size,
            spec.block_size,
            str(spec.dtype).removeprefix("torch."),
        )
        if actual != layout or spec.head_size_v != spec.head_size:
            raise ValueError("runtime KV specification differs from startup layout")
        return group, spec

    def bind_storage(self, model_id, kv_config, device):
        import torch

        group, spec = self.validate_config(model_id, kv_config)
        layout = self.layouts[model_id]
        if self.storage is None:
            self.device = torch.device(device)
            self.storage = torch.zeros(
                (
                    2,
                    self.num_pages,
                    layout.block_size,
                    self.storage_heads,
                    layout.head_size,
                ),
                dtype=spec.dtype,
                device=device,
            )
        elif self.device != torch.device(device):
            raise ValueError("shared KV models must use the same CUDA device")
        return {name: self.storage for name in group.layer_names}


def install_shared_block_pool(kv_manager, shared_manager, model_id):
    """Replace an empty model-private CPU pool, preserving vLLM request logic."""
    from vllm.v1.core.block_pool import BlockPool

    prefix_cache = os.environ.get("SLACKSERVE_SHARED_PREFIX_CACHE") == "1"

    class SharedBlockPool(BlockPool):
        def __init__(self, old):
            if (old.enable_caching and not prefix_cache) or any(
                b.ref_cnt for b in old.blocks
            ):
                raise ValueError(
                    "shared KV binding requires empty, non-prefix-cached pool"
                )
            super().__init__(
                old.num_gpu_blocks,
                old.enable_caching,
                old.hash_block_size,
                old.enable_kv_cache_events,
                old.metrics_collector,
            )
            self.shared_manager = shared_manager
            self.shared_model_id = model_id
            self.shared_leases = {}

        def get_num_free_blocks(self):
            if shared_manager.failure is not None:
                return 0
            physical = shared_manager.free_blocks(model_id)
            # Collection can return delayed logical IDs to this local queue.
            return min(super().get_num_free_blocks(), physical)

        def get_new_blocks(self, num_blocks):
            blocks = super().get_new_blocks(num_blocks)
            try:
                leases = shared_manager.allocate(model_id, [b.block_id for b in blocks])
            except Exception:
                super().free_blocks(blocks)
                raise
            self.shared_leases.update({x.block_id: x for x in leases})
            return blocks

        def can_allocate_after_reclamation(self, num_blocks):
            # Collect before reading the local free-ID queue: callbacks may
            # have returned IDs while querying completion events.
            shared_manager.collect()
            return shared_manager.can_allocate_after_reclamation(
                model_id, num_blocks, super().get_num_free_blocks()
            )

        def free_blocks(self, ordered_blocks):
            shared_manager.check_healthy()
            for block in ordered_blocks:
                if block.is_null:
                    continue
                if block.ref_cnt <= 0:
                    raise ValueError("duplicate shared KV block release")
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    lease = self.shared_leases.pop(block.block_id)
                    shared_manager.retire(
                        lease, lambda b=block: self.free_block_queue.append_n([b])
                    )

    class PrefixCachedSharedBlockPool(SharedBlockPool):
        """Free cached blocks keep their leases until evicted.

        ``cached_free`` holds them in LRU order instead of the free queue, which
        holds only lease-less logical IDs. Allocation takes free pages first,
        then this model's LRU cached blocks (lease reused), then evicts other
        models' cached blocks. Every pool is registered so either model can
        reclaim from the other; a pool without caching never caches.
        """

        def __init__(self, old):
            super().__init__(old)
            self.cached_free = {}
            shared_manager.cache_pools[model_id] = self

        def get_num_free_blocks(self):
            if shared_manager.failure is not None:
                return 0
            shared_manager.collect()
            pages = len(shared_manager._free) + shared_manager.reclaimable_pages(
                model_id
            )
            return len(self.cached_free) + min(
                self.free_block_queue.num_free_blocks,
                pages // shared_manager.units[model_id],
            )

        def get_new_blocks(self, num_blocks):
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            units = shared_manager.units[model_id]
            fresh = min(
                num_blocks,
                self.free_block_queue.num_free_blocks,
                len(shared_manager._free) // units,
            )
            reuse = min(num_blocks - fresh, len(self.cached_free))
            fresh = num_blocks - reuse
            if fresh * units > len(shared_manager._free):
                shared_manager.reclaim(
                    model_id, fresh * units - len(shared_manager._free)
                )
            blocks = self.free_block_queue.popleft_n(fresh)
            try:
                leases = shared_manager.allocate(model_id, [b.block_id for b in blocks])
            except Exception:
                self.free_block_queue.append_n(blocks)
                raise
            self.shared_leases.update({x.block_id: x for x in leases})
            for _ in range(reuse):
                blocks.append(self.cached_free.pop(next(iter(self.cached_free))))
            for block in blocks:
                if self.enable_caching:
                    self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
            return blocks

        def touch(self, blocks):
            for block in blocks:
                if block.ref_cnt == 0 and not block.is_null:
                    del self.cached_free[block.block_id]
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_accessed(block)

        def free_blocks(self, ordered_blocks):
            shared_manager.check_healthy()
            for block in ordered_blocks:
                if block.is_null:
                    continue
                if block.ref_cnt <= 0:
                    raise ValueError("duplicate shared KV block release")
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    if block.block_hash is not None:
                        self.cached_free[block.block_id] = block
                        continue
                    lease = self.shared_leases.pop(block.block_id)
                    shared_manager.retire(
                        lease, lambda b=block: self.free_block_queue.append_n([b])
                    )

        def evict_cached(self, count):
            """Drop LRU free cached blocks; IDs return after the page fence."""
            count = min(count, len(self.cached_free))
            for _ in range(count):
                block = self.cached_free.pop(next(iter(self.cached_free)))
                self._maybe_evict_cached_block(block)
                shared_manager.retire(
                    self.shared_leases.pop(block.block_id),
                    lambda b=block: self.free_block_queue.append_n([b]),
                    collect=False,
                )
            return count

        def can_allocate_after_reclamation(self, num_blocks):
            shared_manager.collect()
            if shared_manager.failure is not None:
                return False
            units = shared_manager.units[model_id]
            need = num_blocks - len(self.cached_free)
            available = len(shared_manager._free) + shared_manager._pending_pages
            pages = available + shared_manager.reclaimable_pages(model_id, fenced=False)
            logical = (
                self.free_block_queue.num_free_blocks
                + shared_manager._pending_blocks[model_id]
            )
            if need > min(pages // units, logical):
                return False
            if need * units > available:
                # Pages become free once the other model's fences complete.
                shared_manager.reclaim(model_id, need * units - available)
            return True

    old = kv_manager.block_pool
    if old.num_gpu_blocks != shared_manager.logical_capacity(model_id):
        raise ValueError("shared KV scheduler capacity mismatch")
    pool = (PrefixCachedSharedBlockPool if prefix_cache else SharedBlockPool)(old)
    kv_manager.block_pool = kv_manager.coordinator.block_pool = pool
    for manager in kv_manager.coordinator.single_type_managers:
        manager.block_pool = pool
