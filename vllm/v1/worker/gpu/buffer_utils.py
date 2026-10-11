# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)


# UVA staging buffers are recycled round-robin; without a guard, a host
# rewrite can race GPU kernels still reading the buffer from an earlier
# use. Slack Serve's multi-lane controller stretches the enqueue-to-
# execution window enough to hit this. A caller that has enqueued every
# consumer of the staged buffers calls its runner's registry.fence() once; each pool
# written since the previous fence then records a per-buffer event on the
# consuming stream, and the pool host-synchronizes that event before the
# buffer is rewritten. Code that never fences (the ordinary single-lane
# runner) behaves exactly as before.
class UvaFenceRegistry:
    """Pending UVA writes owned by one runner, never by a process or thread.

    The runner must serialize staging/fencing and fence after the consuming
    kernels have been enqueued on the supplied stream. Ownership isolation
    does not replace ordering between that runner's execution lanes.
    """

    def __init__(self) -> None:
        self._dirty_pools: list[UvaBufferPool] = []
        self.isolate_lanes = False
        self.lane: str | None = None

    @contextmanager
    def use_lane(self, lane: str):
        """Select host staging slots on the runner's serialized owner thread.

        Decode retains the original slots, including capture/warmup addresses.
        Prefill lanes lazily allocate private slots with the same reuse guards.
        """
        previous = self.lane
        self.lane = (
            lane if self.isolate_lanes and lane not in ("decode", "default") else None
        )
        try:
            yield
        finally:
            self.lane = previous

    def fence(self, stream: torch.cuda.Stream | None = None) -> None:
        """Record consumption of only this runner's pending buffers."""
        pools, self._dirty_pools = self._dirty_pools, []
        for pool in pools:
            pool._record_pending(stream)


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
        *,
        fence_registry: UvaFenceRegistry | None = None,
    ):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency
        self.fence_registry = fence_registry

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0
        # Optional consumption guard (see UvaFenceRegistry.fence).
        self._events: list[torch.cuda.Event | None] = [None] * max_concurrency
        self._event_pending = [False] * max_concurrency
        self._dirty: set[int] = set()
        self._lane_slots: dict[str | None, list[int]] = {
            None: list(range(max_concurrency))
        }
        self._lane_cursors: dict[str | None, int] = {None: 0}
        self._consumer_events: dict[int, dict[int, torch.cuda.Event]] = {}
        self._consumer_pending: dict[int, set[int]] = {}

    def _mark_used(self, index: int) -> None:
        if self.fence_registry is not None:
            if not self._dirty:
                self.fence_registry._dirty_pools.append(self)
            self._dirty.add(index)

    def mark_read(self) -> None:
        """Fence fresh consumers even when an immutable UVA view was not copied."""
        if self.fence_registry is not None and self.fence_registry.isolate_lanes:
            self._mark_used(self._curr)

    def _record_pending(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        for index in self._dirty:
            if self.fence_registry is not None and self.fence_registry.isolate_lanes:
                # One snapshot can be read by its owning lane and another
                # lane without being recopied. Retain every stream's fence;
                # a fast later reader must not replace a slow earlier one.
                stream_key = stream.cuda_stream
                events = self._consumer_events.setdefault(index, {})
                event = events.get(stream_key)
                if event is None:
                    event = torch.cuda.Event()
                    events[stream_key] = event
                event.record(stream)
                self._consumer_pending.setdefault(index, set()).add(stream_key)
                continue
            event = self._events[index]
            if event is None:
                event = torch.cuda.Event()
                self._events[index] = event
            event.record(stream)
            self._event_pending[index] = True
        self._dirty.clear()

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        lane = self.fence_registry.lane if self.fence_registry is not None else None
        if lane not in self._lane_slots:
            start = len(self._uva_bufs)
            self._uva_bufs.extend(
                UvaBuffer(self.size, self.dtype) for _ in range(self.max_concurrency)
            )
            self._events.extend([None] * self.max_concurrency)
            self._event_pending.extend([False] * self.max_concurrency)
            self._lane_slots[lane] = list(range(start, start + self.max_concurrency))
            self._lane_cursors[lane] = 0
        cursor = (self._lane_cursors[lane] + 1) % self.max_concurrency
        self._lane_cursors[lane] = cursor
        self._curr = self._lane_slots[lane][cursor]
        for stream_key in self._consumer_pending.get(self._curr, ()):
            self._consumer_events[self._curr][stream_key].synchronize()
        if self._curr in self._consumer_pending:
            self._consumer_pending[self._curr].clear()
        if self._event_pending[self._curr]:
            # A fenced consumer of this buffer's previous contents may still
            # be in flight; wait before the host rewrites the memory.
            event = self._events[self._curr]
            assert event is not None
            event.synchronize()
            self._event_pending[self._curr] = False
        self._mark_used(self._curr)
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
        *,
        fence_registry: UvaFenceRegistry | None = None,
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(
            size, dtype, max_concurrency, fence_registry=fence_registry
        )
        self.gpu = self.pool.copy_to_uva(self.np)

    @property
    def gpu(self) -> torch.Tensor:
        # A future ticket can consume the last snapshot without copying it
        # again (prompt lengths, for example). Extend its consumption fence.
        self.pool.mark_read()
        return self._gpu

    @gpu.setter
    def gpu(self, value: torch.Tensor) -> None:
        self._gpu = value

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
        *,
        fence_registry: UvaFenceRegistry | None = None,
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(
            UvaBufferPool,
            max_concurrency=max_concurrency,
            fence_registry=fence_registry,
        )

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, self.dtype, self.device, pin_memory=True
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )
