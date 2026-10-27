# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate asynchronous rank responses without imposing completion order."""

from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class PendingRpc:
    ranks: tuple[int, ...]
    aggregate: Callable
    future: Future = field(default_factory=Future)
    results: dict[int, Any] = field(default_factory=dict)


class RpcResponses:
    def __init__(self):
        self._lock = Lock()
        self._next = 0
        self._pending: dict[int, PendingRpc] = {}

    def register(self, ranks: Sequence[int], aggregate: Callable) -> tuple[int, Future]:
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("RPC must expect a nonempty set of unique ranks")
        with self._lock:
            seq = self._next
            self._next += 1
            pending = PendingRpc(tuple(ranks), aggregate)
            self._pending[seq] = pending
        return seq, pending.future

    def discard(self, seq: int) -> None:
        with self._lock:
            self._pending.pop(seq, None)

    def receive(
        self, seq: int, rank: int, value: Any = None, error: Exception | None = None
    ):
        with self._lock:
            pending = self._pending.get(seq)
            if pending is None:
                return  # Late response to a timed-out or failed RPC.
            if rank not in pending.ranks or rank in pending.results:
                error = RuntimeError(
                    f"Unexpected or duplicate rank {rank} for RPC {seq}"
                )
            if error is None:
                pending.results[rank] = value
                if len(pending.results) != len(pending.ranks):
                    return
            del self._pending[seq]
        if error is None:
            try:
                value = pending.aggregate([pending.results[r] for r in pending.ranks])
            except Exception as exc:
                error = exc
        with suppress(InvalidStateError):
            if error is None:
                pending.future.set_result(value)
            else:
                pending.future.set_exception(error)

    def fail_all(self, error: Exception):
        with self._lock:
            pending, self._pending = self._pending, {}
        for call in pending.values():
            with suppress(InvalidStateError):
                call.future.set_exception(error)
