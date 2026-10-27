# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A short decode must publish while an older prefill remains unfinished."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# This component is pure Python; loading it directly avoids CUDA initialization.
path = Path(__file__).resolve().parents[3] / "vllm/v1/executor/rpc_responses.py"
spec = importlib.util.spec_from_file_location("rpc_responses_under_test", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
RpcResponses = module.RpcResponses


def test_decode_publishes_before_older_prefill_with_rank_order_preserved():
    responses = RpcResponses()
    prefill, p = responses.register([0, 1], tuple)
    decode, d = responses.register([0, 1], tuple)
    responses.receive(prefill, 0, "prefill-0")
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(responses.receive, decode, 1, "decode-1")
        b = pool.submit(responses.receive, decode, 0, "decode-0")
        a.result(timeout=1)
        b.result(timeout=1)
    assert d.result(timeout=1) == ("decode-0", "decode-1")
    assert not p.done()
    responses.receive(prefill, 1, "prefill-1")
    assert p.result(timeout=1) == ("prefill-0", "prefill-1")


def test_timed_out_response_cannot_be_matched_to_a_later_ticket():
    responses = RpcResponses()
    old, _ = responses.register([0], tuple)
    responses.discard(old)
    new, future = responses.register([0], tuple)
    assert new != old
    responses.receive(old, 0, "late")
    assert not future.done()
    responses.receive(new, 0, "current")
    assert future.result(timeout=1) == ("current",)


def test_worker_failure_and_queue_shutdown_release_waiters():
    responses = RpcResponses()
    failed, a = responses.register([0, 1], tuple)
    _, b = responses.register([0], tuple)
    responses.receive(failed, 1, error=RuntimeError("worker error"))
    with pytest.raises(RuntimeError, match="worker error"):
        a.result(timeout=1)
    assert not b.done()
    responses.fail_all(RuntimeError("queue closed"))
    with pytest.raises(RuntimeError, match="queue closed"):
        b.result(timeout=1)


def test_duplicate_rank_does_not_prematurely_satisfy_collective():
    responses = RpcResponses()
    seq, future = responses.register([0, 1], tuple)
    responses.receive(seq, 0, "first")
    responses.receive(seq, 0, "duplicate")
    with pytest.raises(RuntimeError, match="duplicate"):
        future.result(timeout=1)
