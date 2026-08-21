# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler


def _scheduler(*request_ids: str) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._hb_inflight_req_ids = set()
    scheduler._hb_async_decode_refs = defaultdict(int)
    scheduler.running = [
        SimpleNamespace(request_id=request_id) for request_id in request_ids
    ]
    return scheduler


def test_async_decode_refs_are_counted_per_ticket():
    scheduler = _scheduler("request")

    scheduler.mark_hb_async_decode({"request"})
    scheduler.mark_hb_async_decode({"request"})
    assert scheduler.is_hb_inflight("request")

    scheduler.release_hb_async_decode({"request"})
    assert scheduler.hb_has_async_decode_ref("request")
    scheduler.release_hb_async_decode({"request"})
    assert not scheduler.is_hb_inflight("request")

    with pytest.raises(RuntimeError, match="unreferenced"):
        scheduler.release_hb_async_decode({"request"})


def test_preemption_excludes_prefill_and_decode_ticket_owners():
    scheduler = _scheduler("prefill", "decode", "safe")
    scheduler._hb_inflight_req_ids.add("prefill")
    scheduler.mark_hb_async_decode({"decode"})

    assert [
        request.request_id for request in scheduler.hb_safe_preemption_victims()
    ] == ["safe"]

    scheduler.mark_hb_async_decode({"safe"})
    assert scheduler.hb_safe_preemption_victims() == []
