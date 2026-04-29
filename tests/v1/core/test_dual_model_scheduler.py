# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.pooling_params import PoolingParams
from vllm.platforms import current_platform
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.dual_model_helpers import DualModelConfig

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_dual_model_embed_requests_allocate_and_free_kv(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler()
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    decode_request = create_requests(num_requests=1, num_tokens=8)[0]
    embed_request = Request(
        request_id="embed-0",
        prompt_token_ids=[7] * 8,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )

    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    allocate_mock = Mock(side_effect=original_allocate_slots)
    scheduler.kv_cache_manager.allocate_slots = allocate_mock

    original_free = scheduler.kv_cache_manager.free
    free_mock = Mock(side_effect=original_free)
    scheduler.kv_cache_manager.free = free_mock

    scheduler.add_request(decode_request)
    scheduler.add_request(embed_request)

    output = scheduler.schedule()
    new_req_by_id = {req.req_id: req for req in output.scheduled_new_reqs}

    assert allocate_mock.call_count == 2
    assert {
        call.args[0].request_id for call in allocate_mock.call_args_list
    } == {decode_request.request_id, embed_request.request_id}
    assert new_req_by_id[embed_request.request_id].block_ids != ([],)
    assert output.num_scheduled_tokens[embed_request.request_id] == 8

    scheduler.finish_requests(embed_request.request_id, RequestStatus.FINISHED_ABORTED)
    assert free_mock.call_count == 1

    scheduler.finish_requests(decode_request.request_id, RequestStatus.FINISHED_ABORTED)
    assert free_mock.call_count == 2


def test_dual_model_scheduler_output_records_schedule_time_counts(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=32)
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    decode_request = create_requests(num_requests=1, num_tokens=8)[0]
    embed_request = Request(
        request_id="embed-0",
        prompt_token_ids=[7] * 8,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )
    scheduler.add_request(decode_request)
    scheduler.add_request(embed_request)

    output = scheduler.schedule()

    assert output.schedule_start_running_reqs == 0
    assert output.schedule_start_waiting_reqs == 2
    assert output.schedule_start_running_decode_reqs == 0
    assert output.schedule_start_running_embed_reqs == 0
    assert output.schedule_start_waiting_decode_reqs == 1
    assert output.schedule_start_waiting_embed_reqs == 1
    assert output.schedule_end_running_reqs == 2
    assert output.schedule_end_waiting_reqs == 0
    assert output.schedule_end_running_decode_reqs == 1
    assert output.schedule_end_running_embed_reqs == 1
    assert output.schedule_end_waiting_decode_reqs == 0
    assert output.schedule_end_waiting_embed_reqs == 0
    assert output.schedule_max_tokens == 32
    assert output.schedule_max_running_reqs == 4
    assert output.schedule_running_slots_remaining == 2
    assert output.schedule_waiting_stop_reason == "no_waiting_requests"


def test_dual_model_without_embed_gate_does_not_scan_model_counts(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=32)
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    decode_request = create_requests(num_requests=1, num_tokens=8)[0]
    embed_request = Request(
        request_id="embed-0",
        prompt_token_ids=[7] * 8,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )
    scheduler.add_request(decode_request)
    scheduler.add_request(embed_request)

    scheduler._count_running_by_model = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected running model-count scan")
    )
    scheduler._count_queue_by_model = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected waiting model-count scan")
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[decode_request.request_id] == 8
    assert output.num_scheduled_tokens[embed_request.request_id] == 8


def test_dual_model_embed_gate_uses_cached_model_counts(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=32)
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model",
        embed_release_when_decode_waiting_drained=True,
    )

    decode_requests = create_requests(num_requests=2, num_tokens=8)
    embed_request = Request(
        request_id="embed-0",
        prompt_token_ids=[7] * 8,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )
    scheduler.add_request(decode_requests[0])
    scheduler.add_request(embed_request)
    scheduler.add_request(decode_requests[1])

    scheduler._count_running_by_model = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected running model-count scan")
    )
    scheduler._count_queue_by_model = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected waiting model-count scan")
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        decode_requests[0].request_id: 8,
        decode_requests[1].request_id: 8,
    }
