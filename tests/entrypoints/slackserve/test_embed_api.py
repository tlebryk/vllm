# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for Slack Serve's PD-E endpoint and failure boundary."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm.config.compilation import CompilationConfig
from vllm.entrypoints.slackserve.config import configure_dense_sidecar
from vllm.entrypoints.slackserve.embed_api_router import (
    SlackServeEmbeddingRequest,
    create_embedding,
)
from vllm.v1.engine.hb_embed_sidecar import HbEmbedSidecar


class _Core:
    async def call_utility_async(self, method: str, request_id: str, text: str):
        assert method == "submit_hb_embedding"
        return {
            "request_id": request_id,
            "data": [0.25, 0.75],
            "prompt_tokens": len(text.split()),
        }


def test_embedding_routes_to_sidecar(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "embed")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(engine_client=SimpleNamespace(engine_core=_Core()))
        )
    )

    response = asyncio.run(
        create_embedding.__wrapped__(
            SlackServeEmbeddingRequest(model="embed", input=["hello", "two words"]),
            raw_request,
        )
    )

    assert [item.embedding for item in response.data] == [
        [0.25, 0.75],
        [0.25, 0.75],
    ]
    assert response.usage.prompt_tokens == 3


def test_embedding_rejects_unknown_model(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "embed")
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    response = asyncio.run(
        create_embedding.__wrapped__(
            SlackServeEmbeddingRequest(model="wrong", input="hello"), raw_request
        )
    )

    assert response.status_code == 404
    assert "does not exist" in json.loads(response.body)["error"]["message"]


@pytest.fixture
def clean_hb_environment():
    original = {k: v for k, v in os.environ.items() if k.startswith("HB_")}
    for name in original:
        os.environ.pop(name)
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith("HB_"):
                os.environ.pop(name)
        os.environ.update(original)


def test_hybrid_defaults_are_reproducible(clean_hb_environment):
    args = SimpleNamespace(
        slackserve_dense_model="embedding-model",
        slackserve_dense_served_model_name="embed",
        slackserve_dense_gpu_memory_utilization=0.18,
        slackserve_dense_max_model_len=11264,
        slackserve_dense_max_num_batched_tokens=8192,
        slackserve_dense_max_num_seqs=192,
        worker_cls="auto",
        scheduler_cls=None,
        async_scheduling=None,
        gpu_memory_utilization=0.9,
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        enable_prefix_caching=None,
        compilation_config=CompilationConfig(),
    )

    configure_dense_sidecar(args)

    assert args.gpu_memory_utilization == 0.65
    assert args.max_model_len == 11264
    assert args.max_num_batched_tokens == 65536
    assert args.max_num_seqs == 192
    assert args.enable_prefix_caching is False
    assert args.worker_cls.endswith("SlackServeGPUWorker")
    assert args.compilation_config.cudagraph_mode.name == "FULL"
    assert args.compilation_config.custom_ops == ["none"]
    assert os.environ["HB_P2_EMBED_GMU"] == "0.18"
    assert os.environ["HB_P2_PREFILL_KV_WATERMARK"] == "0.02"


def test_hybrid_forces_topology_b_invariants(clean_hb_environment):
    os.environ.update(
        {
            "HB_P2_ASYNC_DECODE": "0",
            "HB_P2_PREFILL_LANES": "2",
            "HB_P2_PREFILL_SHARED_STREAM": "0",
            "HB_P2_EMBED_ASYNC_SCHED": "1",
        }
    )
    args = SimpleNamespace(
        slackserve_dense_model="embedding-model",
        slackserve_dense_served_model_name="embed",
        slackserve_dense_gpu_memory_utilization=0.18,
        slackserve_dense_max_model_len=11264,
        slackserve_dense_max_num_batched_tokens=8192,
        slackserve_dense_max_num_seqs=192,
        worker_cls="auto",
        scheduler_cls=None,
        async_scheduling=None,
        gpu_memory_utilization=0.9,
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        enable_prefix_caching=None,
        compilation_config=CompilationConfig(),
    )

    configure_dense_sidecar(args)

    assert os.environ["HB_P2_ASYNC_DECODE"] == "1"
    assert os.environ["HB_P2_PREFILL_LANES"] == "1"
    assert os.environ["HB_P2_PREFILL_SHARED_STREAM"] == "1"
    assert os.environ["HB_P2_EMBED_ASYNC_SCHED"] == "0"


def test_sidecar_start_failure_is_terminal():
    """A stream-bind failure must fail both current and later submissions."""
    sidecar = HbEmbedSidecar.__new__(HbEmbedSidecar)
    pending: Future[object] = Future()
    sidecar._live_pending = queue.Queue()
    sidecar._live_pending.put(("request", "text", pending))
    sidecar._live_futures = {}
    sidecar._live_state_lock = threading.Lock()
    sidecar._bind_dense_stream = lambda: (_ for _ in ()).throw(
        RuntimeError("stream bind failed")
    )

    sidecar._drain_live()

    assert pending.done()
    assert str(pending.exception()) == "stream bind failed"
