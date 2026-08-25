# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only coverage for Slack Serve's dense endpoints and failure boundary."""

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
from vllm.entrypoints.pooling.score.protocol import RerankRequest
from vllm.entrypoints.slackserve.config import configure_dense_sidecar
from vllm.entrypoints.slackserve.embed_api_router import (
    SlackServeEmbeddingRequest,
    SlackServeVerifyRequest,
    create_embedding,
    create_rerank,
    create_verification,
)
from vllm.v1.engine.hb_embed_sidecar import HbEmbedSidecar


class _Core:
    async def call_utility_async(self, method: str, request_id: str, *args):
        if method == "submit_hb_score":
            _, document, *_ = args
            return {
                "request_id": request_id,
                "data": [0.9 if "Paris" in document else 0.1],
                "prompt_tokens": 7,
            }
        text = args[0]
        return {
            "request_id": request_id,
            "data": [0.8] if method == "submit_hb_classification" else [0.25, 0.75],
            "prompt_tokens": len(text.split()),
        }


def _raw_request():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(engine_client=SimpleNamespace(engine_core=_Core()))
        )
    )


def test_embedding_routes_to_sidecar(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "embed")
    monkeypatch.setenv("HB_P2_DENSE_TASK", "embed")

    response = asyncio.run(
        create_embedding.__wrapped__(
            SlackServeEmbeddingRequest(model="embed", input=["hello", "two words"]),
            _raw_request(),
        )
    )

    assert [item.embedding for item in response.data] == [
        [0.25, 0.75],
        [0.25, 0.75],
    ]
    assert response.usage.prompt_tokens == 3


def test_embedding_rejects_unknown_model(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "embed")
    monkeypatch.setenv("HB_P2_DENSE_TASK", "embed")
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    response = asyncio.run(
        create_embedding.__wrapped__(
            SlackServeEmbeddingRequest(model="wrong", input="hello"), raw_request
        )
    )

    assert response.status_code == 404
    assert "does not exist" in json.loads(response.body)["error"]["message"]


def test_verification_routes_to_classification_sidecar(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "reward")
    monkeypatch.setenv("HB_P2_DENSE_TASK", "classify")

    response = asyncio.run(
        create_verification.__wrapped__(
            SlackServeVerifyRequest(model="reward", input=["prompt one", "prompt two"]),
            _raw_request(),
        )
    )

    assert [item.score for item in response.data] == [0.8, 0.8]
    assert response.usage.prompt_tokens == 4


def test_rerank_uses_native_query_document_semantics(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_SERVED_MODEL", "reranker")
    monkeypatch.setenv("HB_P2_DENSE_TASK", "score")

    response = asyncio.run(
        create_rerank.__wrapped__(
            RerankRequest(
                model="reranker",
                query="What is the capital of France?",
                documents=["Brasilia is in Brazil.", "Paris is in France."],
                top_n=1,
            ),
            _raw_request(),
        )
    )

    assert len(response.results) == 1
    assert response.results[0].index == 1
    assert response.results[0].relevance_score == 0.9
    assert response.usage.prompt_tokens == 14


def test_endpoint_rejects_wrong_dense_task(monkeypatch):
    monkeypatch.setenv("HB_P2_DENSE_TASK", "classify")

    response = asyncio.run(
        create_embedding.__wrapped__(
            SlackServeEmbeddingRequest(model="embed", input="hello"),
            _raw_request(),
        )
    )

    assert response.status_code == 400
    assert "classify" in json.loads(response.body)["error"]["message"]


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


def _hybrid_args(task: str = "embed"):
    return SimpleNamespace(
        slackserve_dense_model="dense-model",
        slackserve_dense_served_model_name="dense",
        slackserve_dense_task=task,
        slackserve_dense_hf_overrides=None,
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


def test_hybrid_defaults_are_reproducible(clean_hb_environment):
    args = _hybrid_args()

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
    assert os.environ["HB_P2_DENSE_TASK"] == "embed"


def test_hybrid_forces_topology_b_invariants(clean_hb_environment):
    os.environ.update(
        {
            "HB_P2_ASYNC_DECODE": "0",
            "HB_P2_PREFILL_LANES": "2",
            "HB_P2_PREFILL_SHARED_STREAM": "0",
            "HB_P2_EMBED_ASYNC_SCHED": "1",
        }
    )
    args = _hybrid_args()

    configure_dense_sidecar(args)

    assert os.environ["HB_P2_ASYNC_DECODE"] == "1"
    assert os.environ["HB_P2_PREFILL_LANES"] == "1"
    assert os.environ["HB_P2_PREFILL_SHARED_STREAM"] == "1"
    assert os.environ["HB_P2_EMBED_ASYNC_SCHED"] == "0"


def test_hybrid_exports_reranker_configuration(clean_hb_environment):
    args = _hybrid_args("score")
    args.slackserve_dense_hf_overrides = '{"architectures":["Reranker"]}'

    configure_dense_sidecar(args)

    assert os.environ["HB_P2_DENSE_TASK"] == "score"
    assert os.environ["HB_P2_DENSE_HF_OVERRIDES"] == '{"architectures":["Reranker"]}'


def test_hybrid_rejects_invalid_hf_overrides(clean_hb_environment):
    args = _hybrid_args("score")
    args.slackserve_dense_hf_overrides = "not-json"

    with pytest.raises(ValueError, match="valid JSON"):
        configure_dense_sidecar(args)


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
