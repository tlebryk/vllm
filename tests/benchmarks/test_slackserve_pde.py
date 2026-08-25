# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Any

import pytest

from vllm.benchmarks.lib import endpoint_request_func as requests


def test_pde_request_preserves_generation_and_dense_work(monkeypatch):
    async def completion(*args, **kwargs):
        return requests.RequestFuncOutput(
            success=True,
            prompt_len=8,
            output_tokens=2,
            latency=0.2,
        )

    async def dense(*args, **kwargs):
        return requests.RequestFuncOutput(
            success=True,
            prompt_len=9,
            latency=0.1,
        )

    monkeypatch.setattr(requests, "async_request_openai_completions", completion)
    monkeypatch.setattr(requests, "_request_dense", dense)
    request = requests.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=8,
        output_len=2,
        model="llm",
        aux_model_name="embed",
    )

    output = asyncio.run(requests.async_request_slackserve_pde(request, session=None))

    assert output.success
    assert output.prompt_len == 8
    assert output.output_tokens == 2
    assert output.aux_success
    assert output.aux_prompt_len == 9


def test_pde_request_fails_if_dense_fails(monkeypatch):
    async def completion(*args, **kwargs):
        return requests.RequestFuncOutput(success=True, prompt_len=8)

    async def dense(*args, **kwargs):
        return requests.RequestFuncOutput(success=False, error="dense failed")

    monkeypatch.setattr(requests, "async_request_openai_completions", completion)
    monkeypatch.setattr(requests, "_request_dense", dense)
    request = requests.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=8,
        output_len=2,
        model="llm",
        aux_model_name="embed",
    )

    output = asyncio.run(requests.async_request_slackserve_pde(request, session=None))

    assert not output.success
    assert "dense failed" in output.error


@pytest.mark.parametrize(
    ("task", "hybrid", "endpoint", "payload_field"),
    [
        ("embed", True, "/v1/embeddings", "input"),
        ("classify", True, "/v1/verify", "input"),
        ("classify", False, "/classify", "input"),
        ("score", True, "/rerank", "documents"),
        ("score", False, "/rerank", "documents"),
    ],
)
def test_dense_request_shapes(monkeypatch, task, hybrid, endpoint, payload_field):
    captured: dict[str, Any] = {}

    async def run_pooling(session, api_url, *, payload, headers, pbar):
        captured.update(api_url=api_url, payload=payload)
        return requests.RequestFuncOutput(success=True, prompt_len=8)

    monkeypatch.setattr(requests, "_run_pooling_request", run_pooling)
    request = requests.RequestFuncInput(
        prompt="document",
        api_url=(
            "http://localhost:8000/v1/completions"
            if hybrid
            else "http://localhost:8000/classify"
        ),
        prompt_len=8,
        output_len=2,
        model="llm",
        aux_model_name="dense",
        aux_task=task,
        aux_query="query",
    )

    output = asyncio.run(requests._request_dense(request, session=None, hybrid=hybrid))

    assert output.success
    assert captured["api_url"].endswith(endpoint)
    assert payload_field in captured["payload"]
    assert captured["payload"]["model"] == ("dense" if hybrid else "llm")
