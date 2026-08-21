# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from vllm.benchmarks.lib import endpoint_request_func as requests


def test_pde_request_preserves_generation_and_embedding_work(monkeypatch):
    async def completion(*args, **kwargs):
        return requests.RequestFuncOutput(
            success=True,
            prompt_len=8,
            output_tokens=2,
            latency=0.2,
        )

    async def embedding(*args, **kwargs):
        return requests.RequestFuncOutput(
            success=True,
            prompt_len=9,
            latency=0.1,
        )

    monkeypatch.setattr(requests, "async_request_openai_completions", completion)
    monkeypatch.setattr(requests, "async_request_openai_embeddings", embedding)
    request = requests.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=8,
        output_len=2,
        model="llm",
        aux_model_name="embed",
    )

    output = asyncio.run(
        requests.async_request_slackserve_pde(request, session=None)
    )

    assert output.success
    assert output.prompt_len == 8
    assert output.output_tokens == 2
    assert output.aux_success
    assert output.aux_prompt_len == 9


def test_pde_request_fails_if_embedding_fails(monkeypatch):
    async def completion(*args, **kwargs):
        return requests.RequestFuncOutput(success=True, prompt_len=8)

    async def embedding(*args, **kwargs):
        return requests.RequestFuncOutput(success=False, error="dense failed")

    monkeypatch.setattr(requests, "async_request_openai_completions", completion)
    monkeypatch.setattr(requests, "async_request_openai_embeddings", embedding)
    request = requests.RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=8,
        output_len=2,
        model="llm",
        aux_model_name="embed",
    )

    output = asyncio.run(
        requests.async_request_slackserve_pde(request, session=None)
    )

    assert not output.success
    assert "dense failed" in output.error
