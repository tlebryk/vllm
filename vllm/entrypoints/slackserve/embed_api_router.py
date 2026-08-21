# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI embedding endpoint backed by Slack Serve's in-process sidecar."""

import asyncio
import os
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingResponse,
    EmbeddingResponseData,
)
from vllm.entrypoints.utils import with_cancellation
from vllm.utils import random_uuid

router = APIRouter()


class SlackServeEmbeddingRequest(OpenAIBaseModel):
    """Minimal OpenAI-compatible embedding request for the PD-E POC."""

    model: str
    input: str | list[str]


def _error(message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorInfo(message=message, type="InvalidRequestError", code=status_code)
    )
    return JSONResponse(content=body.model_dump(), status_code=status_code)


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": EmbeddingResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_embedding(
    request: SlackServeEmbeddingRequest, raw_request: Request
) -> EmbeddingResponse:
    served_name = os.environ.get("HB_P2_DENSE_SERVED_MODEL", "embed")
    if request.model != served_name:
        return _error(
            f"The embedding model `{request.model}` does not exist.",
            HTTPStatus.NOT_FOUND.value,
        )

    engine_client = raw_request.app.state.engine_client
    engine_core = getattr(engine_client, "engine_core", None)
    if engine_core is None or not hasattr(engine_core, "call_utility_async"):
        return _error(
            "Slack Serve embeddings require the V1 asynchronous EngineCore client.",
            HTTPStatus.BAD_REQUEST.value,
        )

    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        return _error("`input` must not be empty.", HTTPStatus.BAD_REQUEST.value)

    request_ids = [f"embd-{random_uuid()}" for _ in texts]
    results = await asyncio.gather(
        *(
            engine_core.call_utility_async("submit_hb_embedding", request_id, text)
            for request_id, text in zip(request_ids, texts, strict=True)
        )
    )
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    return EmbeddingResponse(
        id=request_ids[0],
        model=served_name,
        data=[
            EmbeddingResponseData(index=index, embedding=result["data"])
            for index, result in enumerate(results)
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
