# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling endpoints backed by Slack Serve's in-process dense sidecar."""

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
from vllm.entrypoints.pooling.score.protocol import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from vllm.entrypoints.utils import with_cancellation
from vllm.utils import random_uuid

router = APIRouter()


class SlackServeEmbeddingRequest(OpenAIBaseModel):
    """Minimal OpenAI-compatible embedding request for the PD-E POC."""

    model: str
    input: str | list[str]


class SlackServeVerifyRequest(OpenAIBaseModel):
    """One or more complete prompts for a scalar reward model."""

    model: str
    input: str | list[str]
    use_activation: bool | None = None


class VerifyResponseData(OpenAIBaseModel):
    index: int
    score: float


class VerifyResponse(OpenAIBaseModel):
    id: str
    model: str
    data: list[VerifyResponseData]
    usage: UsageInfo


def _error(message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorInfo(message=message, type="InvalidRequestError", code=status_code)
    )
    return JSONResponse(content=body.model_dump(), status_code=status_code)


def _served_name() -> str:
    return os.environ.get("HB_P2_DENSE_SERVED_MODEL", "embed")


def _require_task(task: str) -> JSONResponse | None:
    configured = os.environ.get("HB_P2_DENSE_TASK", "embed")
    if configured == task:
        return None
    return _error(
        f"This dense model is configured for task `{configured}`, not `{task}`.",
        HTTPStatus.BAD_REQUEST.value,
    )


def _engine_core(raw_request: Request):  # noqa: ANN202
    engine_client = raw_request.app.state.engine_client
    return getattr(engine_client, "engine_core", None)


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
    if task_error := _require_task("embed"):
        return task_error
    served_name = _served_name()
    if request.model != served_name:
        return _error(
            f"The embedding model `{request.model}` does not exist.",
            HTTPStatus.NOT_FOUND.value,
        )

    engine_core = _engine_core(raw_request)
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


@router.post(
    "/v1/verify",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": VerifyResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_verification(
    request: SlackServeVerifyRequest, raw_request: Request
) -> VerifyResponse:
    if task_error := _require_task("classify"):
        return task_error
    served_name = _served_name()
    if request.model != served_name:
        return _error(
            f"The verifier model `{request.model}` does not exist.",
            HTTPStatus.NOT_FOUND.value,
        )
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        return _error("`input` must not be empty.", HTTPStatus.BAD_REQUEST.value)
    engine_core = _engine_core(raw_request)
    if engine_core is None or not hasattr(engine_core, "call_utility_async"):
        return _error(
            "Slack Serve verification requires the V1 EngineCore client.",
            HTTPStatus.BAD_REQUEST.value,
        )

    request_ids = [f"verify-{random_uuid()}" for _ in texts]
    results = await asyncio.gather(
        *(
            engine_core.call_utility_async(
                "submit_hb_classification",
                request_id,
                text,
                request.use_activation,
            )
            for request_id, text in zip(request_ids, texts, strict=True)
        )
    )
    if any(len(result["data"]) != 1 for result in results):
        return _error(
            "Verifier output must contain exactly one scalar score.",
            HTTPStatus.BAD_REQUEST.value,
        )
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    return VerifyResponse(
        id=request_ids[0],
        model=served_name,
        data=[
            VerifyResponseData(index=index, score=float(result["data"][0]))
            for index, result in enumerate(results)
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )


@router.post("/rerank", dependencies=[Depends(validate_json_request)])
@router.post("/v1/rerank", dependencies=[Depends(validate_json_request)])
@router.post("/v2/rerank", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def create_rerank(request: RerankRequest, raw_request: Request) -> RerankResponse:
    if task_error := _require_task("score"):
        return task_error
    served_name = _served_name()
    if request.model is not None and request.model != served_name:
        return _error(
            f"The reranker model `{request.model}` does not exist.",
            HTTPStatus.NOT_FOUND.value,
        )
    if not isinstance(request.query, str):
        return _error(
            "The initial shared-stream reranker supports text queries only.",
            HTTPStatus.BAD_REQUEST.value,
        )
    documents = (
        request.documents
        if isinstance(request.documents, list)
        else [request.documents]
    )
    if not documents or not all(isinstance(document, str) for document in documents):
        return _error(
            "`documents` must be a non-empty string or list of strings.",
            HTTPStatus.BAD_REQUEST.value,
        )
    engine_core = _engine_core(raw_request)
    if engine_core is None or not hasattr(engine_core, "call_utility_async"):
        return _error(
            "Slack Serve reranking requires the V1 EngineCore client.",
            HTTPStatus.BAD_REQUEST.value,
        )

    request_id = f"rerank-{random_uuid()}"
    results = await asyncio.gather(
        *(
            engine_core.call_utility_async(
                "submit_hb_score",
                f"{request_id}-{index}",
                request.query,
                document,
                request.use_activation,
                request.truncate_prompt_tokens,
                request.truncation_side,
            )
            for index, document in enumerate(documents)
        )
    )
    if any(len(result["data"]) != 1 for result in results):
        return _error(
            "Reranker output must contain exactly one scalar score.",
            HTTPStatus.BAD_REQUEST.value,
        )
    ranked = sorted(
        (
            RerankResult(
                index=index,
                document=RerankDocument(text=document),
                relevance_score=float(result["data"][0]),
            )
            for index, (document, result) in enumerate(
                zip(documents, results, strict=True)
            )
        ),
        key=lambda item: item.relevance_score,
        reverse=True,
    )
    top_n = request.top_n if request.top_n > 0 else len(ranked)
    prompt_tokens = sum(result["prompt_tokens"] for result in results)
    return RerankResponse(
        id=request_id,
        model=served_name,
        results=ranked[:top_n],
        usage=RerankUsage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
