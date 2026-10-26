# SPDX-License-Identifier: Apache-2.0
"""Native AsyncLLM/OpenAI integration for model-owned lane execution."""

import json
import os
from contextlib import asynccontextmanager

from starlette.responses import JSONResponse


@asynccontextmanager
async def build_multi_model_clients(args):
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.multi_model_backend import (
        ModelCoreClient,
        MultiModelBackend,
        prepare_models,
    )
    from vllm.v1.executor import Executor

    if (
        args.api_server_count not in (None, 1)
        or args.headless
        or args.tensor_parallel_size != 1
        or args.pipeline_parallel_size != 1
        or args.data_parallel_size != 1
        or args.enable_lora
        or args.speculative_config
        or os.environ.get("HB_P2_SERVE") == "1"
    ):
        raise ValueError(
            "shared dual-model serving requires one API process/GPU without "
            "distributed, legacy sidecar, LoRA or speculative modes"
        )
    specs, configs, budget = prepare_models(
        args.slackserve_models,
        mode=args.slackserve_shared_kv,
        budget=args.slackserve_kv_memory_bytes,
        decode_depth=args.slackserve_decode_depth,
    )
    backend = MultiModelBackend(
        specs,
        configs,
        mode=args.slackserve_shared_kv,
        budget=budget,
        admission=args.slackserve_kv_admission,
        topology=args.slackserve_stream_topology,
        decode_depth=args.slackserve_decode_depth,
        max_requests=args.slackserve_max_requests,
        offline_serial_counts=(
            json.loads(args.slackserve_offline_serial_counts)
            if args.slackserve_offline_serial_counts is not None
            else None
        ),
        step_trace=args.slackserve_step_trace,
        smctrl_mask=args.slackserve_smctrl_mask,
        smctrl_lib=args.slackserve_smctrl_lib,
        smctrl_min_decode_requests=args.slackserve_smctrl_min_decode_requests,
        hybrid_min_decode=args.slackserve_hybrid_min_decode,
        mask_tiers=(
            json.loads(args.slackserve_mask_tiers)
            if args.slackserve_mask_tiers is not None
            else None
        ),
        dense_step_tokens=args.slackserve_dense_step_tokens,
        adaptive_dense_steps=args.slackserve_dense_adaptive_step,
        smctrl_indices=(
            json.loads(args.slackserve_smctrl_indices)
            if args.slackserve_smctrl_indices is not None
            else None
        ),
        smctrl_max_dense_wait_ms=args.slackserve_smctrl_max_dense_wait_ms,
    )
    clients = {}
    try:
        await backend.start()
        for spec, config in zip(specs, configs):
            clients[spec.model_id] = AsyncLLM(
                config,
                Executor.get_class(config),
                log_stats=not args.disable_log_stats,
                log_requests=args.enable_log_requests,
                engine_core_client=ModelCoreClient(backend, spec.model_id),
            )
        primary = clients[specs[0].model_id]
        primary.model_clients = clients
        primary.multi_model_backend = backend
        primary.multi_model_specs = specs
        yield primary
    finally:
        for client in clients.values():
            client.shutdown()
        await backend.close()


class MultiModelRoutingMiddleware:
    """Route model-local native handlers; bound live HTTP bodies/streams."""

    generation_paths = {"/v1/completions", "/v1/chat/completions"}
    pooling_paths = {
        "/classify",
        "/pooling",
        "/v1/verify",
        "/v1/embeddings",
        "/score",
        "/v1/score",
        "/rerank",
        "/v1/rerank",
        "/v2/rerank",
    }
    model_paths = (
        generation_paths
        | pooling_paths
        | {
            "/tokenize",
            "/detokenize",
            "/v1/completions/render",
            "/v1/chat/completions/render",
        }
    )

    def __init__(self, app, max_requests=8192, max_body_bytes=1048576):
        self.app = app
        self.max_requests, self.max_body_bytes = max_requests, max_body_bytes
        self.active = 0

    async def _error(self, scope, receive, send, message, code=400):
        await JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "BadRequestError",
                    "param": None,
                    "code": code,
                }
            },
            status_code=code,
        )(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        state = scope["app"].state
        if path == "/slackserve/stats" and scope["method"] == "GET":
            return await JSONResponse(
                state.engine_client.multi_model_backend.snapshot()
            )(scope, receive, send)
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)
        if path not in self.model_paths:
            return await self._error(
                scope,
                receive,
                send,
                "endpoint is unsupported in dual-model serving",
                501,
            )
        if self.active >= self.max_requests:
            return await self._error(
                scope, receive, send, "too many active requests; retry later", 429
            )
        self.active += 1
        try:
            body = bytearray()
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                body.extend(event.get("body", b""))
                if len(body) > self.max_body_bytes:
                    return await self._error(
                        scope, receive, send, "request body is too large", 413
                    )
                if not event.get("more_body", False):
                    break
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError()
            except (ValueError, TypeError):
                return await self._error(
                    scope, receive, send, "request must be a JSON object"
                )
            states = state.multi_model_states
            mid = payload.get("model")
            if mid is None:
                mid = next(iter(states))
            if not isinstance(mid, str):
                return await self._error(
                    scope, receive, send, "model must be a string", 422
                )
            if mid not in states:
                return await self._error(
                    scope, receive, send, f"model {mid!r} does not exist", 404
                )
            config = getattr(
                getattr(states[mid], "engine_client", None), "model_config", None
            )
            if config is not None:
                generation = config.runner_type == "generate"
                if path in self.generation_paths and not generation:
                    return await self._error(
                        scope, receive, send, "this model does not generate"
                    )
                if path in self.pooling_paths and generation:
                    return await self._error(
                        scope, receive, send, "this model does not support pooling"
                    )
            if path in self.generation_paths:
                fmt = payload.get("response_format")
                if fmt is None:
                    fmt = {}
                count = payload.get("n", 1)
                if not isinstance(fmt, dict) or type(count) is not int:
                    return await self._error(
                        scope, receive, send, "invalid response_format or n", 422
                    )
                if (
                    payload.get("structured_outputs") is not None
                    or any(
                        payload.get(k) is not None
                        for k in (
                            "guided_json",
                            "guided_regex",
                            "guided_choice",
                            "guided_grammar",
                        )
                    )
                    or fmt.get("type", "text") != "text"
                    or payload.get("tools")
                ):
                    return await self._error(
                        scope,
                        receive,
                        send,
                        "structured generation and tool grammars are unsupported "
                        "in dual-model serving",
                    )
                if payload.get("n", 1) > 16:
                    return await self._error(
                        scope, receive, send, "n must not exceed 16"
                    )
                if (
                    getattr(
                        getattr(state, "engine_client", None),
                        "multi_model_backend",
                        None,
                    )
                    is not None
                    and state.engine_client.multi_model_backend.offline_serial_counts
                    is not None
                    and (
                        count != 1
                        or path == "/v1/completions"
                        and isinstance(payload.get("prompt"), list)
                        and any(type(p) is not int for p in payload["prompt"])
                    )
                ):
                    return await self._error(
                        scope,
                        receive,
                        send,
                        "offline model-serial counts require one prompt and n=1 "
                        "per HTTP request",
                    )
            backend = getattr(
                getattr(state, "engine_client", None), "multi_model_backend", None
            )
            if (
                path in self.pooling_paths
                and backend is not None
                and backend.offline_serial_counts is not None
            ):
                # Counts refer to engine requests, not HTTP batch envelopes.
                # Reject fan-out before admitting any part of the envelope.
                fields = ("input", "documents", "text_1", "text_2")
                if any(
                    isinstance(payload.get(field), list)
                    and (
                        not payload[field]
                        or not all(type(item) is int for item in payload[field])
                        and len(payload[field]) != 1
                    )
                    for field in fields
                ):
                    return await self._error(
                        scope,
                        receive,
                        send,
                        "offline model-serial counts require one pooling input "
                        "per HTTP request",
                    )
            scope.setdefault("state", {})["slackserve_model_state"] = states[mid]
            replayed = False

            async def replay_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            await self.app(scope, replay_receive, send)
        finally:
            self.active -= 1
