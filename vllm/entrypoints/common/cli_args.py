# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file contains the command line arguments for the vLLM's online server.
It is kept in a separate file for documentation purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Literal, get_args

import vllm.envs as envs
from vllm.config import config
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.anthropic.protocol import (
    AnthropicDisabledThinkingEffortOption,
)
from vllm.entrypoints.launchers.utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.logger import init_logger
from vllm.renderers import ReasoningEffort
from vllm.renderers.chat_utils import (
    ChatTemplateContentFormatOption,
    validate_chat_template,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.tool_strict_level import ToolStrictLevelName
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


class LoRAParserAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ):
        if values is None:
            values = []
        if isinstance(values, str):
            raise TypeError("Expected values to be a list")

        lora_list: list[LoRAModulePath] = []
        for item in values:
            if item in [None, ""]:  # Skip if item is None or empty string
                continue
            if "=" in item and "," not in item:  # Old format: name=path
                name, path = item.split("=", 1)
                lora_list.append(LoRAModulePath(name, path))
            else:  # Assume JSON format
                try:
                    lora_dict = json.loads(item)
                    lora = LoRAModulePath(**lora_dict)
                    lora_list.append(lora)
                except json.JSONDecodeError:
                    parser.error(f"Invalid JSON format for --lora-modules: {item}")
                except TypeError as e:
                    parser.error(
                        f"Invalid fields for --lora-modules: {item} - {str(e)}"
                    )
        setattr(namespace, self.dest, lora_list)


@config
class BaseFrontendArgs:
    """Base arguments for the OpenAI-compatible frontend server.

    This base class does not include host, port, and server-specific arguments
    like SSL, CORS, and HTTP server settings. Those arguments are added by
    the subclasses.
    """

    lora_modules: list[LoRAModulePath] | None = None
    """LoRA modules configurations in either 'name=path' format or JSON format
    or JSON list format. Example (old format): `'name=path'` Example (new
    format): `{\"name\": \"name\", \"path\": \"lora_path\",
    \"base_model_name\": \"id\"}`"""
    chat_template: str | None = None
    """The file path to the chat template, or the template in single-line form
    for the specified model."""
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    """The format to render message content within a chat template.

    * "string" will render the content as a string. Example: `"Hello World"`
    * "openai" will render the content as a list of dictionaries, similar to
      OpenAI schema. Example: `[{"type": "text", "text": "Hello world!"}]`"""
    trust_request_chat_template: bool = False
    """Whether to trust the chat template provided in the request. If False,
    the server will always use the chat template specified by `--chat-template`
    or the ones from tokenizer."""
    trust_request_mm_kwargs: bool = False
    """Whether to trust per-request multimodal kwargs (`mm_processor_kwargs`
    and `media_io_kwargs`). If False, the server rejects non-empty values
    because they can change multimodal preprocessing resource usage. Only
    enable this when API clients are trusted."""
    default_chat_template_kwargs: dict[str, Any] | None = None
    """Default keyword arguments to pass to the chat template renderer.
    These will be merged with request-level chat_template_kwargs,
    with request values taking precedence. Useful for setting default
    behavior for reasoning models. Example: '{"enable_thinking": false}'
    to disable thinking mode by default for Qwen3/DeepSeek models."""
    reasoning_effort_map: dict[str, str] | None = None
    """JSON mapping that rewrites a request's reasoning effort to a value the
    chat template supports, for templates that silently fall back on efforts
    they do not recognize. Applies to the effort set through the Chat
    Completions, Responses and Anthropic Messages APIs; efforts not listed
    pass through unchanged. Example: '{"medium": "low", "xhigh": "max"}'
    for a template that only knows low, high and max."""
    response_role: str = "assistant"
    """The role name to return if `request.add_generation_prompt=true`."""
    return_tokens_as_token_ids: bool = False
    """When `--max-logprobs` is specified, represents single tokens as
    strings of the form 'token_id:{token_id}' so that tokens that are not
    JSON-encodable can be identified."""
    enable_auto_tool_choice: bool = False
    """Enable auto tool choice for supported models. Use `--tool-call-parser`
    to specify which parser to use."""
    exclude_tools_when_tool_choice_none: bool = False
    """If specified, exclude tool definitions in prompts when
    tool_choice='none'."""
    tool_call_parser: str | None = None
    """Select the tool call parser depending on the model that you're using.
    This is used to parse the model-generated tool call into OpenAI API format.
    Required for `--enable-auto-tool-choice`. You can choose any option from
    the built-in parsers or register a plugin via `--tool-parser-plugin`."""
    tool_parser_plugin: str = ""
    """Special the tool parser plugin write to parse the model-generated tool
    into OpenAI API format, the name register in this plugin can be used in
    `--tool-call-parser`."""
    tool_strict_level: ToolStrictLevelName | None = None
    """Server-side floor for structural-tag based tool calling, applied on top
    of the per-tool `strict` field. `auto` follows the request's tool choice
    and per-tool strictness; `function` constrains the
    tool-call envelope (markup and function name) for every request with
    tools; `parameter` additionally pins argument schemas, as if every tool
    were `strict: true`. When unset, the tool parser's default level applies,
    which is `auto` for most parsers."""
    tool_server: str | None = None
    """Comma-separated list of host:port pairs (IPv4, IPv6, or hostname).
    Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for
    built-in demo tools (browser and Python code interpreter). WARNING:
    The `demo` Python tool executes model-generated code in Docker without
    network isolation by default. See the security guide for more
    information."""
    max_log_len: int | None = None
    """Max number of prompt characters or prompt ID numbers being printed in
    log. The default of None means unlimited."""
    enable_prompt_tokens_details: bool = False
    """If set to True, enable prompt_tokens_details in usage."""
    enable_per_request_metrics: bool = False
    """If set to True, include per-request timing metrics in API responses."""
    enable_server_load_tracking: bool = False
    """If set to True, enable tracking server_load_metrics in the app state."""
    enable_force_include_usage: bool = False
    """If set to True, including usage on every request."""
    sse_keep_alive_interval: int = 0
    """Send an SSE keep-alive comment line every this many seconds when a
    `/v1/chat/completions` or `/v1/completions` streaming response is idle
    (queued, prefill, or between tokens), to prevent reverse proxies/tunnels
    with read timeouts from closing the connection. Defaults to 0, which
    disables keep-alive comments entirely.
    """
    enable_tokenizer_info_endpoint: bool = False
    """Enable the `/tokenizer_info` endpoint. May expose chat
    templates and other tokenizer configuration."""
    enable_log_outputs: bool = False
    """If set to True, log model outputs (generations). Requires
    `--enable-log-requests`. Output text and finish reasons are logged at INFO,
    while output token IDs are logged at DEBUG."""
    enable_log_deltas: bool = True
    """If set to False, output deltas will not be logged. Relevant only if 
    --enable-log-outputs is set.
    """
    cohere_is_reasoning_model: bool = True
    """Cohere ``/cohere/v2/chat`` only. Whether the served model is a
    reasoning Command-family model. When True (default), the assistant's
    chain-of-thought is surfaced as a ``thinking`` content block (or
    ``content-*`` events on the stream). When False, reasoning is
    surfaced as Cohere's ``tool_plan`` field (or ``tool-plan-delta``
    events) whenever the model emits tool calls, matching older non-
    reasoning Command models. Has no effect on the non-Cohere
    endpoints."""
    cohere_format: str = "cmd4"
    """Cohere ``--tokenizer-mode cohere`` only. Which Cohere prompt
    format to render: ``cmd4`` (current Command A+ models; default) or
    ``cmd3`` (earlier Cmd-A and Cmd-A reasoning models). Selecting the
    wrong format silently produces a prompt the model wasn't trained
    on, which most commonly manifests as the model emitting text but no
    citations / tool calls / thinking blocks. Equivalent to passing
    ``--default-chat-template-kwargs '{"cohere_format": "..."}'`` -- any
    explicit request-level ``chat_template_kwargs.cohere_format`` takes
    priority."""
    anthropic_disabled_thinking_effort: AnthropicDisabledThinkingEffortOption = "auto"
    """The ``reasoning_effort`` used for Anthropic Messages, token counting,
    and render requests with ``thinking: {"type": "disabled"}``.
    ``none`` turns thinking off for models that support it; ``low`` suits
    models that always think
    (e.g. GLM-5.3) or reject ``none`` (e.g. gpt-oss). ``auto`` (default) uses
    ``low`` when the renderer rejects ``none`` or renders it the same as a
    thinking effort, and ``none`` otherwise."""
    log_error_stack: bool = envs.VLLM_SERVER_DEV_MODE
    """If set to True, log the stack trace of error responses"""
    tokens_only: bool = False
    """
    If set to True, only enable the Tokens In<>Out endpoint.
    This is intended for use in a Disaggregated Everything setup.
    """
    enable_scale_out: bool = False
    """
    If set to True, register the scale-out endpoints (`/render`, `/derender`,
    `/inference/v1/generate` and `/inference/v1/abort_requests`) on `vllm serve`.
    Has no effect on `vllm launch render` or `vllm serve --tokens-only` which
    always register their required endpoints regardless of this flag.
    """
    fingerprint_mode: Literal["full", "hash", "custom", "none"] = "full"
    """Controls the ``system_fingerprint`` field on responses.

    - ``full`` (default): ``vllm-<version>[-<parallelism>]-<hash8>``. Encodes
      server version, non-trivial parallelism degrees (tp/pp/dp/ep), and an
      8-char config hash.
    - ``hash``: ``vllm-<version>-<hash8>``. Parallelism stripped.
    - ``custom``: emits the literal string from ``--fingerprint-value``.
    - ``none``: the field is omitted (serialized as ``null``).
    """
    fingerprint_value: str | None = None
    """Literal fingerprint string used when ``--fingerprint-mode=custom``."""

    slackserve_models: str | None = None
    """JSON configuration for one or two native models on one GPU.

    A single-model P/D run requires --slackserve-shared-kv private.
    """
    slackserve_shared_kv: Literal["private", "head", "layer"] = "layer"
    """KV capacity layout for the native dual-model backend."""
    slackserve_kv_memory_bytes: int | None = None
    """Total shared KV bytes; defaults to the sum of model budgets."""
    slackserve_kv_admission: Literal["current", "full-output"] = "current"
    """Native footprint admission or full requested-output reservation."""
    slackserve_stream_topology: Literal[
        "single-stream", "two-stream", "three-stream-ordered", "three-stream-overlap"
    ] = "three-stream-overlap"
    """Execution streams; single-stream serializes both models' GPU work."""
    slackserve_decode_depth: int = 2
    """Global number of decode tickets, shared across models."""
    slackserve_max_requests: int = 8192
    """Bound on active/queued native requests and live HTTP streams."""
    slackserve_offline_serial_counts: str | None = None
    """JSON request counts by model ID for one server-gated model-serial drain.

    Accept both queues immediately; run models in configuration order. The
    expected counts prevent an initial arrival gap from releasing model B.
    """
    slackserve_step_trace: str | None = None
    """Optional streamed JSONL step records and NVTX labels for diagnostics."""
    slackserve_smctrl_mask: str | None = None
    """Optional MODEL:prefill:TPCS spread mask while a partner has decode work.

    Restricted to concurrent topology and eager prefill. Decode graphs retain
    their original placement; replay-time stream masks cannot restrict them.
    """
    slackserve_smctrl_lib: str | None = None
    """libsmctrl shared library required by --slackserve-smctrl-mask."""
    slackserve_smctrl_indices: str | None = None
    """Optional JSON list of enabled physical TPC indices, replacing spread."""
    slackserve_mask_tiers: str | None = None
    """Optional JSON list of load tiers for the generation prefill mask:
    [{"min_decode": N, "indices": [TPC...]}, ...]. While the mask is active,
    the tier with the largest min_decode not above the decode load is applied.
    """
    slackserve_mask_candidates: str | None = None
    """Optional JSON list of prefill-mask candidates for the calibrated policy:
    [{"name": "spread48", "indices": [TPC...]}, {"name": "none", "indices": null}].
    Defaults to the mask's own indices, every tier's indices, and unrestricted.
    """
    slackserve_mask_calibration: str = "off"
    """Calibrated per-ticket mask policy: off, auto (reuse a cached table for
    this deployment key, else measure at startup), or force (always measure).
    """
    slackserve_mask_calibration_dir: str | None = None
    """Cache directory for calibration tables (default ~/.cache/slackserve)."""
    slackserve_smctrl_max_dense_wait_ms: float = 0
    """Restore full-GPU dense access once its oldest request reaches this age.

    Zero disables the age guard. Generation prefill retains its decode gate.
    """
    slackserve_smctrl_min_decode_requests: int = 1
    """Apply the prefill mask only at this many decode-ready requests.

    This controls placement, not admission or scheduler batch capacity.
    """
    slackserve_hybrid_min_decode: int = 0
    """Below this many decode-ready requests, run stock hybrid batching.

    The controller dispatches the scheduler's merged batch (prefill chunks and
    decodes in one forward) on the decode stream, unmasked, instead of split
    prefill/decode steps. Zero disables the fallback.
    """
    slackserve_dense_model: str | None = None
    """Auxiliary pooling checkpoint for the native multi-model controller.

    Runs on the shared prefill stream, with generation prefill taking priority.
    Uses private KV and the same request transport as native dual LLM serving.
    """
    slackserve_dense_step_tokens: int = 0
    """Candidate dense work-unit tokens per dispatch (0 uses full capacity).

    Does not resize engine buffers or reduce scheduler capacity. Sequential
    controls should leave this unset and tune the dense engine capacity.
    """
    slackserve_dense_adaptive_step: bool = False
    """Use small dense work units only while its dispatch is masked.

    Full capacity is restored with little decode demand or an age override.
    Requires a mask and a positive --slackserve-dense-step-tokens.
    """
    slackserve_dense_served_model_name: str = "embed"
    """Public ``model`` name accepted by the Slack Serve dense endpoint."""
    slackserve_dense_task: Literal["embed", "score", "classify"] = "embed"
    """Auxiliary model task: embedding, cross-encoder reranking, or reward
    classification."""
    slackserve_dense_hf_overrides: str | None = None
    """Optional JSON Hugging Face config overrides for the auxiliary model."""
    slackserve_dense_gpu_memory_utilization: float = 0.18
    """GPU-memory fraction reserved for the auxiliary embedding engine."""
    slackserve_dense_kv_memory_bytes: int | None = None
    """Explicit private dense KV bytes; otherwise use utilization-based sizing."""
    slackserve_dense_max_model_len: int = 11264
    """Maximum input length served by the auxiliary embedding engine."""
    slackserve_dense_max_num_batched_tokens: int = 8192
    """Embedding token budget per shared-dense-stream work unit."""
    slackserve_dense_max_num_seqs: int = 192
    """Auxiliary scheduler capacity; physical memory remains the residency limit."""

    @classmethod
    def _customize_cli_kwargs(
        cls,
        frontend_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Customize argparse kwargs before arguments are registered.

        Subclasses should override this and call
        ``super()._customize_cli_kwargs(frontend_kwargs)`` first.
        """
        # Special case: default_chat_template_kwargs needs json.loads type
        frontend_kwargs["default_chat_template_kwargs"]["type"] = json.loads
        frontend_kwargs["reasoning_effort_map"]["type"] = json.loads

        # Special case: LoRA modules need custom parser action and
        # optional_type(str)
        frontend_kwargs["lora_modules"]["type"] = optional_type(str)
        frontend_kwargs["lora_modules"]["action"] = LoRAParserAction

        # Special case: Tool call parser shows built-in options.
        valid_tool_parsers = list(ToolParserManager.list_registered())
        parsers_str = ",".join(valid_tool_parsers)
        frontend_kwargs["tool_call_parser"]["metavar"] = (
            f"{{{parsers_str}}} or name registered in --tool-parser-plugin"
        )
        return frontend_kwargs

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Register CLI arguments for this frontend class.

        Subclasses should override ``_customize_cli_kwargs`` instead of
        this method so that base-class postprocessing is always applied.
        """
        from vllm.engine.arg_utils import get_kwargs

        frontend_kwargs = get_kwargs(cls)
        frontend_kwargs = cls._customize_cli_kwargs(frontend_kwargs)

        group_name = cls.__name__.replace("Args", "")
        frontend_group = parser.add_argument_group(
            title=group_name,
            description=cls.__doc__,
        )
        for key, value in frontend_kwargs.items():
            extra_flags = value.pop("flags", [])
            frontend_group.add_argument(
                *extra_flags, f"--{key.replace('_', '-')}", **value
            )

        return parser


@config
class FrontendArgs(BaseFrontendArgs):
    """Arguments for the OpenAI-compatible frontend server."""

    host: str | None = None
    """Host name."""
    port: int = 8000
    """Port number."""
    grpc_port: int | None = None
    """Enable the Rust frontend's additional gRPC Inference and Control services
    on this port. Requires `VLLM_USE_RUST_FRONTEND=1 vllm serve`; HTTP remains on
    `--port`. Binds on `--host`, or 127.0.0.1 with `--uds`. Cannot be combined
    with the Python gRPC server's `--grpc` flag."""
    data_parallel_supervisor_port: int = 9256
    """HTTP port for aggregated health endpoints in multi-port external LB
    mode."""
    dp_supervisor_probe_interval_s: float = 5.0
    """Seconds between aggregated health probes in multi-port external LB mode."""
    dp_supervisor_probe_timeout_s: float = 5.0
    """Seconds to wait between retries when a child health probe fails with a
    connection error in multi-port external LB mode."""
    dp_supervisor_probe_failure_threshold: int = 3
    """Number of consecutive connection-error retries before a child health
    probe is declared failed in multi-port external LB mode."""
    uds: str | None = None
    """Unix domain socket path. If set, host and port arguments are ignored."""
    uvicorn_log_level: Literal[
        "critical", "error", "warning", "info", "debug", "trace"
    ] = "info"
    """Log level for uvicorn."""
    disable_uvicorn_access_log: bool = False
    """Disable uvicorn access log."""
    disable_access_log_for_endpoints: str | None = None
    """Comma-separated list of endpoint paths to exclude from uvicorn access
    logs. This is useful to reduce log noise from high-frequency endpoints
    like health checks. Example: "/health,/metrics,/ping".
    When set, access logs for requests to these paths will be suppressed
    while keeping logs for other endpoints."""
    allow_credentials: bool = False
    """Allow credentials."""
    allowed_origins: list[str] = field(default_factory=list)
    """Allowed origins. Cross-origin browser access is disabled by default."""
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    """Allowed methods."""
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    """Allowed headers."""
    api_key: list[str] | None = None
    """If provided, the server will require one of these keys to be presented in
    the header.

    Warning: this only authenticates endpoints under the `/v1`, `/v2`, and
    `/inference` path prefixes. Other endpoints on the same server, including
    `/invocations` (which exposes the same inference capabilities as `/v1`),
    remain unauthenticated. Do not rely on `--api-key` alone to secure vLLM;
    see
    https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations
    for what it does and does not protect."""
    ssl_keyfile: str | None = None
    """The file path to the SSL key file."""
    ssl_certfile: str | None = None
    """The file path to the SSL cert file."""
    ssl_ca_certs: str | None = None
    """The CA certificates file."""
    enable_ssl_refresh: bool = False
    """Refresh SSL Context when SSL certificate files change"""
    ssl_cert_reqs: int = int(ssl.CERT_NONE)
    """Whether client certificate is required (see stdlib ssl module's)."""
    ssl_ciphers: str | None = None
    """SSL cipher suites for HTTPS (TLS 1.2 and below only).
    Example: 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305'"""
    root_path: str | None = None
    """FastAPI root_path when app is behind a path based routing proxy."""
    middleware: list[str] = field(default_factory=lambda: [])
    """Additional ASGI middleware to apply to the app. We accept multiple
    --middleware arguments. The value should be an import path. If a function
    is provided, vLLM will add it to the server using
    `@app.middleware('http')`. If a class is provided, vLLM will
    add it to the server using `app.add_middleware()`."""
    enable_request_id_headers: bool = False
    """If specified, API server will add X-Request-Id header to responses."""
    disable_fastapi_docs: bool = False
    """Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."""
    h11_max_incomplete_event_size: int = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    """Maximum size (bytes) of an incomplete HTTP event (header or body) for
    h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB)."""
    h11_max_header_count: int = H11_MAX_HEADER_COUNT_DEFAULT
    """Maximum number of HTTP headers allowed in a request for h11 parser.
    Helps mitigate header abuse. Default: 256."""
    enable_offline_docs: bool = False
    """
    Enable offline FastAPI documentation for air-gapped environments.
    Uses vendored static assets bundled with vLLM.
    """
    enable_flash_late_interaction: bool = True
    """If set, run pooling score MaxSim on GPU in the API server process.
    Can significantly improve late-interaction scoring performance.
    When disabled, the setting is also propagated to the engine's
    `PoolerConfig`, so the engine-side scorer uses the reference MaxSim
    path instead of the fused Triton kernel."""

    @classmethod
    def _customize_cli_kwargs(
        cls,
        frontend_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        frontend_kwargs = super()._customize_cli_kwargs(frontend_kwargs)

        # Special case: allowed_origins, allowed_methods, allowed_headers all
        # need json.loads type
        # Should also remove nargs
        frontend_kwargs["allowed_origins"]["type"] = json.loads
        frontend_kwargs["allowed_methods"]["type"] = json.loads
        frontend_kwargs["allowed_headers"]["type"] = json.loads
        del frontend_kwargs["allowed_origins"]["nargs"]
        del frontend_kwargs["allowed_methods"]["nargs"]
        del frontend_kwargs["allowed_headers"]["nargs"]

        # Special case: Middleware needs to append action
        frontend_kwargs["middleware"]["action"] = "append"
        frontend_kwargs["middleware"]["type"] = str
        if "nargs" in frontend_kwargs["middleware"]:
            del frontend_kwargs["middleware"]["nargs"]
        frontend_kwargs["middleware"]["default"] = []

        # Special case: disable_access_log_for_endpoints is a single
        # comma-separated string, not a list
        if "nargs" in frontend_kwargs["disable_access_log_for_endpoints"]:
            del frontend_kwargs["disable_access_log_for_endpoints"]["nargs"]

        return frontend_kwargs


def resolve_default_chat_template_kwargs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve renderer defaults, including the dedicated Cohere format flag."""
    kwargs = dict(args.default_chat_template_kwargs or {})
    if getattr(args, "cohere_format", None):
        kwargs.setdefault("cohere_format", args.cohere_format)
    return kwargs


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server.

    We rely on the helper methods of `FrontendArgs` and `AsyncEngineArgs` to
    register all arguments instead of manually enumerating them here. This
    avoids code duplication and keeps the argument definitions in one place.
    """
    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=None,
        help="How many API server processes to run. "
        "Defaults to data_parallel_size if not specified.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html",
    )
    parser.add_argument(
        "--grpc",
        action="store_true",
        default=False,
        help="Launch the Python gRPC (SMG VllmEngine) server on --port instead "
        "of the HTTP OpenAI-compatible server. Requires: pip install vllm[grpc]. "
        "Cannot be combined with --grpc-port, which enables the Rust "
        "frontend's gRPC services.",
    )
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


def validate_grpc_port_arg(args: argparse.Namespace) -> None:
    """Validate the Rust frontend's optional gRPC listener."""
    if getattr(args, "grpc_port", None) is None:
        return

    if args.grpc:
        raise ValueError(
            "--grpc and --grpc-port are mutually exclusive. Use --grpc "
            "--port for the Python gRPC server, or VLLM_USE_RUST_FRONTEND=1 "
            "with --grpc-port for the Rust frontend."
        )
    if not envs.VLLM_USE_RUST_FRONTEND or getattr(args, "subparser", None) != "serve":
        raise ValueError("--grpc-port requires VLLM_USE_RUST_FRONTEND=1 vllm serve")
    if args.headless or (
        args.api_server_count is not None and args.api_server_count <= 0
    ):
        raise ValueError(
            "--grpc-port requires a Rust frontend; remove --headless "
            "and non-positive --api-server-count"
        )
    if args.data_parallel_multi_port_external_lb:
        raise ValueError(
            "--grpc-port is incompatible with "
            "--data-parallel-multi-port-external-lb: "
            "its frontends would share the same gRPC port"
        )
    if not 0 <= args.grpc_port <= 65535:
        raise ValueError("--grpc-port must be between 0 and 65535")
    if args.grpc_port != 0 and args.grpc_port == args.port and not args.uds:
        raise ValueError("--grpc-port must differ from --port when HTTP uses TCP")


def validate_reasoning_effort_map(effort_map: Any) -> None:
    if effort_map is None:
        return

    efforts = get_args(ReasoningEffort)
    if not isinstance(effort_map, dict) or not all(
        isinstance(value, str) for value in effort_map.values()
    ):
        raise ValueError(
            "Error: --reasoning-effort-map must be a JSON object mapping "
            "reasoning efforts to strings."
        )
    if unknown := sorted(set(effort_map) - set(efforts)):
        raise ValueError(
            f"Error: --reasoning-effort-map has unknown efforts {unknown}; "
            f"keys must be one of {list(efforts)}."
        )


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    # `vllm launch <component>` builds its parser with make_arg_parser too (see
    # LaunchSubcommandBase.add_cli_args), so its args are serve args as well.
    if hasattr(args, "subparser") and args.subparser not in ("serve", "launch"):
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)
    if args.chat_template is not None and "hf" in (
        args.tool_call_parser,
        getattr(args, "reasoning_parser", None),
    ):
        logger.warning(
            "--chat-template is set; the hf parser still expects "
            "the checkpoint's output format."
        )

    validate_grpc_port_arg(args)
    validate_reasoning_effort_map(getattr(args, "reasoning_effort_map", None))

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
    if args.enable_log_outputs and not args.enable_log_requests:
        raise TypeError("Error: --enable-log-outputs requires --enable-log-requests")

    # SSE keep-alive interval must be zero (disabled) or a positive integer.
    if getattr(args, "sse_keep_alive_interval", 0) < 0:
        raise ValueError(
            "Error: --sse-keep-alive-interval must be a non-negative integer "
            "(0 disables keep-alive comments)."
        )

    if getattr(args, "enable_per_request_metrics", False) and getattr(
        args, "disable_log_stats", False
    ):
        raise ValueError(
            "Error: --enable-per-request-metrics requires engine statistics "
            "logging; remove --disable-log-stats to enable per-request metrics."
        )

    if args.data_parallel_multi_port_external_lb:
        from vllm.entrypoints.launchers.dp_supervisor import (
            validate_multi_port_external_lb_args,
        )

        validate_multi_port_external_lb_args(args)


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m vllm.entrypoints.launchers.api_server.entry"
    )
    return make_arg_parser(parser_for_docs)


def propagate_flash_late_interaction(args, engine_args) -> None:
    """Propagate `--disable-flash-late-interaction` into the engine config.

    The frontend flag alone only disables the API-server scoring path;
    mirroring it into `PoolerConfig.enable_flash_late_interaction` lets the
    engine-side scorer fall back to the reference MaxSim path too.
    """
    if getattr(args, "enable_flash_late_interaction", True):
        return
    from vllm.config.pooler import PoolerConfig

    if engine_args.pooler_config is None:
        engine_args.pooler_config = PoolerConfig(enable_flash_late_interaction=False)
    else:
        engine_args.pooler_config.enable_flash_late_interaction = False


def init_parser_plugin(args: argparse.Namespace):
    from vllm.reasoning import ReasoningParserManager

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)


def validate_api_server_args(args: argparse.Namespace):
    from vllm.reasoning import ReasoningParserManager

    valid_tool_parses = ToolParserManager.list_registered()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )
