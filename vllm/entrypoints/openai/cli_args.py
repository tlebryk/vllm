# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Literal

import vllm.envs as envs
from vllm.config import config
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    validate_chat_template,
)
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager
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
                name, path = item.split("=")
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
    default_chat_template_kwargs: dict[str, Any] | None = None
    """Default keyword arguments to pass to the chat template renderer.
    These will be merged with request-level chat_template_kwargs,
    with request values taking precedence. Useful for setting default
    behavior for reasoning models. Example: '{"enable_thinking": false}'
    to disable thinking mode by default for Qwen3/DeepSeek models."""
    response_role: str = "assistant"
    """The role name to return if `request.add_generation_prompt=true`."""
    return_tokens_as_token_ids: bool = False
    """When `--max-logprobs` is specified, represents single tokens as
    strings of the form 'token_id:{token_id}' so that tokens that are not
    JSON-encodable can be identified."""
    disable_frontend_multiprocessing: bool = False
    """If specified, will run the OpenAI frontend server in the same process as
    the model serving engine."""
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
    tool_server: str | None = None
    """Comma-separated list of host:port pairs (IPv4, IPv6, or hostname).
    Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for
    built-in demo tools (browser and Python code interpreter). WARNING:
    The `demo` Python tool executes model-generated code in Docker without
    network isolation by default. See the security guide for more
    information."""
    log_config_file: str | None = envs.VLLM_LOGGING_CONFIG_PATH
    """Path to logging config JSON file for both vllm and uvicorn"""
    max_log_len: int | None = None
    """Max number of prompt characters or prompt ID numbers being printed in
    log. The default of None means unlimited."""
    enable_prompt_tokens_details: bool = False
    """If set to True, enable prompt_tokens_details in usage."""
    enable_server_load_tracking: bool = False
    """If set to True, enable tracking server_load_metrics in the app state."""
    enable_force_include_usage: bool = False
    """If set to True, including usage on every request."""
    enable_tokenizer_info_endpoint: bool = False
    """Enable the `/tokenizer_info` endpoint. May expose chat
    templates and other tokenizer configuration."""
    enable_log_outputs: bool = False
    """If set to True, log model outputs (generations).
    Requires `--enable-log-requests`. As with `--enable-log-requests`,
    information is only logged at INFO level at maximum."""
    enable_log_deltas: bool = True
    """If set to False, output deltas will not be logged. Relevant only if 
    --enable-log-outputs is set.
    """
    log_error_stack: bool = envs.VLLM_SERVER_DEV_MODE
    """If set to True, log the stack trace of error responses"""
    tokens_only: bool = False
    """
    If set to True, only enable the Tokens In<>Out endpoint. 
    This is intended for use in a Disaggregated Everything setup.
    """
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
    slackserve_smctrl_max_dense_wait_ms: float = 0
    """Restore full-GPU dense access once its oldest request reaches this age.

    Zero disables the age guard. Generation prefill retains its decode gate.
    """
    slackserve_smctrl_min_decode_requests: int = 1
    """Apply the prefill mask only at this many decode-ready requests.

    This controls placement, not admission or scheduler batch capacity.
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
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    """Allowed origins."""
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    """Allowed methods."""
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    """Allowed headers."""
    api_key: list[str] | None = None
    """If provided, the server will require one of these keys to be presented in
    the header."""
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
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "serve":
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
    if args.enable_log_outputs and not args.enable_log_requests:
        raise TypeError("Error: --enable-log-outputs requires --enable-log-requests")


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m vllm.entrypoints.openai.api_server"
    )
    return make_arg_parser(parser_for_docs)
