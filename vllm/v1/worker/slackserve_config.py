# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, per-model lane settings; environment fallback for old harnesses."""

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlackServeConfig:
    model_id: str = "model"
    prefill_lanes: int = 1
    shared_prefill_stream: bool = False
    decode_graphs_only: bool = False
    compile_decode: bool = False
    compile_prefill: bool = False
    prep_streams: bool = False
    lane_staging: bool = False
    async_decode_depth: int = 1
    fa_sm_margin_decode: int = 0
    fa_sm_margin_prefill: int = 0
    prefill_custom_ops: str = ""
    prefill_step_tokens: int = 0
    prefill_sm_target: int = 0
    sm_target_min_m: int = 256
    # Explicit multi-model configurations use controller-owned lane masks.
    legacy_env_masks: bool = False
    embed_sidecar: bool = False
    dense_lock: bool = True

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("SlackServe model_id must be nonempty")
        if self.prefill_lanes not in (1, 2):
            raise ValueError("SlackServe prefill_lanes must be 1 or 2")
        if not 1 <= self.async_decode_depth <= 4:
            raise ValueError("SlackServe async_decode_depth must be between 1 and 4")
        if self.compile_decode and self.compile_prefill:
            raise ValueError("only one lane may own the compiled callable")
        if (
            min(
                self.fa_sm_margin_decode,
                self.fa_sm_margin_prefill,
                self.prefill_sm_target,
                self.prefill_step_tokens,
            )
            < 0
            or self.sm_target_min_m < 1
        ):
            raise ValueError("invalid SlackServe SM target/margin or minimum M")

    def prefill_lane_names(self) -> list[str]:
        return ["prefill"] if self.prefill_lanes == 1 else ["prefill", "prefill1"]

    @classmethod
    def from_vllm_config(cls, config: Any) -> "SlackServeConfig":
        additional = config.additional_config
        if isinstance(additional, dict) and "slackserve" in additional:
            return cls(**additional["slackserve"])
        return cls.from_env()

    @classmethod
    def from_env(cls) -> "SlackServeConfig":
        """Read legacy settings once, never at a model switch."""

        def flag(name: str) -> bool:
            return os.environ.get(name) == "1"

        def integer(name: str, default: int = 0) -> int:
            return int(os.environ.get(name) or default)

        return cls(
            prefill_lanes=2 if os.environ.get("HB_P2_PREFILL_LANES") == "2" else 1,
            shared_prefill_stream=flag("HB_P2_PREFILL_SHARED_STREAM"),
            decode_graphs_only=flag("HB_P2_DECODE_GRAPHS_ONLY"),
            compile_decode=flag("HB_P2_COMPILE_DECODE"),
            compile_prefill=flag("HB_P2_COMPILE_PREFILL"),
            prep_streams=flag("HB_P2_PREP_STREAM"),
            lane_staging=flag("HB_P2_LANE_STAGING"),
            async_decode_depth=integer(
                "HB_P2_ASYNC_DECODE_DEPTH", 2 if flag("HB_P2_ASYNC_DECODE") else 1
            ),
            fa_sm_margin_decode=integer("HB_P2_FA_SM_MARGIN_DECODE"),
            fa_sm_margin_prefill=integer("HB_P2_FA_SM_MARGIN_PREFILL"),
            prefill_custom_ops=os.environ.get("HB_P2_PREFILL_CUSTOM_OPS", ""),
            legacy_env_masks=True,
            embed_sidecar=flag("HB_P2_EMBED_SIDECAR"),
            dense_lock=os.environ.get("HB_P2_DENSE_LOCK", "1") != "0",
        )
