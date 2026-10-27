# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Slack Serve GPU worker.

Keeps lane-specific stream routing out of vLLM's common GPU worker.
"""

import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.slackserve_config import SlackServeConfig


def prefill_lane_names() -> list[str]:
    """Prefill lane names for the configured depth (default one lane)."""
    if os.environ.get("HB_P2_PREFILL_LANES") == "2":
        return ["prefill", "prefill1"]
    return ["prefill"]


class SlackServeGPUWorker(GPUWorker):
    def init_colocated_pooling(self, model_id, config):
        from vllm.v1.worker.colocated_pooling import initialize

        return initialize(self, model_id, config)

    def colocated_pooling_rpc(self, model_id, method, args, kwargs):
        from vllm.v1.worker.colocated_pooling import dispatch

        return dispatch(self, model_id, method, args, kwargs)

    def drop_colocated_pooling(self, model_id):
        from vllm.v1.worker.colocated_pooling import drop

        drop(self, model_id)

    def configure_colocated_streams(self, serial=False):
        if serial:
            self.bind_lane_streams(self.decode_stream, self.decode_stream, serial=True)
        for worker in getattr(self, "_colocated_pooling", {}).values():
            worker.prefill_stream = self.lane_streams["prefill"]
        return dict(
            rank=self.rank,
            pid=os.getpid(),
            prefill=self.lane_streams["prefill"].cuda_stream,
            decode=self.decode_stream.cuda_stream,
            pooling_models=list(getattr(self, "_colocated_pooling", {})),
        )

    def init_device(self) -> None:
        self.slackserve_config = SlackServeConfig.from_vllm_config(self.vllm_config)
        super().init_device()
        self.decode_stream = torch.cuda.Stream(device=self.device)
        # HB_P2_PREFILL_SHARED_STREAM=1 maps every prefill lane onto ONE
        # CUDA stream: the extra lane then only provides a second in-flight
        # ticket context (host prep pipelined behind the running ticket)
        # while the GPU executes prefill tickets strictly in order - no
        # added SM contention, and same-request KV ordering comes free.
        shared = self.slackserve_config.shared_prefill_stream
        self.lane_streams: dict[str, torch.cuda.Stream] = {}
        for index, lane in enumerate(self.slackserve_config.prefill_lane_names()):
            if shared and index > 0:
                self.lane_streams[lane] = self.lane_streams["prefill"]
            else:
                self.lane_streams[lane] = torch.cuda.Stream(device=self.device)
        # Embed sidecar shares the prefill lane's stream; prefill-lane
        # launches and embed micro-steps are host-serialized by one FIFO
        # mutex (no kernel interleaving on the dense stream, no
        # forward-context race, launch-ahead comes free from the stream
        # FIFO). Runs on the executor thread — the controller never blocks.
        # Optional libsmctrl TPC placement masks (HB_SMCTRL_*). The embed
        # sidecar shares the prefill lane's stream, so the prefill mask
        # covers embed launches too.
        if self.slackserve_config.legacy_env_masks and os.environ.get("HB_SMCTRL_LIB"):
            from vllm.v1.worker.hb_smctrl import apply_env_masks

            apply_env_masks(
                {
                    "prefill": self.lane_streams["prefill"],
                    "decode": self.decode_stream,
                }
            )
        self._init_remote_lanes()
        self._hb_dense_lock: tuple[Callable[[], None], Callable[[], None]] | None = None
        self._hb_record_prefill_tail: Callable[[torch.cuda.Stream], None] | None = None
        if self.slackserve_config.embed_sidecar and self.slackserve_config.dense_lock:
            from vllm.v1.engine.hb_embed_sidecar import (
                prefill_launch_lock_acquire,
                prefill_launch_lock_release,
                record_prefill_tail,
            )

            self._hb_dense_lock = (
                prefill_launch_lock_acquire,
                prefill_launch_lock_release,
            )
            self._hb_record_prefill_tail = record_prefill_tail

    def _init_remote_lanes(self) -> None:
        """Workers in their own processes (TP>1) own the lane machinery the
        single-process controller otherwise provides: a second TP communicator
        for the prefill stream (one communicator driven from two streams
        deadlocks) and mask application from the ticket's TPC set.
        """
        from vllm.distributed import parallel_state

        self._prefill_tp_group = None
        self._mask_ctrl = None
        self._applied_mask: tuple[int, ...] | None = None
        tp = parallel_state.get_tp_group()
        if tp.world_size > 1:
            world = torch.distributed.get_world_size()
            groups = [
                list(range(start, start + tp.world_size))
                for start in range(0, world, tp.world_size)
            ]
            self._prefill_tp_group = parallel_state.init_model_parallel_group(
                groups,
                parallel_state.get_world_group().local_rank,
                "nccl",
                group_name="slackserve_prefill",
            )
        if self.slackserve_config.smctrl_lib:
            from vllm.v1.worker.hb_smctrl import SmCtrl

            self._mask_ctrl = SmCtrl(self.slackserve_config.smctrl_lib)

    def _remote_lane(self, lane: str) -> bool:
        return lane not in ("decode", "default") and (
            getattr(self, "_prefill_tp_group", None) is not None
            or getattr(self, "_mask_ctrl", None) is not None
        )

    def _set_tp_override(self, group) -> None:
        from vllm.distributed import parallel_state

        parallel_state._TP_OVERRIDE.group = group

    def _apply_ticket_mask(self, lane: str, mask: list[int] | None) -> None:
        if self._mask_ctrl is None:
            return
        wanted = tuple(mask) if mask else ()
        if wanted == self._applied_mask:
            return
        from vllm.v1.worker.hb_smctrl import _disable_mask_from_indices

        stream = self._lane_stream(lane)
        self._mask_ctrl._set_mask(
            stream,
            _disable_mask_from_indices(list(wanted)) if wanted else 0,
            f"TPCs {list(wanted)}" if wanted else "unrestricted",
            log=False,
        )
        self._applied_mask = wanted

    # override factory hook to use new model
    def _create_model_runner(self) -> Any:
        from vllm.v1.worker.gpu.slackserve_model_runner import (
            SlackServeModelRunner,
        )

        return SlackServeModelRunner(self.vllm_config, self.device)

    def bind_lane_streams(
        self,
        prefill_stream: torch.cuda.Stream,
        decode_stream: torch.cuda.Stream,
        *,
        serial: bool = False,
    ) -> None:
        """Bind an idle model to the controller's shared execution lanes.

        Call after initialization/capture and before admitting requests. Captured
        graphs keep this runner's addresses; replay uses the supplied stream.
        Explicit serial mode uses one stream for both phases while preserving
        separate input contexts and asynchronous output copies.
        """
        if prefill_stream.device != self.device or decode_stream.device != self.device:
            raise ValueError("shared lane streams must use the worker's device")
        same_stream = prefill_stream.cuda_stream == decode_stream.cuda_stream
        if serial and not same_stream:
            raise ValueError("serial lanes must use the same execution stream")
        if same_stream and not serial:
            raise ValueError("prefill and decode require different streams")
        if any(
            context.execute_model_state is not None
            for context in self.model_runner.contexts.values()
        ):
            raise RuntimeError("cannot rebind a worker with a pending forward")
        ready = torch.cuda.current_stream(self.device)
        prefill_stream.wait_stream(ready)
        decode_stream.wait_stream(ready)
        self.decode_stream = decode_stream
        self.lane_streams = {
            lane: prefill_stream for lane in self.slackserve_config.prefill_lane_names()
        }

    def _lane_stream(self, lane: str) -> torch.cuda.Stream:
        # ``default`` remains a compatibility alias for decode.
        if lane in ("decode", "default"):
            return self.decode_stream
        stream = self.lane_streams.get(lane)
        if stream is None:
            raise ValueError(f"Unknown execution lane: {lane!r}")
        return stream

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        lane = scheduler_output.execution_lane
        dense_lock = self._hb_dense_lock
        record_prefill_tail = self._hb_record_prefill_tail
        locked = dense_lock is not None and lane not in ("decode", "default")
        if (
            lane not in ("decode", "default")
            and self.slackserve_config.legacy_env_masks
        ):
            # Load-conditional prefill mask (HB_SMCTRL_MASK_MIN_RUNNING):
            # re-mask/uncap the prefill stream before this chunk's launches.
            from vllm.v1.worker.hb_smctrl import maybe_mask_prefill_for_load

            maybe_mask_prefill_for_load(self._lane_stream(lane))
        if locked:
            assert dense_lock is not None
            # NVTX spans the lock acquire too, so a trace shows prefill's
            # wait behind an embed unit as the gap from range-start to the
            # first prefill kernel.
            torch.cuda.nvtx.range_push("dense/prefill")
            dense_lock[0]()
        remote = self._remote_lane(lane)
        if remote:
            self._apply_ticket_mask(
                lane, getattr(scheduler_output, "execution_mask", None)
            )
            self._set_tp_override(self._prefill_tp_group)
        try:
            with (
                torch.cuda.stream(self._lane_stream(lane)),
                self.model_runner.uva_fence_registry.use_lane(lane),
            ):
                output = super().execute_model(scheduler_output)
                if output is None and scheduler_output.total_num_scheduled_tokens > 0:
                    # Enqueue sampling while this ticket's lane-local inputs
                    # and request slots are still current. The returned
                    # AsyncOutput is waited by UniProcExecutor off-thread, so
                    # dispatch remains nonblocking and another lane can launch.
                    output = self.model_runner.sample_tokens(
                        None, lane=lane, return_async=True
                    )
                if locked:
                    assert record_prefill_tail is not None
                    record_prefill_tail(self._lane_stream(lane))
                return output
        finally:
            if remote:
                self._set_tp_override(None)
            if locked:
                assert dense_lock is not None
                dense_lock[1]()
                torch.cuda.nvtx.range_pop()

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None", lane="decode"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # Sampling, prompt logprobs, the async D2H copy handoff, and
        # postprocess must all run on the same stream as the lane's forward.
        # AsyncOutput's stream() helper restores ``main_stream`` as the
        # ambient stream on exit, so running this on any other stream would
        # silently launch postprocess unordered w.r.t. the sampler.
        dense_lock = self._hb_dense_lock
        record_prefill_tail = self._hb_record_prefill_tail
        locked = dense_lock is not None and lane not in ("decode", "default")
        if locked:
            assert dense_lock is not None
            dense_lock[0]()
        remote = self._remote_lane(lane)
        if remote:
            self._set_tp_override(self._prefill_tp_group)
        try:
            with (
                torch.cuda.stream(self._lane_stream(lane)),
                self.model_runner.uva_fence_registry.use_lane(lane),
            ):
                output = self.model_runner.sample_tokens(grammar_output, lane=lane)
                if locked:
                    assert record_prefill_tail is not None
                    record_prefill_tail(self._lane_stream(lane))
                return output
        finally:
            if remote:
                self._set_tp_override(None)
            if locked:
                assert dense_lock is not None
                dense_lock[1]()
