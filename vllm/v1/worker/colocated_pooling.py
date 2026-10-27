# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooling model lifecycle inside an existing generation TP worker/context."""

from contextlib import contextmanager


@contextmanager
def pooling_scope(owner, worker):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state

    previous = getattr(parallel_state._TP_OVERRIDE, "group", None)
    parallel_state._TP_OVERRIDE.group = owner._prefill_tp_group
    try:
        with (
            set_current_vllm_config(worker.vllm_config),
            torch.cuda.stream(worker.prefill_stream),
        ):
            yield
    finally:
        parallel_state._TP_OVERRIDE.group = previous


def initialize(owner, model_id, config):
    from vllm.config import set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.utils.mem_utils import MemorySnapshot
    from vllm.v1.worker.slackserve_pooling_worker import SlackServePoolingWorker

    workers = getattr(owner, "_colocated_pooling", None)
    if workers is None:
        workers = owner._colocated_pooling = {}
    if model_id in workers:
        raise ValueError(f"duplicate colocated model {model_id}")
    if config.model_config.runner_type != "pooling":
        raise ValueError("only pooling models may use the colocated executor")
    if config.cache_config.kv_cache_memory_bytes is None:
        raise ValueError("colocated pooling initially requires explicit private KV")
    if (
        config.parallel_config.tensor_parallel_size
        != owner.parallel_config.tensor_parallel_size
    ):
        raise ValueError("colocated models require identical TP sizes")
    if config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("colocated pooling does not support pipeline parallelism")
    if config.kv_transfer_config is not None or config.model_config.enable_sleep_mode:
        raise ValueError("colocated pooling excludes KV connectors and sleep mode")
    # Reuse the device, context, distributed groups and workspace manager.
    # Calling init_device() again would reinitialize process-global resources.
    with set_current_vllm_config(config):
        worker = SlackServePoolingWorker(
            config,
            owner.local_rank,
            owner.rank,
            owner.distributed_init_method,
            owner.is_driver_worker,
        )
        worker.use_v2_model_runner = False
        worker.device = owner.device
        worker.prefill_stream = owner.lane_streams["prefill"]
        worker.init_snapshot = MemorySnapshot(device=owner.device)
        worker.requested_memory = config.cache_config.kv_cache_memory_bytes
        current_platform.check_if_supports_dtype(config.model_config.dtype)
        with pooling_scope(owner, worker):
            worker.model_runner = worker._create_model_runner()
            worker.load_model()
        current_platform.update_block_size_for_backend(config)
    workers[model_id] = worker
    return dict(
        model_id=model_id,
        rank=owner.rank,
        device=str(owner.device),
        prefill_stream=worker.prefill_stream.cuda_stream,
        same_context=True,
    )


def dispatch(owner, model_id, method, args, kwargs):
    worker = owner._colocated_pooling[model_id]
    if method == "shutdown":
        raise ValueError("auxiliary model cannot shut down shared worker resources")
    if method == "initialize_from_config":
        # Executor initialization broadcasts the per-rank list, normally
        # selected by WorkerWrapperBase before calling the GPU worker.
        args = (args[0][owner.rank],) + tuple(args[1:])
    if method == "execute_model":
        ticket = args[0]
        if getattr(ticket, "execution_lane", "default") == "decode":
            raise ValueError("pooling model cannot receive a decode ticket")
        # The prefill stream is physically shared: use the owner's mask cache,
        # so a generator ticket also restores its mask after a verifier ticket.
        owner._apply_ticket_mask("prefill", getattr(ticket, "execution_mask", None))
    with pooling_scope(owner, worker):
        if isinstance(method, str):
            return getattr(worker, method)(*args, **kwargs)
        return method(worker, *args, **kwargs)


def drop(owner, model_id):
    worker = getattr(owner, "_colocated_pooling", {}).pop(model_id, None)
    if worker is not None:
        worker.prefill_stream.synchronize()
        # Releasing only this runner leaves the owner's process/communicators
        # alive. Full process teardown remains the generation executor's job.
        worker.model_runner = None
