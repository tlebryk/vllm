# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pooling engine using the generation executor's existing TP processes.

Separate executors/processes would create separate CUDA contexts and lose
kernel overlap. This proxy owns only the auxiliary model, never the transport.
"""

from vllm.v1.executor.abstract import Executor


def colocated_pooling_executor(parent, model_id):
    """Bind one engine-local executor class to its already initialized owner."""

    class ColocatedPoolingExecutor(Executor):
        def _init_executor(self):
            self._closed = False
            parent.collective_rpc(
                "init_colocated_pooling", args=(model_id, self.vllm_config)
            )

        def collective_rpc(
            self, method, timeout=None, args=(), kwargs=None, non_block=False
        ):
            if self._closed:
                raise RuntimeError("colocated pooling executor is closed")
            return parent.collective_rpc(
                "colocated_pooling_rpc",
                args=(model_id, method, args, kwargs or {}),
                timeout=timeout,
                non_block=non_block,
            )

        def execute_model(self, scheduler_output, non_block=False):
            if self._closed:
                raise RuntimeError("colocated pooling executor is closed")
            # Preserve the owner's sequence-numbered async output transport.
            # A pooling completion event is retired on its output thread.
            return parent.collective_rpc(
                "colocated_pooling_rpc",
                args=(model_id, "execute_model", (scheduler_output,), {}),
                non_block=non_block,
                unique_reply_rank=parent.output_rank,
            )

        def take_draft_token_ids(self):
            return None

        def check_health(self):
            parent.check_health()

        def shutdown(self):
            if not self._closed:
                self._closed = True
                # Owner shutdown controls NCCL/process lifetime. Do not call
                # the pooled Worker's global KV-transfer teardown here.
                if not getattr(parent, "is_failed", False):
                    parent.collective_rpc("drop_colocated_pooling", args=(model_id,))

    return ColocatedPoolingExecutor
