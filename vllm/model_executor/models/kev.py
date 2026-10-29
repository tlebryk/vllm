# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kev decision model (github.com/jaredpalmer/kev) as a vLLM pooling model.

One request is one Kev question row:
<state> state <q> instructions (<opt> option </opt>)+ <decide>,
token ids built by the client (slackserve.kev.rows, the layout of
kev.model.encode). The backbone has Kev's LoRA folded in
(slackserve.kev.export_vllm): Qwen3 (KevQwen3DecisionModel, kev-4b@qwen3)
or Qwen3.5 (KevDecisionModel). The pooler keeps the final hidden states at
every </opt> and at <decide> (the last prompt token), across prefill
chunks, and returns softmax(k(h_opt) . q(h_decide) / sqrt(dp) / T) over
the options.
"""

from collections.abc import Iterable, Set

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen3 import Qwen3ForCausalLM
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor


class KevPointerHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, temperature: float):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.k = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.scale = head_dim**-0.5
        self.temperature = temperature

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale / self.temperature
        return torch.softmax(z, -1)


class KevPointerPooler(Pooler):
    def __init__(self, head: KevPointerHead, opt_end_id: int, decide_id: int):
        super().__init__()
        self.head = head
        self.opt_end_id = opt_end_id
        self.decide_id = decide_id

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> PoolerOutput:
        cursor = pooling_metadata.get_pooling_cursor()
        token_ids = pooling_metadata.get_prompt_token_ids()
        firsts = cursor.first_token_indices_gpu.tolist()
        out: list[torch.Tensor | None] = []
        for i, state in enumerate(pooling_metadata.pooling_states):
            prompt_len = int(cursor.prompt_lens_cpu[i])
            end = int(cursor.seq_lens_cpu[i])
            start = end - int(cursor.num_scheduled_tokens_cpu[i])
            ids = token_ids[i][start:end]
            keep = ids == self.opt_end_id
            if end == prompt_len:
                keep[-1] = True
            rows = keep.nonzero().squeeze(-1)
            if rows.numel():
                state.hidden_states_cache.append(
                    hidden_states[firsts[i] + rows].float()
                )
            if end != prompt_len:
                out.append(None)
                continue
            picked = torch.cat(state.hidden_states_cache, 0)
            state.clean()
            if int(token_ids[i][prompt_len - 1]) != self.decide_id or len(picked) < 2:
                # not a Kev row (the engine's dummy profiling run): NaN, which
                # clients reject as a nonfinite score
                out.append(picked.new_full((1,), float("nan")))
                continue
            out.append(self.head(picked[-1], picked[:-1]))
        return out


class _KevHeadMixin:
    """Pointer head + pooler from config.kev; kev_head.* weights load
    beside the backbone."""

    def _init_kev(self, config, hidden_size: int) -> None:
        kev = config.kev
        self.kev_head = KevPointerHead(hidden_size, kev["head_dim"], kev["temperature"])
        self.pooler = KevPointerPooler(
            self.kev_head,
            kev["special_ids"]["opt_end"],
            kev["special_ids"]["decide"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        head, rest = [], []
        for name, tensor in weights:
            (head if name.startswith("kev_head.") else rest).append((name, tensor))
        loaded = super().load_weights(rest)
        params = dict(self.kev_head.named_parameters())
        for name, tensor in head:
            key = name.removeprefix("kev_head.")
            params[key].data.copy_(tensor.to(params[key].dtype))
            loaded.add(name)
        return loaded


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class KevQwen3DecisionModel(_KevHeadMixin, Qwen3ForCausalLM):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self._init_kev(config, config.hidden_size)


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class KevDecisionModel(_KevHeadMixin, Qwen3_5ForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self._init_kev(config, config.text_config.hidden_size)
