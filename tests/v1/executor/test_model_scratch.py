# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse index aliases stay shared within a stream and isolated between streams."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/slackserve_scratch.py"
spec = importlib.util.spec_from_file_location("model_scratch_under_test", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ModelScratch = module.ModelScratch


def test_indexer_and_attention_keep_same_private_buffer_and_restore_on_failure():
    model = torch.nn.Module()
    model.indexer = torch.nn.Module()
    model.attention = torch.nn.Module()
    original = torch.full((4, 8), 17, dtype=torch.int32)
    model.indexer.topk_indices_buffer = original
    model.attention.register_buffer("_topk_indices_buffer", original)
    scratch = ModelScratch(model)
    assert scratch.bytes == original.numel() * original.element_size()
    with pytest.raises(RuntimeError, match="interrupt"), scratch.use():
        private = model.indexer.topk_indices_buffer
        assert private is model.attention._topk_indices_buffer
        assert private.data_ptr() != original.data_ptr()
        private.fill_(43)
        assert torch.all(original == 17)
        raise RuntimeError("interrupt")
    assert model.indexer.topk_indices_buffer is original
    assert model.attention._topk_indices_buffer is original
    with scratch.use():
        assert model.indexer.topk_indices_buffer is private
        assert torch.all(private == 43)
