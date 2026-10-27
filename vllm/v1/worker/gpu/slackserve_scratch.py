# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit stream ownership for sparse-attention model scratch, not weights."""

from contextlib import contextmanager
from copy import copy

import torch


class ModelScratch:
    def __init__(self, model):
        self.bindings = []
        self._tensors = {}
        self._groups = {}
        seen = set()
        for module in model.modules():
            for owner in (module, getattr(module, "impl", None)):
                if owner is None or id(owner) in seen:
                    continue
                seen.add(id(owner))
                for key in ("topk_indices_buffer", "_topk_indices_buffer"):
                    # Do not assign SharedTopkIndicesBuffer's property: its
                    # setter disconnects the live indexer alias.
                    if key not in vars(owner) and key not in getattr(
                        owner, "_buffers", {}
                    ):
                        continue
                    original = getattr(owner, key)
                    if isinstance(original, torch.Tensor):
                        self.bindings.append(
                            (owner, key, original, self._clone(original))
                        )
                group = getattr(owner, "index_group", None)
                if group is not None:
                    self.bindings.append(
                        (owner, "index_group", group, self._clone_group(group))
                    )

    def _clone(self, tensor):
        if id(tensor) not in self._tensors:
            self._tensors[id(tensor)] = torch.empty_like(tensor)
        return self._tensors[id(tensor)]

    def _clone_group(self, group):
        from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroup

        if type(group) is not SparseMLAIndexGroup:
            raise ValueError(f"Unvalidated sparse index group: {type(group).__name__}")
        if id(group) not in self._groups:
            private = copy(group)
            private.logical_topk_indices = self._clone(group.logical_topk_indices)
            private.physical_topk_indices = self._clone(group.physical_topk_indices)
            private.valid_topk_counts = self._clone(group.valid_topk_counts)
            private.side_stream = torch.Stream(device=group.logical_topk_indices.device)
            private.logical_topk_ready = torch.Event()
            private.physical_topk_ready = torch.Event()
            self._groups[id(group)] = private
        return self._groups[id(group)]

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self._tensors.values())

    @contextmanager
    def use(self, stream=None):
        # Prefill's sparse-index conversion must obey the same ticket mask.
        # Running that helper on the current prefill stream also makes its
        # completion part of the ticket's existing FIFO/event dependency.
        if stream is not None:
            for group in self._groups.values():
                group.side_stream = stream
        for owner, key, _, private in self.bindings:
            setattr(owner, key, private)
        try:
            yield
        finally:
            for owner, key, original, _ in reversed(self.bindings):
                setattr(owner, key, original)
