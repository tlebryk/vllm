# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned decode streams with an optional GPU-only ordering control."""


class ModelDecodeStreams:
    """Order launches through immutable events without blocking the CPU.

    Both modes record the same events and retain the same launch-ahead quota.
    The ordered mode adds only an inter-model wait on the previous decode
    ticket, including sampling/D2H enqueued synchronously by dispatch.
    Per-model CUDA stream order is sufficient for consecutive same-model work.
    """

    def __init__(self, streams, event_factory, *, ordered=False):
        self.streams = streams
        self.event_factory = event_factory
        self.ordered = ordered
        self.last_event = None
        self.last_model = None

    def before_dispatch(self, model, lane):
        if (
            lane == "decode"
            and self.ordered
            and self.last_event is not None
            and self.last_model != model.model_id
        ):
            self.streams[model.model_id].wait_event(self.last_event)

    def after_dispatch(self, model, lane):
        if lane == "decode":
            event = self.event_factory()
            event.record(self.streams[model.model_id])
            self.last_event, self.last_model = event, model.model_id
