"""Direct Anthropic ``/v1/messages`` SSE stream handler.

Two classes:

- :class:`AnthropicDirectStreamHandler` -- pure passthrough. The base class
  records every upstream ``data:`` line into ``raw_events``; this subclass only
  extracts usage tokens from ``message_start`` / ``message_delta``. It does
  **not** import ``tool_call_recovery``.

- :class:`AnthropicDirectStreamHandlerWithRecovery` -- subclass that adds the
  :class:`LeakedToolCallTransformer` to recover tool calls Copilot sometimes
  leaks as text. Selected by the route factory when
  ``state.enable_tool_call_recovery`` is True; otherwise the base class is used
  directly so the disabled path is byte-identical to a world without the feature.
"""

import json
from typing import Any, Dict, Iterator, List

from .base import SSEStreamHandler


class AnthropicDirectStreamHandler(SSEStreamHandler):
    """Passthrough for direct Anthropic ``/v1/messages`` streaming.

    The wire output is byte-equal to the upstream stream (modulo line framing
    handled by the base class). Cache stores the raw event list.
    """

    endpoint = "/v1/messages"
    log_prefix = "[Stream Direct Anthropic]"
    emit_event_header = True
    # Anthropic /v1/messages signals end-of-stream via ``message_stop``; the
    # bare ``[DONE]`` sentinel is OpenAI-specific and would break clients that
    # try to JSON-decode every SSE data line.
    emit_done_sentinel = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Tracked for the cache record. Defaults to the requested model; updated
        # when ``message_start`` arrives.
        self.accumulated_model: str = self.original_model

    def on_event(self, event_type: str, event: Dict) -> None:
        if event_type == "message_start":
            msg = event.get("message", {}) or {}
            self.accumulated_model = msg.get("model", self.original_model)
            usage = msg.get("usage", {}) or {}
            self.input_tokens = usage.get("input_tokens", 0)
            self.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
            self.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
        elif event_type == "message_delta":
            usage = event.get("usage", {}) or {}
            self.output_tokens = usage.get("output_tokens", 0)

    def _format_generic_error(self, e: Exception) -> str:
        # Match the existing handler: emit ``event: error\ndata: {...}\n\n``.
        error_event = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
        return f"event: error\ndata: {json.dumps(error_event)}\n\n"


class AnthropicDirectStreamHandlerWithRecovery(AnthropicDirectStreamHandler):
    """Adds leaked-tool-call recovery on top of the passthrough handler.

    Importing this class pulls in ``tool_call_recovery``. The route factory must
    only reference this class when ``state.enable_tool_call_recovery`` is True;
    otherwise prefer the parent class so recovery code is never loaded.
    """

    log_prefix = "[Stream Direct Anthropic +recovery]"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Imported lazily so module-level import of this file at least defers
        # the recovery module load -- not strictly necessary but emphasizes
        # that recovery state is owned only by this subclass.
        from ..tool_call_recovery import LeakedToolCallTransformer

        self._transformer = LeakedToolCallTransformer(enabled=True)

    def forward_event(
        self, event_type: str, event: Dict, raw_data: str
    ) -> Iterator[tuple]:
        for out_type, out_data in self._transformer.process(event_type, event, raw_data):
            yield (out_type, out_data)

    def finalize_stream(self) -> Iterator[tuple]:
        for out_type, out_data in self._transformer.finalize():
            yield (out_type, out_data)

    def extra_cache_fields(self) -> Dict[str, Any]:
        # Surface what the recovery layer produced as a sidecar field so it can
        # be inspected without mutating raw_events.
        return {"recovered_content": self._transformer.build_response_content()}
