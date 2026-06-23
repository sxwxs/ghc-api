"""OpenAI Responses (``/v1/responses``) SSE stream handler.

Pure passthrough. The upstream stream uses both ``event:`` and ``data:`` SSE
lines; we forward the ``event:`` line verbatim (single newline) and the
``data:`` JSON unchanged (double newline). Matches the wire format the
pre-refactor ``stream_responses`` produced.
"""

from typing import Dict

from .base import SSEStreamHandler


class OpenAIResponsesStreamHandler(SSEStreamHandler):
    endpoint = "/v1/responses"
    log_prefix = "[Stream Responses]"
    # /v1/responses sends an ``event: TYPE`` line *before* each ``data:`` line.
    # We pass the event header through verbatim and emit only the data line
    # ourselves (the original handler's convention).
    emit_event_header = False

    def on_event(self, event_type: str, event: Dict) -> None:
        if event_type == "response.completed":
            resp = event.get("response", {}) or {}
            usage = resp.get("usage", {}) or {}
            self.input_tokens = usage.get("input_tokens", 0)
            self.output_tokens = usage.get("output_tokens", 0)
            details = usage.get("input_tokens_details", {}) or {}
            self.cache_creation_input_tokens = details.get("cached_tokens", 0)
