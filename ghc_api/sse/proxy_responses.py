"""Streaming handler for configured-proxy OpenAI Responses API traffic."""

import json
from typing import Dict, Iterator

from ..utils import responses_usage_counts
from .base import SSEStreamHandler


class ProxyResponsesStreamHandler(SSEStreamHandler):
    log_prefix = "[Configured Proxy Responses]"
    emit_event_header = False

    def __init__(
        self,
        *args,
        endpoint: str,
        profile_name: str,
        public_model: str,
        rewrite_model: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint
        self.profile_name = profile_name
        self.public_model = public_model
        self.rewrite_model = rewrite_model

    def on_event(self, event_type: str, event: Dict) -> None:
        if event_type in ("response.completed", "response.incomplete"):
            response = event.get("response", {}) or {}
            counts = responses_usage_counts(response.get("usage"))
            self.input_tokens = counts["input_tokens"]
            self.output_tokens = counts["output_tokens"]
            self.cache_creation_input_tokens = counts["cache_creation_input_tokens"]
            self.cache_read_input_tokens = counts["cache_read_input_tokens"]
        elif event_type == "response.failed":
            self.error_occurred = True
            self.status_code = 502

    def forward_event(
        self, event_type: str, event: Dict, raw_data: str
    ) -> Iterator[tuple]:
        if not self.rewrite_model:
            yield event_type, raw_data
            return

        response = event.get("response")
        if not isinstance(response, dict) or "model" not in response:
            yield event_type, raw_data
            return

        rewritten = dict(event)
        rewritten_response = dict(response)
        rewritten_response["model"] = self.public_model
        rewritten["response"] = rewritten_response
        yield event_type, json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))

    def extra_cache_fields(self) -> Dict:
        return {
            "upstream_provider": "configured_proxy",
            "upstream_profile": self.profile_name,
            "upstream_api": "responses",
        }
