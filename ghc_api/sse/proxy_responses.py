"""Streaming handler for configured-proxy OpenAI Responses API traffic."""

import json
from typing import Dict, Iterator

from .openai_responses import OpenAIResponsesStreamHandler


class ProxyResponsesStreamHandler(OpenAIResponsesStreamHandler):
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
