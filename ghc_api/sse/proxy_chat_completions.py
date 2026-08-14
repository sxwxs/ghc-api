"""Streaming handler for configured-proxy OpenAI Chat Completions traffic."""

import json
from typing import Dict, Iterator

from .base import SSEStreamHandler


class ProxyChatCompletionsStreamHandler(SSEStreamHandler):
    log_prefix = "[Configured Proxy Chat Completions]"
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
        usage = event.get("usage")
        if not isinstance(usage, dict):
            return
        self.input_tokens = usage.get("prompt_tokens", 0)
        self.output_tokens = usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        self.cache_read_input_tokens = details.get("cached_tokens", 0)

    def forward_event(
        self, event_type: str, event: Dict, raw_data: str
    ) -> Iterator[tuple]:
        if not self.rewrite_model or "model" not in event:
            yield event_type, raw_data
            return

        rewritten = dict(event)
        rewritten["model"] = self.public_model
        yield event_type, json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))

    def forward_raw_line(self, line: str) -> Iterator[str]:
        # Some otherwise OpenAI-compatible gateways terminate a stream with a
        # bare ``[DONE]`` line instead of the required ``data: [DONE]`` SSE
        # field. Normalize that narrow drift while retaining the base handler's
        # behavior for every other non-SSE line.
        if line.strip() == "[DONE]":
            if self.emit_done_sentinel:
                yield "data: [DONE]\n\n"
            return
        yield from super().forward_raw_line(line)

    def extra_cache_fields(self) -> Dict:
        return {
            "upstream_provider": "configured_proxy",
            "upstream_profile": self.profile_name,
            "upstream_api": "chat_completions",
        }
