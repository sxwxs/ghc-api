"""Streaming handler for configured-proxy Anthropic Messages traffic."""

import json
from typing import Dict, Iterator

from .anthropic_direct import AnthropicDirectStreamHandler


class ProxyAnthropicMessagesStreamHandler(AnthropicDirectStreamHandler):
    """Pass Anthropic Messages SSE through while recording proxy metadata.

    Anthropic streams terminate with ``message_stop`` rather than ``[DONE]``.
    The parent handler also extracts Anthropic prompt-cache usage and emits
    protocol-compatible ping/error events.
    """

    log_prefix = "[Configured Proxy Anthropic Messages]"

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
        if not self.rewrite_model or event_type != "message_start":
            yield event_type, raw_data
            return

        message = event.get("message")
        if not isinstance(message, dict) or "model" not in message:
            yield event_type, raw_data
            return

        rewritten = dict(event)
        rewritten_message = dict(message)
        rewritten_message["model"] = self.public_model
        rewritten["message"] = rewritten_message
        yield event_type, json.dumps(
            rewritten, ensure_ascii=False, separators=(",", ":")
        )

    def extra_cache_fields(self) -> Dict:
        return {
            "upstream_provider": "configured_proxy",
            "upstream_profile": self.profile_name,
            "upstream_api": "messages",
        }
