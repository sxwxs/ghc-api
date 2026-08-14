"""GLM 5.2 NVFP4 to OpenAI Chat Completions streaming adapter."""

from __future__ import annotations

from typing import Dict, Mapping

from ..proxy.glm_5_2 import GLM_5_2_NVFP4, GlmTextStream
from .proxy_chat_completions import ProxyChatCompletionsStreamHandler
from .proxy_kimi_k3 import KimiK3ChatCompletionsStreamHandler


class Glm52ChatCompletionsStreamHandler(KimiK3ChatCompletionsStreamHandler):
    """Translate GLM thinking/native tools and synthesize a terminal chunk."""

    def __init__(
        self,
        *args,
        declared_tools: Mapping[str, Dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, declared_tools=(), **kwargs)
        self.declared_tool_schemas = dict(declared_tools or {})

    def _stream_for(self, index: int) -> GlmTextStream:
        stream = self._text_streams.get(index)
        if stream is None:
            stream = GlmTextStream(self.declared_tool_schemas)
            self._text_streams[index] = stream
        return stream

    def extra_cache_fields(self) -> Dict:
        # Skip Kimi's hard-coded compatibility label while retaining the common
        # configured-proxy accounting fields.
        fields = ProxyChatCompletionsStreamHandler.extra_cache_fields(self)
        fields.update({
            "compatibility": GLM_5_2_NVFP4,
            "public_events": list(self.public_events),
            "raw_sse_lines": list(self.raw_sse_lines),
            "response_body": list(self.public_events),
        })
        return fields
