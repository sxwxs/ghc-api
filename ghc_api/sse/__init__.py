"""SSE stream handlers.

Base class plus per-endpoint subclasses. The base captures every upstream
``data:`` line verbatim into the cache and forwards every upstream line to the
client unchanged. Subclasses opt into translations (request-shape rewriting,
leaked-tool-call recovery) by overriding hook methods.

Phase 1 covers the two passthrough paths:
  - ``AnthropicDirectStreamHandler``         -> /v1/messages (passthrough)
  - ``AnthropicDirectStreamHandlerWithRecovery`` -> /v1/messages with leaked-tool-call recovery
  - ``OpenAIResponsesStreamHandler``         -> /v1/responses (passthrough)
"""

from .base import SSEStreamHandler
from .anthropic_direct import (
    AnthropicDirectStreamHandler,
    AnthropicDirectStreamHandlerWithRecovery,
)
from .openai_responses import OpenAIResponsesStreamHandler

__all__ = [
    "SSEStreamHandler",
    "AnthropicDirectStreamHandler",
    "AnthropicDirectStreamHandlerWithRecovery",
    "OpenAIResponsesStreamHandler",
]
