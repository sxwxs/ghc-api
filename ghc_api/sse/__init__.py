"""SSE stream handlers.

Base class plus per-endpoint subclasses. The base captures every upstream
``data:`` line verbatim into the cache and forwards every upstream line to the
client unchanged. Subclasses opt into translations (request-shape rewriting,
leaked-tool-call recovery) by overriding hook methods.

Available transports:
  - ``AnthropicDirectStreamHandler``         -> /v1/messages (passthrough)
  - ``AnthropicDirectStreamHandlerWithRecovery`` -> /v1/messages with leaked-tool-call recovery
  - ``OpenAIResponsesStreamHandler``         -> /v1/responses (passthrough)
  - ``AnthropicResponsesStreamHandler``      -> Responses translated to /v1/messages
"""

from .base import SSEStreamHandler
from .anthropic_direct import (
    AnthropicDirectStreamHandler,
    AnthropicDirectStreamHandlerWithRecovery,
)
from .openai_responses import OpenAIResponsesStreamHandler
from .anthropic_responses import (
    AnthropicResponsesStreamHandler,
    ResponsesAnthropicEventTranslator,
    StopSequenceScanner,
)

__all__ = [
    "SSEStreamHandler",
    "AnthropicDirectStreamHandler",
    "AnthropicDirectStreamHandlerWithRecovery",
    "OpenAIResponsesStreamHandler",
    "AnthropicResponsesStreamHandler",
    "ResponsesAnthropicEventTranslator",
    "StopSequenceScanner",
]
