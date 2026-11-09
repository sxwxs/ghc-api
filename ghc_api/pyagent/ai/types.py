"""Messages, content blocks and usage: the data shared by providers, the agent loop and the wire protocol."""

import secrets
import time
from typing import Any, Literal

import msgspec


class Wire(msgspec.Struct, rename="camel", omit_defaults=True, kw_only=True):
    """Base of every wire type: camelCase keys; an absent optional field means its zero value."""


def new_id() -> str:
    """Return a short random id for messages and session entries."""
    return secrets.token_hex(4)


def now_ms() -> int:
    """Return the current Unix time in milliseconds."""
    return int(time.time() * 1000)


class Content(Wire, tag_field="type"):
    """Base of content blocks."""


class TextContent(Content, tag="text"):
    text: str
    # Opaque provider data needed to send the block back: the Responses API message phase.
    signature: str | None = None


class ImageContent(Content, tag="image"):
    data: str  # base64
    mime_type: str


class ThinkingContent(Content, tag="thinking"):
    thinking: str
    # Opaque provider data needed to send the block back: the Anthropic signature (or redacted payload), the
    # Responses API reasoning item as JSON, or the name of the OpenAI-compatible field the reasoning arrived in.
    signature: str | None = None
    redacted: bool = False


class ToolCall(Content, tag="tool_call"):
    id: str
    name: str
    arguments: dict[str, Any]


UserContent = TextContent | ImageContent
AssistantContent = TextContent | ThinkingContent | ToolCall
ToolResultContent = TextContent | ImageContent

StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]
ThinkingLevel = Literal["off", "low", "medium", "high", "max"]


class Cost(Wire):
    """Cost in USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(Wire):
    """Token usage of one response. `input` excludes cached tokens."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: Cost = msgspec.field(default_factory=Cost)

    @property
    def context_tokens(self) -> int:
        """Tokens the conversation occupies after this response."""
        return self.input + self.cache_read + self.cache_write + self.output


class MessageBase(Wire, tag_field="role"):
    """Base of messages. `id` doubles as the session entry id. Subclasses set kw_only: msgspec does not inherit it."""

    id: str = msgspec.field(default_factory=new_id)
    timestamp: int = msgspec.field(default_factory=now_ms)


class UserMessage(MessageBase, tag="user", kw_only=True):
    content: list[UserContent]


class AssistantMessage(MessageBase, tag="assistant", kw_only=True):
    content: list[AssistantContent]
    provider: str
    model: str
    stop_reason: StopReason
    usage: Usage = msgspec.field(default_factory=Usage)
    error_message: str | None = None

    def tool_calls(self) -> list[ToolCall]:
        """Return the tool calls in content order."""
        return [block for block in self.content if isinstance(block, ToolCall)]

    def text(self) -> str:
        """Return the concatenated text blocks."""
        return "".join(block.text for block in self.content if isinstance(block, TextContent))


class ToolResultMessage(MessageBase, tag="tool_result", kw_only=True):
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent]
    details: Any = None
    is_error: bool = False


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolSpec(Wire):
    """Tool declaration sent to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


class Context(msgspec.Struct, kw_only=True):
    """Everything one model request needs. In-process only."""

    system_prompt: str
    messages: list[Message]
    tools: list[ToolSpec]


def user_message(text: str) -> UserMessage:
    """Build a user message with one text block."""
    return UserMessage(content=[TextContent(text=text)])
