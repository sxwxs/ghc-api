"""Provider adapter contract: request options, the adapter function type, and the block writer adapters build with."""

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import msgspec

from .events import BlockDelta, BlockEnd, BlockEvent, BlockStart
from .models import Model, compute_cost
from .types import (
    AssistantContent,
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    Usage,
)


@dataclass(kw_only=True)
class StreamOptions:
    """Per-request options. `client` lets tests and hosts supply their own HTTP client."""

    api_key: str | None = None
    thinking: ThinkingLevel = "off"
    client: httpx.AsyncClient | None = None


# An adapter streams one reply into the given message and yields its block events. It may raise;
# `stream_assistant` turns exceptions into an error message.
ProviderFn = Callable[[Model, Context, StreamOptions, AssistantMessage], AsyncGenerator[BlockEvent, None]]


class ProviderError(Exception):
    """A provider rejected the request or the stream reported an error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(f"{status}: {message}" if status else message)
        self.status = status


# A stream that stops before its final event holds an unfinished reply: not one to keep or whose tools to run.
STREAM_CUT = "The stream ended before the reply was complete"


class Blocks:
    """Builds an assistant message's content blocks in place and returns the matching block events.

    At most one block is open; starting a block closes the open one. Tool-call arguments arrive as JSON
    fragments and are parsed in `finish`, so fragments may still be appended to a closed tool-call block.
    """

    def __init__(self, message: AssistantMessage) -> None:
        self.message = message
        self.current: int | None = None
        self._args: dict[int, list[str]] = {}

    def start_text(self) -> list[BlockEvent]:
        """Open a text block."""
        return self._start(TextContent(text=""))

    def start_thinking(self, signature: str | None = None, redacted: bool = False) -> list[BlockEvent]:
        """Open a thinking block."""
        return self._start(ThinkingContent(thinking="", signature=signature, redacted=redacted))

    def start_tool_call(self, call_id: str, name: str) -> list[BlockEvent]:
        """Open a tool-call block; its index is `current` afterwards."""
        return self._start(ToolCall(id=call_id, name=name, arguments={}))

    def append(self, index: int, delta: str) -> BlockDelta:
        """Append a delta to block `index`."""
        block = self.message.content[index]
        if isinstance(block, TextContent):
            block.text += delta
        elif isinstance(block, ThinkingContent):
            block.thinking += delta
        else:
            self._args[index].append(delta)
        return BlockDelta(message_id=self.message.id, index=index, delta=delta)

    def text(self, delta: str) -> list[BlockEvent]:
        """Append text, opening a text block unless one is open."""
        events = [] if self._open_is(TextContent) else self.start_text()
        events.append(self.append(self._index(), delta))
        return events

    def thinking(self, delta: str, signature: str | None = None) -> list[BlockEvent]:
        """Append thinking text, opening a thinking block unless one is open."""
        events = [] if self._open_is(ThinkingContent) else self.start_thinking(signature)
        events.append(self.append(self._index(), delta))
        return events

    def set_signature(self, index: int, signature: str) -> None:
        """Attach provider replay data to text or thinking block `index`."""
        block = self.message.content[index]
        if isinstance(block, TextContent | ThinkingContent):
            block.signature = signature

    def close(self) -> list[BlockEvent]:
        """Close the open block, if any."""
        if self.current is None:
            return []
        index, self.current = self.current, None
        return [BlockEnd(message_id=self.message.id, index=index)]

    def finish(self, model: Model, stop_reason: StopReason, usage: Usage) -> list[BlockEvent]:
        """Close the open block, parse tool arguments, and record stop reason and priced usage."""
        events = self.close()
        for index, parts in self._args.items():
            call = self.message.content[index]
            if isinstance(call, ToolCall):
                call.arguments = _parse_arguments("".join(parts))
        usage.cost = compute_cost(model, usage)
        self.message.usage = usage
        self.message.stop_reason = stop_reason
        return events

    def _start(self, block: AssistantContent) -> list[BlockEvent]:
        events = self.close()
        self.message.content.append(block)
        index = len(self.message.content) - 1
        self.current = index
        if isinstance(block, ToolCall):
            self._args[index] = []
            events.append(
                BlockStart(
                    message_id=self.message.id,
                    index=index,
                    kind="tool_call",
                    tool_call_id=block.id,
                    tool_name=block.name,
                )
            )
        else:
            kind = "text" if isinstance(block, TextContent) else "thinking"
            events.append(BlockStart(message_id=self.message.id, index=index, kind=kind))
        return events

    def _open_is(self, kind: type[TextContent] | type[ThinkingContent]) -> bool:
        return self.current is not None and isinstance(self.message.content[self.current], kind)

    def _index(self) -> int:
        assert self.current is not None  # every caller opened a block first
        return self.current


def _parse_arguments(raw: str) -> dict[str, Any]:
    # Truncated or malformed JSON yields no arguments; schema validation then reports what is missing.
    if not raw.strip():
        return {}
    try:
        return msgspec.json.decode(raw, type=dict[str, Any])
    except msgspec.DecodeError:
        return {}
