"""Scripted provider for tests and demos: replays prepared replies as a real stream, without network access."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field

from ..events import BlockEvent
from ..models import Model
from ..provider import Blocks, ProviderError, StreamOptions
from ..types import (
    AssistantContent,
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)


@dataclass
class Reply:
    """One scripted reply. `error` makes the request fail; `delay` pauses before each chunk."""

    blocks: list[AssistantContent] = field(default_factory=list[AssistantContent])
    stop_reason: StopReason | None = None  # default: tool_use when there are tool calls, else stop
    error: str | None = None
    delay: float = 0.0
    usage: Usage = field(default_factory=Usage)


Script = Reply | Callable[[Context], Reply]


class FauxProvider:
    """A ProviderFn that answers each request with the next scripted reply and records every request context."""

    def __init__(self, replies: list[Script], chunk_size: int = 8) -> None:
        self.replies = list(replies)
        self.contexts: list[Context] = []
        self.chunk_size = chunk_size

    async def __call__(
        self, model: Model, context: Context, options: StreamOptions, message: AssistantMessage
    ) -> AsyncGenerator[BlockEvent, None]:
        # Copy: the loop keeps appending to the transcript list after the request.
        self.contexts.append(
            Context(system_prompt=context.system_prompt, messages=list(context.messages), tools=context.tools)
        )
        if not self.replies:
            raise ProviderError("FauxProvider has no scripted reply left")
        script = self.replies.pop(0)
        reply = script if isinstance(script, Reply) else script(context)
        if reply.error is not None:
            raise ProviderError(reply.error)
        blocks = Blocks(message)
        for block in reply.blocks:
            for event in self._open(blocks, block):
                yield event
            index = blocks.current
            assert index is not None
            for chunk in self._chunks(block):
                if reply.delay:
                    await asyncio.sleep(reply.delay)
                yield blocks.append(index, chunk)
        has_calls = any(isinstance(b, ToolCall) for b in reply.blocks)
        stop = reply.stop_reason or ("tool_use" if has_calls else "stop")
        for event in blocks.finish(model, stop, reply.usage):
            yield event

    def _open(self, blocks: Blocks, block: AssistantContent) -> list[BlockEvent]:
        if isinstance(block, TextContent):
            return blocks.start_text()
        if isinstance(block, ThinkingContent):
            return blocks.start_thinking(block.signature)
        return blocks.start_tool_call(block.id, block.name)

    def _chunks(self, block: AssistantContent) -> list[str]:
        if isinstance(block, TextContent):
            text = block.text
        elif isinstance(block, ThinkingContent):
            text = block.thinking
        else:
            text = json.dumps(block.arguments)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


def text_reply(text: str, delay: float = 0.0) -> Reply:
    """A reply with one text block."""
    return Reply(blocks=[TextContent(text=text)], delay=delay)


def tool_reply(*calls: tuple[str, str, dict[str, object]], text: str = "") -> Reply:
    """A reply calling tools, given as (id, name, arguments) tuples, optionally preceded by text."""
    blocks: list[AssistantContent] = [TextContent(text=text)] if text else []
    blocks += [ToolCall(id=call_id, name=name, arguments=dict(args)) for call_id, name, args in calls]
    return Reply(blocks=blocks)
