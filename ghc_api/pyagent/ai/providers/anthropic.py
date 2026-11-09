"""Adapter for the Anthropic Messages API."""

import re
from collections.abc import AsyncGenerator
from typing import Any

import msgspec

from ..events import BlockEvent
from ..http import raise_for_status, shared_client, sse_data
from ..models import Model
from ..provider import STREAM_CUT, Blocks, ProviderError, StreamOptions
from ..replay import replayable, same_model
from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    StopReason,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

API_VERSION = "2023-06-01"
CACHE = {"type": "ephemeral"}


async def stream(
    model: Model, context: Context, options: StreamOptions, message: AssistantMessage
) -> AsyncGenerator[BlockEvent, None]:
    """Stream one Messages API reply into `message`."""
    headers = {"Content-Type": "application/json", "anthropic-version": API_VERSION, **model.headers}
    if options.api_key:
        headers["x-api-key"] = options.api_key
    client = options.client or shared_client()
    url = model.base_url.rstrip("/") + "/v1/messages"
    async with client.stream("POST", url, json=build_request(model, context, options), headers=headers) as response:
        await raise_for_status(response)
        blocks = Blocks(message)
        # Anthropic block index -> our content index; they differ when a block type is skipped.
        indices: dict[int, int] = {}
        usage = Usage()
        stop_reason: StopReason = "stop"
        finished = False  # message_stop, or a message_delta carrying the stop reason
        async for data in sse_data(response):
            event = EVENT.decode(data)
            match event.type:
                case "message_start" if event.message is not None and event.message.usage is not None:
                    usage = _usage(event.message.usage, usage)
                case "content_block_start" if event.index is not None and event.content_block is not None:
                    for out in _block_start(blocks, indices, event.index, event.content_block):
                        yield out
                case "content_block_delta" if event.index in indices and event.delta is not None:
                    for out in _block_delta(blocks, indices[event.index], event.delta):
                        yield out
                case "content_block_stop" if event.index in indices and indices[event.index] == blocks.current:
                    for out in blocks.close():
                        yield out
                case "message_delta":
                    if event.usage is not None:
                        usage = _usage(event.usage, usage)
                    if event.delta is not None:
                        stop_reason = _stop_reason(event.delta.stop_reason, message)
                        finished = finished or event.delta.stop_reason is not None
                case "message_stop":
                    finished = True
                case "error":
                    raise ProviderError((event.error.message if event.error else None) or data)
                case _:
                    pass
        if not finished:
            raise ProviderError(STREAM_CUT)
        for out in blocks.finish(model, stop_reason, usage):
            yield out


# Stream events are decoded leniently into one shape: every field is optional and unknown fields are ignored.


class _Block(msgspec.Struct):
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    signature: str | None = None
    data: str | None = None


class _Delta(msgspec.Struct):
    type: str | None = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None
    stop_reason: str | None = None


class _Usage(msgspec.Struct):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class _Message(msgspec.Struct):
    usage: _Usage | None = None


class _Error(msgspec.Struct):
    message: str | None = None


class _Event(msgspec.Struct):
    type: str
    index: int | None = None
    content_block: _Block | None = None
    delta: _Delta | None = None
    usage: _Usage | None = None
    message: _Message | None = None
    error: _Error | None = None


EVENT = msgspec.json.Decoder(_Event)


def _block_start(blocks: Blocks, indices: dict[int, int], index: int, block: _Block) -> list[BlockEvent]:
    match block.type:
        case "text":
            events = blocks.start_text()
            if block.text:
                events.append(blocks.append(_current(blocks), block.text))
        case "thinking":
            events = blocks.start_thinking(block.signature or None)
        case "redacted_thinking":
            events = blocks.start_thinking(block.data, redacted=True)
        case "tool_use":
            events = blocks.start_tool_call(block.id or "", block.name or "")
        case _:
            return []
    indices[index] = _current(blocks)
    return events


def _block_delta(blocks: Blocks, index: int, delta: _Delta) -> list[BlockEvent]:
    match delta.type:
        case "text_delta" if delta.text:
            return [blocks.append(index, delta.text)]
        case "thinking_delta" if delta.thinking:
            return [blocks.append(index, delta.thinking)]
        case "input_json_delta" if delta.partial_json:
            return [blocks.append(index, delta.partial_json)]
        case "signature_delta" if delta.signature:
            blocks.set_signature(index, delta.signature)
            return []
        case _:
            return []


def _current(blocks: Blocks) -> int:
    assert blocks.current is not None  # called right after a block was opened
    return blocks.current


def _usage(raw: _Usage, previous: Usage) -> Usage:
    # message_start reports input; message_delta reports cumulative output and may repeat input fields.
    return Usage(
        input=raw.input_tokens if raw.input_tokens is not None else previous.input,
        output=raw.output_tokens if raw.output_tokens is not None else previous.output,
        cache_read=raw.cache_read_input_tokens if raw.cache_read_input_tokens is not None else previous.cache_read,
        cache_write=(
            raw.cache_creation_input_tokens if raw.cache_creation_input_tokens is not None else previous.cache_write
        ),
    )


def _stop_reason(reason: str | None, message: AssistantMessage) -> StopReason:
    match reason:
        case "tool_use":
            return "tool_use"
        case "max_tokens" | "model_context_window_exceeded":
            return "length"
        case "refusal":
            message.error_message = "The model refused to answer"
            return "error"
        case _:
            return "stop"


def build_request(model: Model, context: Context, options: StreamOptions) -> dict[str, Any]:
    """Build the Messages API request body. Cache breakpoints sit on the system prompt and the last message."""
    messages = _messages(model, context.messages)
    if messages and messages[-1]["content"]:
        messages[-1]["content"][-1]["cache_control"] = CACHE
    body: dict[str, Any] = {
        "model": model.id,
        "max_tokens": model.max_tokens,
        "stream": True,
        "system": [{"type": "text", "text": context.system_prompt, "cache_control": CACHE}],
        "messages": messages,
    }
    if context.tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in context.tools
        ]
    if model.reasoning:
        body.update(_thinking_params(model, options.thinking))
    return body


BUDGETS: dict[ThinkingLevel, int] = {"low": 2048, "medium": 8192, "high": 16384, "max": 32000}


def _thinking_params(model: Model, level: ThinkingLevel) -> dict[str, Any]:
    if level == "off":
        return {"thinking": {"type": "disabled"}}
    if model.thinking_format == "anthropic_budget":
        # The budget must stay below max_tokens.
        budget = min(BUDGETS[level], model.max_tokens - 1024)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": level}}


def _messages(model: Model, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in replayable(messages):
        if isinstance(message, UserMessage):
            _add(out, "user", [_content(model, block) for block in message.content])
        elif isinstance(message, AssistantMessage):
            _add(out, "assistant", _assistant_blocks(model, message))
        else:
            _add(out, "user", [_tool_result(model, message)])
    return out


def _add(out: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    # Consecutive same-role messages merge: tool results of one turn travel in one user message.
    if not blocks:
        return
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": blocks})


def _content(model: Model, block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if not model.images:
        return {"type": "text", "text": "(image omitted: model does not accept images)"}
    return {"type": "image", "source": {"type": "base64", "media_type": block.mime_type, "data": block.data}}


def _assistant_blocks(model: Model, message: AssistantMessage) -> list[dict[str, Any]]:
    replay_thinking = same_model(message, model)
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            if block.text:
                blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            blocks.append({"type": "tool_use", "id": _tool_id(block.id), "name": block.name, "input": block.arguments})
        elif replay_thinking and block.signature:
            if block.redacted:
                blocks.append({"type": "redacted_thinking", "data": block.signature})
            else:
                blocks.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})
    return blocks


def _tool_result(model: Model, message: ToolResultMessage) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": _tool_id(message.tool_call_id),
        "content": [_content(model, block) for block in message.content] or [{"type": "text", "text": "(no output)"}],
        "is_error": message.is_error,
    }


def _tool_id(call_id: str) -> str:
    # Ids from other providers may contain characters Anthropic rejects.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
