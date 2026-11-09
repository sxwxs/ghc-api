"""Adapter for OpenAI Chat Completions and compatible endpoints (DeepSeek, Kimi, OpenRouter, Ollama, vLLM, ...)."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import msgspec

from ..events import BlockEvent
from ..http import raise_for_status, shared_client, sse_data
from ..models import Model
from ..provider import Blocks, ProviderError, StreamOptions
from ..replay import replayable, same_model
from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    StopReason,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolResultMessage,
    Usage,
    UserMessage,
    new_id,
)


async def stream(
    model: Model, context: Context, options: StreamOptions, message: AssistantMessage
) -> AsyncGenerator[BlockEvent, None]:
    """Stream one Chat Completions reply into `message`."""
    headers = {"Content-Type": "application/json", **model.headers}
    if options.api_key:
        headers["Authorization"] = f"Bearer {options.api_key}"
    client = options.client or shared_client()
    url = model.base_url.rstrip("/") + "/chat/completions"
    async with client.stream("POST", url, json=build_request(model, context, options), headers=headers) as response:
        await raise_for_status(response)
        blocks = Blocks(message)
        tool_blocks: dict[int, int] = {}
        finish_reason: str | None = None
        usage = Usage()
        async for data in sse_data(response):
            if data == "[DONE]":
                break
            chunk = CHUNK.decode(data)
            if chunk.error is not None:
                raise ProviderError(chunk.error.message or data)
            if chunk.usage is not None:
                usage = _usage(chunk.usage)
            for choice in chunk.choices or []:
                if choice.delta is not None:
                    for event in _delta(blocks, tool_blocks, choice.delta):
                        yield event
                finish_reason = choice.finish_reason or finish_reason
        for event in blocks.finish(model, _stop_reason(finish_reason), usage):
            yield event


# Stream chunks are decoded leniently: every field is optional and unknown fields are ignored.


class _Function(msgspec.Struct):
    name: str | None = None
    arguments: str | None = None


class _ToolCallDelta(msgspec.Struct):
    index: int = 0
    id: str | None = None
    function: _Function | None = None


class _Delta(msgspec.Struct):
    content: str | None = None
    # Providers name the reasoning field differently; some send two copies, so the first non-empty one wins.
    reasoning_content: str | None = None
    reasoning: str | None = None
    reasoning_text: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


class _Choice(msgspec.Struct):
    delta: _Delta | None = None
    finish_reason: str | None = None


class _PromptDetails(msgspec.Struct):
    cached_tokens: int | None = None


class _Usage(msgspec.Struct):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tokens_details: _PromptDetails | None = None
    prompt_cache_hit_tokens: int | None = None  # DeepSeek


class _Error(msgspec.Struct):
    message: str | None = None


class _Chunk(msgspec.Struct):
    choices: list[_Choice] | None = None
    usage: _Usage | None = None
    error: _Error | None = None


CHUNK = msgspec.json.Decoder(_Chunk)


def _delta(blocks: Blocks, tool_blocks: dict[int, int], delta: _Delta) -> list[BlockEvent]:
    events: list[BlockEvent] = []
    for field in ("reasoning_content", "reasoning", "reasoning_text"):
        if text := getattr(delta, field):
            events += blocks.thinking(text, signature=field)
            break
    if delta.content:
        events += blocks.text(delta.content)
    for call in delta.tool_calls or []:
        function = call.function or _Function()
        if call.index not in tool_blocks:
            events += blocks.start_tool_call(call.id or f"call_{new_id()}", function.name or "")
            assert blocks.current is not None  # start_tool_call opened it
            tool_blocks[call.index] = blocks.current
        if function.arguments:
            events.append(blocks.append(tool_blocks[call.index], function.arguments))
    return events


def _stop_reason(finish_reason: str | None) -> StopReason:
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "length":
        return "length"
    if finish_reason == "content_filter":
        return "error"
    return "stop"


def _usage(raw: _Usage) -> Usage:
    details = raw.prompt_tokens_details
    cached = (details.cached_tokens if details else None) or raw.prompt_cache_hit_tokens or 0
    return Usage(input=(raw.prompt_tokens or 0) - cached, output=raw.completion_tokens or 0, cache_read=cached)


def build_request(model: Model, context: Context, options: StreamOptions) -> dict[str, Any]:
    """Build the Chat Completions request body."""
    body: dict[str, Any] = {
        "model": model.id,
        "messages": [{"role": "system", "content": context.system_prompt}, *_messages(model, context.messages)],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if context.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in context.tools
        ]
    if model.reasoning:
        body.update(_thinking_params(model, options.thinking))
    return body


def _thinking_params(model: Model, level: ThinkingLevel) -> dict[str, Any]:
    match model.thinking_format:
        case "reasoning_effort":
            return {} if level == "off" else {"reasoning_effort": "high" if level == "max" else level}
        case "deepseek":
            if level == "off":
                return {"thinking": {"type": "disabled"}}
            # DeepSeek V4 offers only "high" and "max".
            return {"thinking": {"type": "enabled"}, "reasoning_effort": "max" if level == "max" else "high"}
        case "openrouter":
            return {} if level == "off" else {"reasoning": {"effort": "high" if level == "max" else level}}
        case _:
            return {}


def _messages(model: Model, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Tool messages carry text only; their images follow in one user message after the run of tool results.
    pending_images: list[dict[str, Any]] = []
    for message in replayable(messages):
        if pending_images and not isinstance(message, ToolResultMessage):
            out.append(_image_message(pending_images))
            pending_images = []
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": _user_content(model, message)})
        elif isinstance(message, AssistantMessage):
            converted = _assistant(model, message)
            if converted is not None:
                out.append(converted)
        else:
            text = "\n".join(block.text for block in message.content if isinstance(block, TextContent))
            out.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": text or "(no output)"})
            if model.images:
                pending_images += [_image_part(b) for b in message.content if isinstance(b, ImageContent)]
    if pending_images:
        out.append(_image_message(pending_images))
    return out


def _user_content(model: Model, message: UserMessage) -> str | list[dict[str, Any]]:
    if all(isinstance(block, TextContent) for block in message.content):
        return "".join(block.text for block in message.content if isinstance(block, TextContent))
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif model.images:
            parts.append(_image_part(block))
        else:
            parts.append({"type": "text", "text": "(image omitted: model does not accept images)"})
    return parts


def _assistant(model: Model, message: AssistantMessage) -> dict[str, Any] | None:
    text = message.text()
    calls = message.tool_calls()
    if not text and not calls:
        return None
    # Assistant content is sent as a plain string: some models mirror a content-part array literally.
    converted: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        converted["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in calls
        ]
    thinking = [b for b in message.content if isinstance(b, ThinkingContent) and b.thinking]
    if thinking and same_model(message, model):
        converted[thinking[0].signature or "reasoning_content"] = "\n".join(b.thinking for b in thinking)
    elif model.replay_reasoning and model.reasoning:
        converted["reasoning_content"] = ""
    return converted


def _image_part(block: ImageContent) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}


def _image_message(images: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": "Images from the tool results above:"}, *images]}
