"""Adapter for the OpenAI Responses API (`POST {baseUrl}/responses`), stateless: store=false, reasoning replayed."""

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
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


async def stream(
    model: Model, context: Context, options: StreamOptions, message: AssistantMessage
) -> AsyncGenerator[BlockEvent, None]:
    """Stream one Responses API reply into `message`."""
    headers = {"Content-Type": "application/json", **model.headers}
    if options.api_key:
        headers["Authorization"] = f"Bearer {options.api_key}"
    client = options.client or shared_client()
    url = model.base_url.rstrip("/") + "/responses"
    async with client.stream("POST", url, json=build_request(model, context, options), headers=headers) as response:
        await raise_for_status(response)
        state = _State(Blocks(message))
        async for data in sse_data(response):
            for event in state.handle(EVENT.decode(data), data):
                yield event
        for event in state.blocks.finish(model, state.stop_reason(), state.usage):
            yield event


# Stream events are decoded leniently: every field is optional and unknown fields are ignored. Output items
# stay raw, because a reasoning item is replayed exactly as received.


class _Item(msgspec.Struct):
    type: str
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    phase: str | None = None


class _TokenDetails(msgspec.Struct):
    cached_tokens: int | None = None


class _Usage(msgspec.Struct):
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens_details: _TokenDetails | None = None


class _Error(msgspec.Struct):
    message: str | None = None


class _Incomplete(msgspec.Struct):
    reason: str | None = None


class _Response(msgspec.Struct):
    status: str | None = None
    usage: _Usage | None = None
    error: _Error | None = None
    incomplete_details: _Incomplete | None = None


class _Event(msgspec.Struct):
    type: str
    output_index: int | None = None
    summary_index: int | None = None
    delta: str | None = None
    item: msgspec.Raw = msgspec.Raw()  # empty when absent; Raw cannot be optional
    response: _Response | None = None
    message: str | None = None


EVENT = msgspec.json.Decoder(_Event)
ITEM = msgspec.json.Decoder(_Item)


class _State:
    """Maps output items (by output_index) to content blocks while the stream runs."""

    def __init__(self, blocks: Blocks) -> None:
        self.blocks = blocks
        self.items: dict[int, int] = {}
        self.streamed_arguments: set[int] = set()
        self.usage = Usage()
        self.status: str | None = None
        self.incomplete_reason: str | None = None

    def handle(self, event: _Event, data: str) -> list[BlockEvent]:
        index = self.items.get(event.output_index) if event.output_index is not None else None
        match event.type:
            case "response.output_item.added" if event.item and event.output_index is not None:
                return self._item_added(event.output_index, ITEM.decode(event.item))
            case "response.output_item.done" if index is not None and event.item:
                return self._item_done(index, event.item)
            case "response.reasoning_summary_part.added" if index is not None and event.summary_index:
                return [self.blocks.append(index, "\n\n")]
            case "response.reasoning_summary_text.delta" | "response.output_text.delta" | "response.refusal.delta" if (
                index is not None and event.delta
            ):
                return [self.blocks.append(index, event.delta)]
            case "response.function_call_arguments.delta" if index is not None and event.delta:
                self.streamed_arguments.add(index)
                return [self.blocks.append(index, event.delta)]
            case "response.completed" | "response.incomplete" if event.response is not None:
                self._finished(event.response)
                return []
            case "response.failed":
                error = event.response.error if event.response else None
                raise ProviderError((error.message if error else None) or data)
            case "error":
                raise ProviderError(event.message or data)
            case _:
                return []

    def stop_reason(self) -> StopReason:
        if self.status == "incomplete":
            return "error" if self.incomplete_reason == "content_filter" else "length"
        return "tool_use" if self.blocks.message.tool_calls() else "stop"

    def _item_added(self, output_index: int, item: _Item) -> list[BlockEvent]:
        match item.type:
            case "reasoning":
                events = self.blocks.start_thinking()
            case "message":
                events = self.blocks.start_text()
            case "function_call":
                events = self.blocks.start_tool_call(item.call_id or "", item.name or "")
            case _:
                return []
        index = self.blocks.current
        assert index is not None  # a block was just opened
        self.items[output_index] = index
        if item.phase:
            self.blocks.set_signature(index, item.phase)
        return events

    def _item_done(self, index: int, raw: msgspec.Raw) -> list[BlockEvent]:
        item = ITEM.decode(raw)
        events: list[BlockEvent] = []
        if item.type == "reasoning":
            # The final item carries the encrypted reasoning needed to continue statelessly.
            self.blocks.set_signature(index, bytes(raw).decode())
        elif item.type == "function_call" and index not in self.streamed_arguments and item.arguments:
            events.append(self.blocks.append(index, item.arguments))
        if self.blocks.current == index:
            events += self.blocks.close()
        return events

    def _finished(self, response: _Response) -> None:
        self.status = response.status
        self.incomplete_reason = response.incomplete_details.reason if response.incomplete_details else None
        if response.usage is not None:
            usage = response.usage
            details = usage.input_tokens_details
            cached = (details.cached_tokens if details else None) or 0
            self.usage = Usage(
                input=(usage.input_tokens or 0) - cached, output=usage.output_tokens or 0, cache_read=cached
            )


def build_request(model: Model, context: Context, options: StreamOptions) -> dict[str, Any]:
    """Build the request body. Thinking "off" sends no effort; models that cannot disable reasoning use a default."""
    body: dict[str, Any] = {
        "model": model.id,
        "instructions": context.system_prompt,
        "input": _input(model, context.messages),
        "stream": True,
        "store": False,
    }
    if context.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,
            }
            for t in context.tools
        ]
    if model.reasoning:
        reasoning: dict[str, str] = {"summary": "auto"}
        if options.thinking != "off":
            reasoning["effort"] = options.thinking
        body["reasoning"] = reasoning
        # Without stored responses, reasoning continues only through its encrypted form.
        body["include"] = ["reasoning.encrypted_content"]
    return body


def _input(model: Model, messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in replayable(messages):
        if isinstance(message, UserMessage):
            items.append({"role": "user", "content": [_input_content(model, block) for block in message.content]})
        elif isinstance(message, AssistantMessage):
            items += _assistant_items(model, message)
        else:
            items.append(
                {"type": "function_call_output", "call_id": message.tool_call_id, "output": _output(model, message)}
            )
    return items


def _input_content(model: Model, block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "input_text", "text": block.text}
    if not model.images:
        return {"type": "input_text", "text": "(image omitted: model does not accept images)"}
    return {"type": "input_image", "detail": "auto", "image_url": f"data:{block.mime_type};base64,{block.data}"}


def _assistant_items(model: Model, message: AssistantMessage) -> list[dict[str, Any]]:
    replay = same_model(message, model)
    items: list[dict[str, Any]] = []
    for number, block in enumerate(message.content):
        if isinstance(block, ThinkingContent):
            if replay and block.signature:
                items.append(msgspec.json.decode(block.signature, type=dict[str, Any]))
        elif isinstance(block, TextContent):
            if block.text:
                item: dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "id": f"msg_{message.id}_{number}",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": block.text, "annotations": []}],
                }
                if block.signature:
                    item["phase"] = block.signature
                items.append(item)
        else:
            items.append(_function_call(block))
    return items


def _function_call(call: ToolCall) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": json.dumps(call.arguments)}


def _output(model: Model, message: ToolResultMessage) -> str | list[dict[str, Any]]:
    if all(isinstance(block, TextContent) for block in message.content):
        return "\n".join(block.text for block in message.content if isinstance(block, TextContent)) or "(no output)"
    return [_input_content(model, block) for block in message.content]
