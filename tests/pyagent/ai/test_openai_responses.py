"""OpenAI Responses adapter: stream parsing, stateless reasoning replay, and request building."""

import json
from typing import Any

import httpx
import msgspec
import pytest

from ghc_api.pyagent.ai.models import Model
from ghc_api.pyagent.ai.provider import STREAM_CUT, StreamOptions
from ghc_api.pyagent.ai.providers.openai_responses import build_request
from ghc_api.pyagent.ai.stream import empty_reply, stream_assistant
from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    user_message,
)

pytestmark = pytest.mark.anyio

GPT = Model(id="gpt", provider="local", api="openai_responses", base_url="http://r", reasoning=True, images=True)
REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Plan"}],
    "encrypted_content": "ENC",
}


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


async def run_stream(body: bytes) -> tuple[AssistantMessage, httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body)

    reply = empty_reply(GPT)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    context = Context(system_prompt="sys", messages=[user_message("hi")], tools=[])
    await stream_assistant(GPT, context, StreamOptions(client=client), reply, lambda _: None)
    return reply, requests[0]


async def test_streams_reasoning_text_and_function_calls() -> None:
    body = sse(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        },
        {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "Pl"},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "an"},
        {"type": "response.output_item.done", "output_index": 0, "item": REASONING_ITEM},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "phase": "commentary"}},
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "Checking."},
        {"type": "response.output_item.done", "output_index": 1, "item": {"type": "message"}},
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '{"command":'},
        {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '"ls"}'},
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {"type": "function_call", "arguments": '{"command":"ls"}'},
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 100, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 30}},
            },
        },
    )
    reply, request = await run_stream(body)
    assert str(request.url) == "http://r/responses"
    assert "authorization" not in request.headers
    thinking, text, call = reply.content
    assert isinstance(thinking, ThinkingContent) and thinking.thinking == "Plan"
    assert thinking.signature is not None and json.loads(thinking.signature) == REASONING_ITEM
    assert text == TextContent(text="Checking.", signature="commentary")
    assert call == ToolCall(id="call_1", name="bash", arguments={"command": "ls"})
    assert reply.stop_reason == "tool_use"
    assert (reply.usage.input, reply.usage.cache_read, reply.usage.output) == (70, 30, 9)


async def test_arguments_sent_only_with_the_finished_item_are_kept() -> None:
    body = sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "c", "name": "read"},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "arguments": '{"path":"a"}'},
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    )
    reply, _ = await run_stream(body)
    assert reply.content == [ToolCall(id="c", name="read", arguments={"path": "a"})]


async def test_incomplete_and_failed_responses() -> None:
    incomplete = sse(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Trunc"},
        {
            "type": "response.incomplete",
            "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        },
    )
    reply, _ = await run_stream(incomplete)
    assert (reply.stop_reason, reply.text()) == ("length", "Trunc")
    failed, _ = await run_stream(
        sse({"type": "response.failed", "response": {"status": "failed", "error": {"message": "quota"}}})
    )
    assert (failed.stop_reason, failed.error_message) == ("error", "quota")
    cut, _ = await run_stream(
        sse(
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Half"},
        )
    )
    assert (cut.stop_reason, cut.error_message) == ("error", STREAM_CUT)


def test_request_replays_reasoning_phase_and_tool_outputs_for_the_same_model() -> None:
    reply = AssistantMessage(
        content=[
            ThinkingContent(thinking="Plan", signature=json.dumps(REASONING_ITEM)),
            TextContent(text="Checking.", signature="commentary"),
            ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
        ],
        provider="local",
        model="gpt",
        stop_reason="tool_use",
    )
    context = Context(
        system_prompt="sys",
        messages=[
            user_message("hi"),
            reply,
            ToolResultMessage(tool_call_id="call_1", tool_name="bash", content=[TextContent(text="a.py")]),
        ],
        tools=[ToolSpec(name="bash", description="Run", parameters={"type": "object"})],
    )
    body = build_request(GPT, context, StreamOptions(thinking="high"))
    assert body["instructions"] == "sys"
    assert body["store"] is False
    assert body["reasoning"] == {"summary": "auto", "effort": "high"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["tools"] == [
        {"type": "function", "name": "bash", "description": "Run", "parameters": {"type": "object"}, "strict": False}
    ]
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        REASONING_ITEM,
        {
            "type": "message",
            "role": "assistant",
            "id": f"msg_{reply.id}_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Checking.", "annotations": []}],
            "phase": "commentary",
        },
        {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "ls"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "a.py"},
    ]


def test_reasoning_from_another_model_is_dropped_and_off_sends_no_effort() -> None:
    other = msgspec.structs.replace(GPT, id="other")
    reply = AssistantMessage(
        content=[ThinkingContent(thinking="x", signature=json.dumps(REASONING_ITEM)), TextContent(text="ok")],
        provider="local",
        model="gpt",
        stop_reason="stop",
    )
    body = build_request(
        other, Context(system_prompt="s", messages=[user_message("a"), reply], tools=[]), StreamOptions()
    )
    assert [item.get("type", "user") for item in body["input"]] == ["user", "message"]
    assert body["reasoning"] == {"summary": "auto"}
