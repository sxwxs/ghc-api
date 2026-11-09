"""Anthropic adapter: stream parsing, request building with cache breakpoints, and thinking parameters."""

import json
from typing import Any

import httpx
import msgspec
import pytest

from ghc_api.pyagent.ai.models import Model
from ghc_api.pyagent.ai.provider import StreamOptions
from ghc_api.pyagent.ai.providers.anthropic import build_request
from ghc_api.pyagent.ai.stream import empty_reply, stream_assistant
from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    user_message,
)

pytestmark = pytest.mark.anyio

CLAUDE = Model(id="claude", provider="anthropic", api="anthropic", base_url="http://a", reasoning=True, max_tokens=8000)


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


async def run_stream(body: bytes, status: int = 200) -> AssistantMessage:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reply = empty_reply(CLAUDE)
    context = Context(system_prompt="sys", messages=[user_message("hi")], tools=[])
    await stream_assistant(CLAUDE, context, StreamOptions(api_key="k", client=client), reply, lambda _: None)
    assert requests[0].headers["x-api-key"] == "k"
    assert str(requests[0].url) == "http://a/v1/messages"
    return reply


async def test_streams_thinking_text_and_tool_use() -> None:
    body = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 50, "cache_read_input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "bash"}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"command"'},
        },
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ': "ls"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    )
    reply = await run_stream(body)
    assert reply.content == [
        ThinkingContent(thinking="hmm", signature="sig"),
        TextContent(text="Hi"),
        ToolCall(id="t1", name="bash", arguments={"command": "ls"}),
    ]
    assert reply.stop_reason == "tool_use"
    assert (reply.usage.input, reply.usage.cache_read, reply.usage.output) == (50, 10, 30)


async def test_error_events_end_the_reply_as_error() -> None:
    reply = await run_stream(sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    assert (reply.stop_reason, reply.error_message) == ("error", "Overloaded")


def test_request_merges_tool_results_and_sets_cache_breakpoints() -> None:
    own = AssistantMessage(
        content=[
            ThinkingContent(thinking="plan", signature="sig"),
            ToolCall(id="call.1", name="read", arguments={"path": "a"}),
            ToolCall(id="call.2", name="read", arguments={"path": "b"}),
        ],
        provider="anthropic",
        model="claude",
        stop_reason="tool_use",
    )
    context = Context(
        system_prompt="sys",
        messages=[
            user_message("hi"),
            own,
            ToolResultMessage(tool_call_id="call.1", tool_name="read", content=[TextContent(text="A")]),
            ToolResultMessage(tool_call_id="call.2", tool_name="read", content=[TextContent(text="B")], is_error=True),
        ],
        tools=[],
    )
    body = build_request(CLAUDE, context, StreamOptions(thinking="high"))

    assert body["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert body["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a"}},
        {"type": "tool_use", "id": "call_2", "name": "read", "input": {"path": "b"}},
    ]
    results = body["messages"][2]
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["call_1", "call_2"]
    assert results["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


def test_thinking_from_another_model_is_not_replayed() -> None:
    other = AssistantMessage(
        content=[ThinkingContent(thinking="x", signature="s"), TextContent(text="ok")],
        provider="anthropic",
        model="older",
        stop_reason="stop",
    )
    context = Context(system_prompt="s", messages=[user_message("a"), other, user_message("b")], tools=[])
    assert build_request(CLAUDE, context, StreamOptions())["messages"][1]["content"] == [{"type": "text", "text": "ok"}]


def test_budget_thinking_stays_below_max_tokens() -> None:
    model = msgspec.structs.replace(CLAUDE, thinking_format="anthropic_budget", max_tokens=10000)
    body = build_request(model, Context(system_prompt="s", messages=[], tools=[]), StreamOptions(thinking="max"))
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8976}
    off = build_request(model, Context(system_prompt="s", messages=[], tools=[]), StreamOptions(thinking="off"))
    assert off["thinking"] == {"type": "disabled"}
