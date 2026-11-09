"""OpenAI-compatible adapter: stream parsing, request building, and error capture, against a mocked transport."""

import json
from typing import Any

import httpx
import msgspec
import pytest

from ghc_api.pyagent.ai.events import BlockStart, MessageEvent
from ghc_api.pyagent.ai.models import Model
from ghc_api.pyagent.ai.provider import STREAM_CUT, StreamOptions
from ghc_api.pyagent.ai.providers.openai_chat import build_request
from ghc_api.pyagent.ai.stream import empty_reply, stream_assistant
from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    user_message,
)

pytestmark = pytest.mark.anyio


def sse(*chunks: dict[str, Any] | str) -> bytes:
    return "".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks).encode()


def client_for(body: bytes, status: int = 200, requests: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def run_stream(model: Model, client: httpx.AsyncClient) -> tuple[AssistantMessage, list[MessageEvent]]:
    events: list[MessageEvent] = []
    reply = empty_reply(model)
    context = Context(system_prompt="sys", messages=[user_message("hi")], tools=[])
    await stream_assistant(model, context, StreamOptions(api_key="k", client=client), reply, events.append)
    return reply, events


async def test_streams_reasoning_text_and_tool_calls_into_blocks(model: Model) -> None:
    body = sse(
        {"choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"reasoning_content": "more"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read", "arguments": '{"pa'}}]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}},
        },
        "[DONE]",
    )
    requests: list[httpx.Request] = []
    reply, events = await run_stream(model, client_for(body, requests=requests))

    assert reply.content == [
        ThinkingContent(thinking="think more", signature="reasoning_content"),
        TextContent(text="Hello"),
        ToolCall(id="call_1", name="read", arguments={"path": "a.py"}),
    ]
    assert reply.stop_reason == "tool_use"
    assert (reply.usage.input, reply.usage.cache_read, reply.usage.output) == (60, 40, 20)
    assert [e.type for e in events] == [
        "message_start",
        *["block_start", "block_delta", "block_delta", "block_end"],
        *["block_start", "block_delta", "block_end"],
        *["block_start", "block_delta", "block_delta", "block_end"],
        "message_end",
    ]
    assert events[8] == BlockStart(
        message_id=reply.id, index=2, kind="tool_call", tool_call_id="call_1", tool_name="read"
    )
    assert requests[0].url == "http://test.local/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer k"


async def test_http_errors_end_the_reply_with_the_provider_message(model: Model) -> None:
    body = json.dumps({"error": {"message": "rate limited"}}).encode()
    reply, events = await run_stream(model, client_for(body, status=429))
    assert reply.stop_reason == "error"
    assert reply.error_message == "429: rate limited"
    assert [e.type for e in events] == ["message_start", "message_end"]


async def test_stream_error_chunks_end_the_reply_as_error(model: Model) -> None:
    reply, _ = await run_stream(model, client_for(sse({"error": {"message": "overloaded"}})))
    assert (reply.stop_reason, reply.error_message) == ("error", "overloaded")


async def test_a_stream_cut_before_its_finish_reason_is_an_error(model: Model) -> None:
    call = {"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"command": "rm'}}
    reply, _ = await run_stream(model, client_for(sse({"choices": [{"delta": {"tool_calls": [call]}}]})))
    assert (reply.stop_reason, reply.error_message) == ("error", STREAM_CUT)
    # [DONE] without a finish reason is a complete stream from a lax server.
    reply, _ = await run_stream(model, client_for(sse({"choices": [{"delta": {"content": "hi"}}]}, "[DONE]")))
    assert (reply.stop_reason, reply.text()) == ("stop", "hi")


def deepseek() -> Model:
    return Model(
        id="ds",
        provider="deepseek",
        api="openai_chat",
        base_url="http://x",
        thinking_format="deepseek",
        replay_reasoning=True,
        reasoning=True,
    )


def test_request_replays_reasoning_and_drops_failed_replies() -> None:
    model = deepseek()
    own_reply = AssistantMessage(
        content=[
            ThinkingContent(thinking="plan", signature="reasoning_content"),
            ToolCall(id="c1", name="read", arguments={"path": "a"}),
        ],
        provider="deepseek",
        model="ds",
        stop_reason="tool_use",
    )
    other_reply = AssistantMessage(
        content=[ThinkingContent(thinking="x"), TextContent(text="done")],
        provider="other",
        model="o",
        stop_reason="stop",
    )
    failed = AssistantMessage(
        content=[ToolCall(id="c2", name="read", arguments={})], provider="deepseek", model="ds", stop_reason="aborted"
    )
    context = Context(
        system_prompt="sys",
        messages=[
            user_message("hi"),
            own_reply,
            ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="file")]),
            other_reply,
            failed,
            ToolResultMessage(
                tool_call_id="c2", tool_name="read", content=[TextContent(text="Aborted")], is_error=True
            ),
        ],
        tools=[ToolSpec(name="read", description="Read", parameters={"type": "object"})],
    )
    body = build_request(model, context, StreamOptions(thinking="max"))

    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}}
            ],
            "reasoning_content": "plan",
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file"},
        {"role": "assistant", "content": "done", "reasoning_content": ""},
    ]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "max"
    assert body["tools"][0]["function"]["name"] == "read"
    assert body["stream_options"] == {"include_usage": True}


def test_thinking_off_disables_deepseek_thinking() -> None:
    body = build_request(deepseek(), Context(system_prompt="s", messages=[], tools=[]), StreamOptions(thinking="off"))
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_tool_result_images_follow_as_one_user_message(model: Model) -> None:
    vision = msgspec.structs.replace(model, images=True)
    image = ImageContent(data="AAAA", mime_type="image/png")
    context = Context(
        system_prompt="s",
        messages=[
            user_message("look"),
            AssistantMessage(
                content=[ToolCall(id="c1", name="read", arguments={})],
                provider="test",
                model="test-model",
                stop_reason="tool_use",
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="img"), image]),
        ],
        tools=[],
    )
    messages = build_request(vision, context, StreamOptions())["messages"]
    assert messages[-2] == {"role": "tool", "tool_call_id": "c1", "content": "img"}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
