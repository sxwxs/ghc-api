"""Wire encoding of messages: camelCase keys, snake_case tags, ids always present."""

import msgspec

from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    user_message,
)


def test_assistant_message_encodes_camel_case_keys_and_snake_case_tags() -> None:
    message = AssistantMessage(
        content=[TextContent(text="hi"), ToolCall(id="c1", name="read", arguments={"path": "a.py"})],
        provider="p",
        model="m",
        stop_reason="tool_use",
        usage=Usage(input=10, cache_read=5),
    )
    data = msgspec.json.decode(msgspec.json.encode(message))
    assert data["role"] == "assistant"
    assert data["stopReason"] == "tool_use"
    assert data["content"][1] == {"type": "tool_call", "id": "c1", "name": "read", "arguments": {"path": "a.py"}}
    assert data["usage"]["cacheRead"] == 5
    assert data["id"] == message.id and len(message.id) == 8
    assert data["timestamp"] == message.timestamp


def test_messages_round_trip_through_the_union() -> None:
    messages: list[Message] = [
        user_message("hello"),
        ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="x")], is_error=True),
    ]
    decoded = msgspec.json.decode(msgspec.json.encode(messages), type=list[Message])
    assert decoded == messages


def test_context_tokens_add_all_token_kinds() -> None:
    assert Usage(input=1, output=2, cache_read=3, cache_write=4).context_tokens == 10
