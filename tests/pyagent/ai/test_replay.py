"""Which transcript part goes back to a model: failed replies out, every tool call answered exactly once."""

from ghc_api.pyagent.ai.replay import INTERRUPTED, replayable
from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    user_message,
)


def reply(*call_ids: str, stop_reason: StopReason = "tool_use") -> AssistantMessage:
    calls = [ToolCall(id=i, name="bash", arguments={}) for i in call_ids]
    return AssistantMessage(content=[*calls], provider="p", model="m", stop_reason=stop_reason)


def result(call_id: str) -> ToolResultMessage:
    return ToolResultMessage(tool_call_id=call_id, tool_name="bash", content=[TextContent(text="ok")])


def shape(messages: list[Message]) -> list[str]:
    return [
        f"result:{m.tool_call_id}:{'interrupted' if m.content[0] == TextContent(text=INTERRUPTED) else 'ok'}"
        if isinstance(m, ToolResultMessage)
        else ("assistant" if isinstance(m, AssistantMessage) else "user")
        for m in messages
    ]


def test_failed_replies_and_their_orphan_results_are_dropped() -> None:
    messages: list[Message] = [user_message("q"), reply("a", stop_reason="error"), result("a"), user_message("q2")]
    assert shape(replayable(messages)) == ["user", "user"]


def test_calls_left_without_results_are_answered_as_interrupted() -> None:
    # The process stopped after the reply was saved, before its tool results were.
    messages: list[Message] = [user_message("q"), reply("a", "b"), result("a"), user_message("next")]
    assert shape(replayable(messages)) == ["user", "assistant", "result:a:ok", "result:b:interrupted", "user"]
    # A fork that ends with the reply itself.
    assert shape(replayable([user_message("q"), reply("a")])) == ["user", "assistant", "result:a:interrupted"]


def test_complete_transcripts_are_unchanged() -> None:
    messages: list[Message] = [user_message("q"), reply("a"), result("a"), reply(stop_reason="stop")]
    assert replayable(messages) == messages
