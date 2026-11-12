"""Selects the transcript part that can be sent back to a model."""

from .models import Model
from .types import AssistantMessage, Message, TextContent, ToolCall, ToolResultMessage

INTERRUPTED = "No result: the run was interrupted before this tool call finished."


def replayable(messages: list[Message]) -> list[Message]:
    """Drop failed or aborted replies and tool results whose call is not in a kept reply, and answer calls that have
    no result.

    Providers reject a tool result without its call, and a call without its result. A failed reply can hold half a
    tool call; a call loses its result when the process stops mid-run or a session is forked between the two.
    """
    kept: list[Message] = []
    waiting: dict[str, ToolCall] = {}  # calls of the last kept reply without a result yet
    for message in messages:
        if isinstance(message, ToolResultMessage):
            if waiting.pop(message.tool_call_id, None) is not None:
                kept.append(message)
            continue
        kept += _unanswered(waiting)
        waiting = {}
        if isinstance(message, AssistantMessage):
            if message.stop_reason in ("error", "aborted"):
                continue
            waiting = {call.id: call for call in message.tool_calls()}
        kept.append(message)
    return kept + _unanswered(waiting)


def _unanswered(calls: dict[str, ToolCall]) -> list[Message]:
    return [
        ToolResultMessage(
            tool_call_id=call.id, tool_name=call.name, content=[TextContent(text=INTERRUPTED)], is_error=True
        )
        for call in calls.values()
    ]


def same_model(message: AssistantMessage, model: Model) -> bool:
    """Whether provider-specific replay data on `message` (signatures, reasoning fields) fits `model`."""
    return message.provider == model.provider and message.model == model.id
