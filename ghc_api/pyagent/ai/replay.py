"""Selects the transcript part that can be sent back to a model."""

from .models import Model
from .types import AssistantMessage, Message, ToolResultMessage


def replayable(messages: list[Message]) -> list[Message]:
    """Drop failed or aborted replies and tool results whose call is not in a kept reply.

    A failed reply can hold half a tool call; providers reject a tool result without its call.
    """
    kept: list[Message] = []
    calls: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            if message.stop_reason in ("error", "aborted"):
                continue
            calls.update(call.id for call in message.tool_calls())
        elif isinstance(message, ToolResultMessage) and message.tool_call_id not in calls:
            continue
        kept.append(message)
    return kept


def same_model(message: AssistantMessage, model: Model) -> bool:
    """Whether provider-specific replay data on `message` (signatures, reasoning fields) fits `model`."""
    return message.provider == model.provider and message.model == model.id
