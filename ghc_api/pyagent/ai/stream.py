"""Streams one assistant reply and emits its message events, with a single ending path for errors and aborts."""

import asyncio
from collections.abc import Callable
from contextlib import aclosing

from .events import MessageEnd, MessageEvent, MessageStart
from .models import Model
from .provider import ProviderFn, StreamOptions
from .providers import PROVIDERS
from .types import AssistantMessage, Context


def empty_reply(model: Model) -> AssistantMessage:
    """Create the message a reply streams into."""
    return AssistantMessage(content=[], provider=model.provider, model=model.id, stop_reason="stop")


async def stream_assistant(
    model: Model,
    context: Context,
    options: StreamOptions,
    message: AssistantMessage,
    emit: Callable[[MessageEvent], None],
    provider: ProviderFn | None = None,
) -> None:
    """Stream one reply into `message`, emitting message_start, block events and message_end.

    Provider exceptions end the message with stop_reason "error". Cancellation marks it "aborted",
    emits message_end, and re-raises.
    """
    emit(MessageStart(message=message))
    events = (provider or PROVIDERS[model.api])(model, context, options, message)
    try:
        async with aclosing(events):
            async for event in events:
                emit(event)
    except asyncio.CancelledError:
        message.stop_reason = "aborted"
        emit(MessageEnd(message=message))
        raise
    except Exception as error:
        message.stop_reason = "error"
        message.error_message = str(error) or type(error).__name__
    emit(MessageEnd(message=message))
