"""
ACP (Agent Client Protocol) client implementation.

Provides a background asyncio event loop for managing agent subprocess I/O
from synchronous Flask request handlers.
"""

import asyncio
import platform
import threading
from typing import Any, Coroutine

_loop = None  # type: asyncio.AbstractEventLoop | None
_thread = None  # type: threading.Thread | None
_lock = threading.Lock()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create the shared background asyncio event loop."""
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(
                target=_loop.run_forever,
                daemon=True,
                name="acp-event-loop",
            )
            _thread.start()
        return _loop


def run_async(coro: Coroutine, timeout: float = 300) -> Any:
    """Run an async coroutine from synchronous code, blocking until complete."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
