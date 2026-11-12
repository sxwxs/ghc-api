"""Serves the protocol on stdin/stdout, one JSON-RPC message per line. Diagnostics go to stderr."""

import asyncio
import sys
import threading

from ..coding.agent_session import SessionSetup
from .server import Server


async def serve_stdio(setup: SessionSetup, cwd: str) -> None:
    """Serve until stdin closes, then abort active runs and return."""
    loop = asyncio.get_running_loop()
    closed = asyncio.Event()
    out = sys.stdout.buffer

    def send(line: bytes) -> None:
        out.write(line)
        out.flush()

    server = Server(send, setup, cwd)

    def read() -> None:
        # A thread, because Windows event loops cannot watch stdin. Iterating splits on b"\n" only.
        for line in sys.stdin.buffer:
            loop.call_soon_threadsafe(server.receive, line)
        loop.call_soon_threadsafe(closed.set)

    threading.Thread(target=read, name="stdin-reader", daemon=True).start()
    await closed.wait()
    await server.close()
