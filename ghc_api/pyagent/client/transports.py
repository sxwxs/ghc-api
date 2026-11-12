"""Ways to reach a server: in this process, or a `pyagent --mode rpc` child process over its stdin and stdout."""

import asyncio
import sys
from collections.abc import Mapping

from ..coding.agent_session import SessionSetup
from ..server.server import Server
from .client import Client

# Lines can carry whole transcripts (session/messages), far beyond asyncio's 64KB default.
MAX_LINE = 64 * 1024 * 1024
# The pyagent package's import path, wherever it is vendored: `python -m <PACKAGE>` runs the CLI.
PACKAGE = __name__.rsplit(".", 2)[0]


def connect_in_process(setup: SessionSetup, cwd: str) -> tuple[Client, Server]:
    """A client wired to a server in this event loop. Messages still travel as encoded lines, like on any transport."""
    client = Client(lambda line: server.receive(line))
    server = Server(client.receive, setup, cwd)
    return client, server


class ChildProcess:
    """A client connected to a `pyagent --mode rpc` child process."""

    def __init__(self, client: Client, process: asyncio.subprocess.Process, reader: "asyncio.Task[None]") -> None:
        self.client = client
        self.process = process
        self._reader = reader

    @classmethod
    async def start(
        cls, argv: list[str] | None = None, env: Mapping[str, str] | None = None, cwd: str | None = None
    ) -> "ChildProcess":
        """Start the server process; `argv` defaults to this interpreter running `pyagent --mode rpc`."""
        process = await asyncio.create_subprocess_exec(
            *(argv or [sys.executable, "-m", PACKAGE, "--mode", "rpc"]),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            limit=MAX_LINE,
        )
        assert process.stdin is not None and process.stdout is not None  # both were requested as pipes
        client = Client(process.stdin.write)
        return cls(client, process, asyncio.create_task(_read(process.stdout, client)))

    async def close(self) -> int:
        """Close the server's stdin, which aborts its runs and ends it; returns its exit code."""
        assert self.process.stdin is not None  # requested as a pipe
        self.process.stdin.close()
        code = await self.process.wait()
        await self._reader
        return code


async def _read(stdout: asyncio.StreamReader, client: Client) -> None:
    try:
        while line := await stdout.readline():
            client.receive(line)
    finally:
        client.close()
