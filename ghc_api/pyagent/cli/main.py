"""Command-line entry: one-shot runs (text or JSON events), the REPL, or the protocol server on stdio."""

import argparse
import asyncio
import io
import sys

from ..ai.models import ModelError
from ..client.client import RemoteError
from ..client.transports import connect_in_process
from ..coding.config import ConfigError
from ..coding.session_events import EventEnvelope, RunEnd
from ..coding.session_store import SessionError
from ..server.stdio import serve_stdio
from . import options
from .printers import JsonPrinter, TextPrinter
from .repl import run_repl


def main(argv: list[str] | None = None) -> None:
    """Run the CLI. Exit codes: 0 success, 1 the run failed, 2 usage or setup error, 130 interrupted."""
    args = options.parser().parse_args(argv)
    # Pipes default to the locale encoding on Windows, which cannot encode every reply.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        sys.exit(_dispatch(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except (ModelError, ConfigError, SessionError, RemoteError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)


def _dispatch(args: argparse.Namespace) -> int:
    if args.mode == "rpc":
        session_setup, cwd = options.setup(args)
        asyncio.run(serve_stdio(session_setup, cwd))
        return 0
    prompt = " ".join(args.prompt)
    if not prompt and (args.print or not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    if prompt.strip():
        return asyncio.run(_run_once(args, prompt))
    if args.print or args.mode == "json":
        print("error: no prompt given", file=sys.stderr)
        return 2
    return run_repl(args)


async def _run_once(args: argparse.Namespace, prompt: str) -> int:
    session_setup, cwd = options.setup(args)
    client, server = connect_in_process(session_setup, cwd)
    try:
        await client.initialize("pyagent-cli")
        state = await options.open_session(client, args, cwd)
        printer = JsonPrinter() if args.mode == "json" else TextPrinter()
        ended: asyncio.Future[RunEnd] = asyncio.get_running_loop().create_future()

        def on_event(envelope: EventEnvelope) -> None:
            printer(envelope)
            if isinstance(envelope.event, RunEnd) and not ended.done():
                ended.set_result(envelope.event)

        client.on_event(on_event)
        await client.prompt(state.session_id, prompt)
        end = await ended
        return 0 if end.reason == "completed" else 1
    finally:
        await server.close()
