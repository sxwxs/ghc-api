"""Line REPL: a protocol client of an in-process server. The event loop runs in a background thread, so the main
thread can block in input() and receive Ctrl+C, which aborts the active run."""

import argparse
import asyncio
import datetime
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from .. import __version__
from ..ai.types import Message
from ..client.client import Client, RemoteError
from ..client.transports import connect_in_process
from ..coding.session_events import EventEnvelope, RunEnd, SessionState, StateChanged
from ..server.server import Server
from . import options
from .printers import TextPrinter

T = TypeVar("T")

HELP = """Commands:
  /new                      start a new session with the current model
  /sessions                 list recent sessions in this directory
  /resume <id>              resume a session
  /model [provider [model]] show models, or switch
  /thinking [level]         show or set the reasoning level (off, low, medium, high, max)
  /name <name>              name this session
  /state                    show model, context use and cost
  /quit                     exit (also Ctrl+D; Ctrl+Z then Enter on Windows)
Ctrl+C aborts a running reply."""


def run_repl(args: argparse.Namespace) -> int:
    """Run the REPL until the user quits."""
    return Repl(args).run()


class Repl:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session_setup, self.cwd = options.setup(args)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="pyagent-loop", daemon=True)
        self.printer = TextPrinter()
        self.run_ended = threading.Event()

    def run(self) -> int:
        self.thread.start()
        self.client, self.server = self._call(self._connect())
        try:
            self._call(self.client.initialize("pyagent-repl"))
            self.state = self._call(options.open_session(self.client, self.args, self.cwd))
            self.client.on_event(self._on_event)
            model = self.state.model
            print(f"pyagent {__version__} | {model.provider}/{model.id} | thinking {self.state.thinking}")
            print(f"session {self.state.session_id} | /help for commands, Ctrl+C aborts a reply")
            self._loop()
        finally:
            self._call(self.server.close())
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
        return 0

    async def _connect(self) -> tuple[Client, Server]:
        # The server's queue and tasks belong to the loop thread, so they are created there.
        return connect_in_process(self.session_setup, self.cwd)

    def _loop(self) -> None:
        while True:
            try:
                line = input("> ").strip()
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                return
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not self._command(line):
                        return
                else:
                    self._prompt(line)
            except RemoteError as error:
                print(f"error: {error.message}")

    def _prompt(self, text: str) -> None:
        self.run_ended.clear()
        self._call(self.client.prompt(self.state.session_id, text))
        try:
            # Short waits keep the main thread responsive to Ctrl+C on every platform.
            while not self.run_ended.wait(0.1):
                pass
        except KeyboardInterrupt:
            self._call(self.client.abort(self.state.session_id))
            self.run_ended.wait()

    def _command(self, line: str) -> bool:
        name, *rest = line.split(maxsplit=2)
        match name:
            case "/quit" | "/exit":
                return False
            case "/help":
                print(HELP)
            case "/new":
                model = self.state.model
                self._switch(
                    self._call(self.client.new_session(self.cwd, model.provider, model.id, self.state.thinking))
                )
            case "/sessions":
                self._list_sessions()
            case "/resume" if rest:
                self._switch(self._call(self.client.load_session(rest[0])))
            case "/model":
                self._model(rest)
            case "/thinking" if rest and rest[0] in ("off", "low", "medium", "high", "max"):
                self.state = self._call(self.client.set_thinking(self.state.session_id, rest[0]))
                print(f"thinking {self.state.thinking}")
            case "/thinking":
                print(f"thinking {self.state.thinking} (set one of: off, low, medium, high, max)")
            case "/name" if rest:
                self.state = self._call(self.client.set_name(self.state.session_id, " ".join(rest)))
            case "/state":
                self._show_state()
            case _:
                print(f"unknown or incomplete command: {line} (see /help)")
        return True

    def _switch(self, state: SessionState) -> None:
        self.state = state
        print(
            f"session {state.session_id} | {state.model.provider}/{state.model.id} | {len(self._messages())} messages"
        )

    def _messages(self) -> list[Message]:
        return self._call(self.client.messages(self.state.session_id))

    def _list_sessions(self) -> None:
        sessions = self._call(self.client.list_sessions(self.cwd))
        if not sessions:
            print("no saved sessions in this directory")
        for summary in sessions[:15]:
            when = datetime.datetime.fromtimestamp(summary.updated_at / 1000).strftime("%Y-%m-%d %H:%M")
            title = summary.name or (summary.first_message or "").replace("\n", " ")[:60]
            print(f"  {summary.session_id}  {when}  {summary.message_count:>3} msgs  {title}")

    def _model(self, rest: list[str]) -> None:
        if not rest:
            current = self.state.model
            for info in self._call(self.client.models()):
                mark = "*" if (info.provider, info.id) == (current.provider, current.id) else " "
                status = "" if info.available else "  (no credentials)"
                print(f" {mark} {info.provider} {info.id}{status}")
            return
        self.state = self._call(
            self.client.set_model(self.state.session_id, rest[0], rest[1] if len(rest) > 1 else None)
        )
        print(f"model {self.state.model.provider}/{self.state.model.id}")

    def _show_state(self) -> None:
        state = self.state
        share = state.context_tokens / state.context_window if state.context_window else 0
        print(f"session  {state.session_id}{f' ({state.name})' if state.name else ''}")
        print(f"model    {state.model.provider}/{state.model.id}, thinking {state.thinking}")
        print(f"context  {state.context_tokens:,} / {state.context_window:,} tokens ({share:.1%})")
        usage = state.usage
        print(f"usage    {usage.input:,} in, {usage.output:,} out, {usage.cache_read:,} cached", end="")
        print(f", ${usage.cost.total:.4f}")
        print(f"file     {state.file or '(not saved)'}")

    def _on_event(self, envelope: EventEnvelope) -> None:
        # Runs on the loop thread.
        if envelope.session_id != self.state.session_id:
            return
        self.printer(envelope)
        match envelope.event:
            case StateChanged(state=state):
                self.state = state
            case RunEnd():
                self.run_ended.set()
            case _:
                pass

    def _call(self, coroutine: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()
