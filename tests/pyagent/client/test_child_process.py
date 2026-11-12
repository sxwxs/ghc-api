"""A real `pyagent --mode rpc` child process over stdio, against a local fake OpenAI-compatible endpoint."""

import asyncio
import json
import os
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ghc_api.pyagent.client.transports import ChildProcess
from ghc_api.pyagent.coding.session_events import EventEnvelope, RunEnd

pytestmark = pytest.mark.anyio


class FakeChat(BaseHTTPRequestHandler):
    """Answers every chat completion with one streamed text reply."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for part in ["Hello ", "from the\u2028fake"]:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n\n".encode())
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def endpoint() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


async def test_a_child_process_serves_the_protocol_on_stdio(tmp_path: Path, endpoint: str) -> None:
    config = {
        "defaultProvider": "fake",
        "defaultModel": "fake-1",
        "providers": {"fake": {"api": "openai_chat", "baseUrl": endpoint}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    env = {**os.environ, "PYAGENT_HOME": str(tmp_path)}
    child = await ChildProcess.start(
        [sys.executable, "-m", "ghc_api.pyagent", "--mode", "rpc", "--cwd", str(tmp_path)], env=env
    )
    events: list[EventEnvelope] = []
    ended = asyncio.Event()

    def on_event(envelope: EventEnvelope) -> None:
        events.append(envelope)
        if isinstance(envelope.event, RunEnd):
            ended.set()

    child.client.on_event(on_event)
    try:
        await asyncio.wait_for(child.client.initialize("test"), 30)
        state = await child.client.new_session()
        await child.client.prompt(state.session_id, "hi")
        await asyncio.wait_for(ended.wait(), 30)
        messages = await child.client.messages(state.session_id)
    finally:
        code = await asyncio.wait_for(child.close(), 30)

    assert code == 0
    assert messages[-1].content[0].text == "Hello from the\u2028fake"  # type: ignore[union-attr]
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert list((tmp_path / "sessions").glob("*/*.jsonl"))
