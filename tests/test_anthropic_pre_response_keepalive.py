import json
import time
import unittest
from unittest import mock

from flask import Flask

from ghc_api.cache import RequestCache
from ghc_api.routes import anthropic as anthropic_module
from ghc_api.sse import base as base_module


class _FakeStreamResponse:
    status_code = 200
    ok = True
    text = ""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


def _collect_response(response):
    return "".join(
        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for chunk in response.response
    )


class DirectAnthropicPreResponseKeepaliveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.cache = RequestCache()
        self._patches = [
            mock.patch.object(anthropic_module, "cache", self.cache),
            mock.patch.object(base_module, "cache", self.cache),
            mock.patch.object(anthropic_module.state, "sse_keepalive_interval", 0.05),
            mock.patch.object(anthropic_module.state, "enable_tool_call_recovery", False),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()

    def _payload(self):
        return {
            "model": "claude-opus-4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "stream": True,
        }

    def _upstream_response(self):
        message_start = json.dumps({
            "type": "message_start",
            "message": {"model": "claude-opus-4", "usage": {"input_tokens": 1}},
        })
        return _FakeStreamResponse([
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"data: [DONE]",
        ])

    def test_slow_upstream_headers_emit_ping_before_real_events(self):
        def slow_post(*args, **kwargs):
            time.sleep(0.12)
            return self._upstream_response()

        with mock.patch.object(anthropic_module.requests, "post", side_effect=slow_post):
            with self.app.test_request_context("/v1/messages", method="POST"):
                response = anthropic_module.handle_direct_anthropic_request(
                    self._payload(),
                    request_id="req-pre-header",
                    start_time=time.time(),
                    original_model="claude-opus-4",
                    translated_model="claude-opus-4",
                    original_request_body=self._payload(),
                    request_headers={},
                )
                body = _collect_response(response)

        self.assertTrue(body.startswith('event: ping\ndata: {"type": "ping"}\n\n'))
        self.assertIn("event: message_start\n", body)
        self.assertIn('"type": "message_start"', body)

    def test_fast_upstream_headers_do_not_emit_pre_response_ping(self):
        with mock.patch.object(anthropic_module.requests, "post", return_value=self._upstream_response()):
            with self.app.test_request_context("/v1/messages", method="POST"):
                response = anthropic_module.handle_direct_anthropic_request(
                    self._payload(),
                    request_id="req-fast-header",
                    start_time=time.time(),
                    original_model="claude-opus-4",
                    translated_model="claude-opus-4",
                    original_request_body=self._payload(),
                    request_headers={},
                )
                body = _collect_response(response)

        self.assertNotIn('event: ping\ndata: {"type": "ping"}\n\n', body)
        self.assertIn("event: message_start\n", body)

    def test_translated_stream_slow_upstream_headers_emit_ping(self):
        def slow_post(*args, **kwargs):
            time.sleep(0.12)
            return _FakeStreamResponse([b"data: [DONE]"])

        with mock.patch.object(anthropic_module.requests, "post", side_effect=slow_post):
            with self.app.test_request_context("/v1/messages", method="POST"):
                response = anthropic_module.stream_anthropic_messages(
                    openai_payload={
                        "model": "gpt-5",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                    headers={},
                    request_id="req-translated-pre-header",
                    anthropic_payload=self._payload(),
                    request_size=10,
                    start_time=time.time(),
                    original_model="claude-via-openai",
                    translated_model="gpt-5",
                    original_request_body=self._payload(),
                    request_headers={},
                )
                body = _collect_response(response)

        self.assertTrue(body.startswith('event: ping\ndata: {"type": "ping"}\n\n'))


if __name__ == "__main__":
    unittest.main()
