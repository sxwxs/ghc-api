"""Tests for ghc_api.sse base handler.

The base class owns the SSE pipeline (line iteration, raw-event capture, cache
lifecycle, error arms). These tests exercise it via the AnthropicDirectStreamHandler
subclass since the base alone has no concrete endpoint string.
"""

import json
import unittest
from unittest import mock

import requests

from ghc_api.cache import RequestCache
from ghc_api.sse import AnthropicDirectStreamHandler, OpenAIResponsesStreamHandler
from ghc_api.sse import base as base_module


class _FakeResponse:
    """A minimal stand-in for requests.Response with a controllable iter_lines."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        pass


def _collect(generator):
    """Drain a generator into a list."""
    return list(generator)


class SSEBasePassthroughTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()

    def _make_handler(self, lines, status_code=200):
        return AnthropicDirectStreamHandler(
            response=_FakeResponse(lines, status_code=status_code),
            request_id="req-1",
            request_size=42,
            start_time=0.0,
            original_model="claude-opus-4",
            translated_model="claude-opus-4",
            request_body_for_cache={"model": "claude-opus-4"},
            original_request_body=None,
            request_headers={},
            client_ip="1.2.3.4",
            user_id="anonymous",
        )

    def test_passthrough_forwards_every_data_line_verbatim(self):
        message_start = json.dumps({"type": "message_start", "message": {"model": "claude-opus-4", "usage": {"input_tokens": 5}}})
        delta = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        message_delta = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}})
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            b"event: message_delta",
            f"data: {message_delta}".encode(),
            b"data: [DONE]",
        ]
        handler = self._make_handler(lines)
        out = "".join(_collect(handler._generate()))
        # Every upstream event_type/data pair appears in the output.
        self.assertIn("event: message_start\n", out)
        self.assertIn(f"data: {message_start}\n", out)
        self.assertIn("event: content_block_delta\n", out)
        self.assertIn(f"data: {delta}\n", out)
        self.assertIn(f"data: {message_delta}\n", out)
        # Anthropic /v1/messages must NOT forward the OpenAI-style [DONE]
        # sentinel -- it signals end-of-stream via message_stop.
        self.assertNotIn("data: [DONE]\n\n", out)

    def test_cache_captures_raw_events_verbatim(self):
        message_start = json.dumps({"type": "message_start", "message": {"model": "claude-opus-4", "usage": {"input_tokens": 11, "cache_read_input_tokens": 3}}})
        delta = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        message_delta = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}})
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            b"event: message_delta",
            f"data: {message_delta}".encode(),
            b"data: [DONE]",
        ]
        handler = self._make_handler(lines)
        _collect(handler._generate())

        entry = self.cache.get_request("req-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["raw_events"], [message_start, delta, message_delta])
        self.assertIsNone(entry["response_body"])
        self.assertEqual(entry["input_tokens"], 11)
        self.assertEqual(entry["cache_read_input_tokens"], 3)
        self.assertEqual(entry["output_tokens"], 9)
        self.assertEqual(entry["state"], RequestCache.STATE_COMPLETED)

    def test_malformed_json_is_preserved_but_does_not_break_stream(self):
        good = json.dumps({"type": "message_start", "message": {"usage": {}}})
        lines = [
            b"event: message_start",
            f"data: {good}".encode(),
            b"data: not-json-at-all",
            b"data: [DONE]",
        ]
        handler = self._make_handler(lines)
        out = "".join(_collect(handler._generate()))
        self.assertIn("data: not-json-at-all\n\n", out)
        entry = self.cache.get_request("req-1")
        self.assertEqual(entry["raw_events"], [good, "not-json-at-all"])

    def test_generator_exit_marks_cache_error_499(self):
        message_start = json.dumps({"type": "message_start", "message": {"usage": {}}})
        delta = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            # Never reaches [DONE] -- simulate client disconnect.
        ]
        handler = self._make_handler(lines)
        gen = handler._generate()
        next(gen)
        gen.close()
        entry = self.cache.get_request("req-1")
        self.assertEqual(entry["status_code"], 499)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)

    def test_read_timeout_sets_504_and_completes_cache(self):
        lines = [
            requests.exceptions.ReadTimeout("upstream slow"),
        ]
        handler = self._make_handler(lines)
        _collect(handler._generate())
        entry = self.cache.get_request("req-1")
        # complete_request stores the chosen status_code; STATE_ERROR because >= 400.
        self.assertEqual(entry["status_code"], 504)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)


class OpenAIResponsesPassthroughTest(unittest.TestCase):
    """The /v1/responses handler uses a different SSE convention -- ``event:``
    lines are forwarded verbatim, ``data:`` lines as bare ``data: JSON\\n\\n``."""

    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()

    def test_event_lines_pass_through_and_data_is_bare(self):
        completed = json.dumps({
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 12, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 2}}},
        })
        lines = [
            b"event: response.created",
            b"data: {\"type\":\"response.created\"}",
            b"event: response.completed",
            f"data: {completed}".encode(),
            b"data: [DONE]",
        ]
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse(lines),
            request_id="req-2",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
            original_request_body=None,
            request_headers={},
            client_ip="1.2.3.4",
            user_id="anonymous",
        )
        out = "".join(_collect(handler._generate()))
        self.assertIn("event: response.created\n", out)
        self.assertIn("event: response.completed\n", out)
        self.assertIn(f"data: {completed}\n\n", out)
        self.assertIn("data: [DONE]\n\n", out)

        entry = self.cache.get_request("req-2")
        self.assertEqual(entry["input_tokens"], 12)
        self.assertEqual(entry["output_tokens"], 4)
        self.assertEqual(entry["cache_creation_input_tokens"], 2)


class SSEKeepaliveIntegrationTest(unittest.TestCase):
    """The base handler must translate an idle stream into a client keepalive.
    AnthropicDirectStreamHandler emits an Anthropic ``ping`` event."""

    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()

    def test_idle_stream_emits_anthropic_ping(self):
        import time as _time

        message_start = json.dumps({"type": "message_start", "message": {"usage": {}}})

        class _SlowResponse:
            status_code = 200
            ok = True
            text = ""

            def iter_lines(self):
                # Idle long enough to trip the 0.1s keepalive before the line.
                _time.sleep(0.25)
                yield f"data: {message_start}".encode()
                yield b"data: [DONE]"

            def close(self):
                pass

        handler = AnthropicDirectStreamHandler(
            response=_SlowResponse(),
            request_id="req-ka",
            request_size=10,
            start_time=0.0,
            original_model="claude-opus-4",
            translated_model="claude-opus-4",
            request_body_for_cache={"model": "claude-opus-4"},
        )
        with mock.patch.object(base_module.state, "sse_keepalive_interval", 0.1):
            out = "".join(_collect(handler._generate()))
        self.assertIn('event: ping\ndata: {"type": "ping"}\n\n', out)
        # The real event still comes through after the ping.
        self.assertIn(f"data: {message_start}\n", out)


if __name__ == "__main__":
    unittest.main()
