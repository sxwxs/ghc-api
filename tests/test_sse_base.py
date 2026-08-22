"""Tests for ghc_api.sse base handler.

The base class owns the SSE pipeline (line iteration, raw-event capture, cache
lifecycle, error arms). These tests exercise it via the AnthropicDirectStreamHandler
subclass since the base alone has no concrete endpoint string.
"""

import json
import threading
import time
import unittest
from unittest import mock

import requests

from ghc_api.cache import RequestCache
from ghc_api.sse import (
    AnthropicDirectStreamHandler,
    OpenAIResponsesStreamHandler,
    RetryingResponsesResponse,
)
from ghc_api.sse import base as base_module


class _FakeResponse:
    """A minimal stand-in for requests.Response with a controllable iter_lines."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        self.closed = True


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

    def _make_handler(self, lines, status_code=200, **kwargs):
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
            **kwargs,
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
        message_stop = json.dumps({"type": "message_stop"})
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            b"event: message_delta",
            f"data: {message_delta}".encode(),
            b"event: message_stop",
            f"data: {message_stop}".encode(),
            b"data: [DONE]",
        ]
        handler = self._make_handler(lines)
        _collect(handler._generate())

        entry = self.cache.get_request("req-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["raw_events"], [message_start, delta, message_delta, message_stop])
        self.assertIsNone(entry["response_body"])
        self.assertEqual(entry["input_tokens"], 11)
        self.assertEqual(entry["cache_read_input_tokens"], 3)
        self.assertEqual(entry["output_tokens"], 9)
        self.assertEqual(entry["state"], RequestCache.STATE_COMPLETED)

    def test_malformed_json_is_preserved_but_does_not_break_stream(self):
        good = json.dumps({"type": "message_start", "message": {"usage": {}}})
        stop = json.dumps({"type": "message_stop"})
        lines = [
            b"event: message_start",
            f"data: {good}".encode(),
            b"data: not-json-at-all",
            b"event: message_stop",
            f"data: {stop}".encode(),
            b"data: [DONE]",
        ]
        handler = self._make_handler(lines)
        out = "".join(_collect(handler._generate()))
        self.assertIn("data: not-json-at-all\n\n", out)
        entry = self.cache.get_request("req-1")
        self.assertEqual(entry["raw_events"], [good, "not-json-at-all", stop])

    def test_clean_eof_without_message_stop_is_treated_as_truncated(self):
        """An Anthropic stream that ends cleanly before ``message_stop`` is a
        truncated upstream body: the client gets an ``error`` event and the
        cache record must not claim success."""
        start = json.dumps({"type": "message_start", "message": {"usage": {}}})
        delta = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}})
        lines = [
            b"event: message_start",
            f"data: {start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
        ]
        handler = self._make_handler(lines)
        output = "".join(_collect(handler._generate()))

        self.assertIn("event: error\n", output)
        entry = self.cache.get_request("req-1")
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)
        self.assertEqual(entry["stream_error"]["category"], "upstream_stream_error")

    def test_upstream_error_event_is_terminal_for_anthropic(self):
        """A definitive upstream ``error`` event terminates an Anthropic stream;
        a transport failure right after it must not produce a second error
        event or downgrade the request."""
        upstream_error = json.dumps({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        })
        lines = [
            b"event: error",
            f"data: {upstream_error}".encode(),
            requests.exceptions.ChunkedEncodingError("missing final chunk"),
        ]
        handler = self._make_handler(lines)
        output = "".join(_collect(handler._generate()))

        # Exactly one error event (the forwarded upstream one), no synthesized
        # transport error appended after it.
        self.assertEqual(output.count("event: error"), 1)
        self.assertIn(upstream_error, output)

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
        self.assertEqual(entry["raw_events"], [message_start])
        # request_count means terminal request attempts, not successful
        # responses.  A 499 is retained once for diagnostics and statistics.
        self.assertEqual(self.cache.request_count, 1)
        self.assertEqual(self.cache.model_stats["claude-opus-4"]["request_count"], 1)
        self.assertEqual(self.cache.endpoint_stats["/v1/messages"]["request_count"], 1)
        self.assertEqual(self.cache.user_totals["anonymous"]["request_count"], 1)
        handler._complete_cache()
        self.assertEqual(self.cache.request_count, 1)

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

    def test_raw_capture_limit_bounds_memory_without_truncating_client_stream(self):
        first = json.dumps({"type": "ping", "value": "first"})
        second = json.dumps({"type": "ping", "value": "second"})
        handler = self._make_handler(
            [f"data: {first}".encode(), f"data: {second}".encode()],
            max_raw_capture_bytes=len(first.encode("utf-8")),
        )

        output = "".join(handler._generate())

        self.assertIn(first, output)
        self.assertIn(second, output)
        self.assertTrue(handler.raw_capture_truncated)
        self.assertEqual(handler.raw_events, [first])
        entry = self.cache.get_request("req-1")
        self.assertEqual(
            entry["response_size"],
            len(first.encode("utf-8")) + len(second.encode("utf-8")),
        )

    def test_raw_capture_limit_counts_both_retained_stream_views(self):
        payload = json.dumps({"type": "ping", "value": "fixture"})
        wire_line = f"data: {payload}"
        retained_bytes = len(wire_line.encode("utf-8")) + len(payload.encode("utf-8"))
        handler = self._make_handler(
            [wire_line.encode("utf-8")],
            max_raw_capture_bytes=retained_bytes,
        )
        handler.capture_raw_sse_lines = True

        output = "".join(handler._generate())

        self.assertIn(payload, output)
        self.assertFalse(handler.raw_capture_truncated)
        self.assertEqual(handler.raw_sse_lines, [wire_line])
        self.assertEqual(handler.raw_events, [payload])
        self.assertEqual(handler.raw_capture_bytes, retained_bytes)


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

    def test_incomplete_response_records_terminal_usage(self):
        incomplete = json.dumps({
            "type": "response.incomplete",
            "response": {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 1},
                }
            },
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: response.incomplete",
                f"data: {incomplete}".encode(),
            ]),
            request_id="req-incomplete",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        _collect(handler._generate())

        entry = self.cache.get_request("req-incomplete")
        self.assertEqual(entry["input_tokens"], 7)
        self.assertEqual(entry["output_tokens"], 3)
        self.assertEqual(entry["cache_creation_input_tokens"], 1)

    def test_terminal_response_failed_is_recorded_as_error(self):
        failed = json.dumps({
            "type": "response.failed",
            "response": {"error": None, "status": "failed"},
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: response.failed",
                f"data: {failed}".encode(),
            ]),
            request_id="req-failed",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        _collect(handler._generate())

        entry = self.cache.get_request("req-failed")
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)

    def test_chunked_body_failure_emits_standard_error_and_records_502(self):
        created = json.dumps({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"status": "in_progress"},
        })
        delta = json.dumps({
            "type": "response.function_call_arguments.delta",
            "sequence_number": 1,
            "delta": '{"path":"partial',
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: response.created",
                f"data: {created}".encode(),
                b"event: response.function_call_arguments.delta",
                f"data: {delta}".encode(),
                requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            ]),
            request_id="req-truncated-chunk",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        output = "".join(handler._generate())

        self.assertIn("event: error\n", output)
        error_data = output.split("event: error\ndata: ", 1)[1].split("\n\n", 1)[0]
        error = json.loads(error_data)
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["code"], "upstream_stream_error")
        self.assertEqual(error["sequence_number"], 2)
        entry = self.cache.get_request("req-truncated-chunk")
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)
        self.assertEqual(entry["stream_error"]["type"], "ChunkedEncodingError")
        self.assertEqual(entry["stream_error"]["category"], "upstream_stream_error")

    def test_chunked_failure_after_terminal_event_keeps_completed_result(self):
        completed = json.dumps({
            "type": "response.completed",
            "sequence_number": 7,
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: response.completed",
                f"data: {completed}".encode(),
                requests.exceptions.ChunkedEncodingError("missing final HTTP chunk"),
            ]),
            request_id="req-terminal-before-chunk-error",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        output = "".join(handler._generate())

        self.assertNotIn("event: error", output)
        entry = self.cache.get_request("req-terminal-before-chunk-error")
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["state"], RequestCache.STATE_COMPLETED)
        self.assertNotIn("stream_error", entry)
        self.assertEqual(entry["input_tokens"], 3)
        self.assertEqual(entry["output_tokens"], 2)

    def test_clean_eof_without_terminal_event_is_treated_as_truncated(self):
        created = json.dumps({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"status": "in_progress"},
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: response.created",
                f"data: {created}".encode(),
            ]),
            request_id="req-missing-terminal",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        output = "".join(handler._generate())

        self.assertIn("event: error\n", output)
        self.assertIn('"code": "upstream_stream_error"', output)
        entry = self.cache.get_request("req-missing-terminal")
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(
            entry["stream_error"]["type"], "UpstreamStreamProtocolError"
        )

    def test_terminal_error_event_is_recorded_as_error(self):
        """A chained ghc-api consumes the ``error`` event its upstream proxy
        emits after committing a 200. Without this the failure would be filed as
        200/completed and every success-rate metric would be wrong.
        """
        error = json.dumps({
            "type": "error",
            "code": "upstream_error",
            "message": "rate limited",
            "param": None,
            "sequence_number": 0,
        })
        handler = OpenAIResponsesStreamHandler(
            response=_FakeResponse([
                b"event: error",
                f"data: {error}".encode(),
            ]),
            request_id="req-error-event",
            request_size=10,
            start_time=0.0,
            original_model="gpt-5",
            translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

        _collect(handler._generate())

        entry = self.cache.get_request("req-error-event")
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)


class _LazyStreamResponse:
    """A ``requests.Response`` stand-in with real streaming semantics.

    ``_FakeResponse`` above has a plain ``text = ""`` attribute, so it cannot
    catch code that eagerly reads ``.text``. On a real ``stream=True`` response
    ``.text`` goes through ``.content``, which drains the socket and blocks
    until the upstream is done -- destroying streaming. This double reproduces
    that: reading ``.text`` consumes the line iterator and records the access.
    """

    def __init__(self, lines):
        self._lines = iter(lines)
        self._buffered = []
        self.status_code = 200
        self.ok = True
        self.text_accessed = False
        self.closed = False

    @property
    def text(self):
        self.text_accessed = True
        self._buffered.extend(self._lines)  # blocks until upstream finishes
        return b"\n".join(self._buffered).decode()

    def iter_lines(self):
        buffered, self._buffered = self._buffered, []
        yield from buffered
        yield from self._lines

    def close(self):
        self.closed = True


class RetryingResponsesResponseTest(unittest.TestCase):
    @staticmethod
    def _event(event_type, **extra):
        return f'data: {json.dumps({"type": event_type, **extra})}'.encode()

    def test_construction_does_not_read_the_streaming_body(self):
        """The wrapper is built inside the request handler, before Flask starts
        iterating. Touching ``.text`` there would block the route until the
        model finished generating and buffer the whole response in memory.
        """
        upstream = _LazyStreamResponse([self._event("response.created")])

        wrapper = RetryingResponsesResponse(upstream, mock.Mock(), 1, "req-lazy")

        self.assertFalse(
            upstream.text_accessed,
            "RetryingResponsesResponse must not read response.text; on a real "
            "streaming response that drains the body and kills streaming",
        )
        self.assertEqual(wrapper.status_code, 200)
        self.assertTrue(wrapper.ok)

    def test_output_reaches_the_client_before_upstream_finishes(self):
        """End-to-end liveness: a delta must be forwarded while the upstream is
        still generating, not after the stream closes.
        """
        upstream_still_generating = threading.Event()

        def lines():
            yield self._event("response.output_text.delta", delta="first")
            # Upstream keeps the connection open while the model thinks.
            upstream_still_generating.wait(10)
            yield self._event("response.output_text.delta", delta="second")

        upstream = _LazyStreamResponse(lines())
        wrapper = RetryingResponsesResponse(upstream, mock.Mock(), 1, "req-live")

        started = time.time()
        stream = wrapper.iter_lines()
        try:
            first = next(stream)
            elapsed = time.time() - started
        finally:
            upstream_still_generating.set()
            stream.close()

        self.assertEqual(first, self._event("response.output_text.delta", delta="first"))
        self.assertFalse(upstream.text_accessed)
        self.assertLess(
            elapsed, 5,
            "the first delta must be yielded immediately, not after the whole "
            "upstream stream has been consumed",
        )

    def test_retries_early_response_failed_without_forwarding_failed_attempt(self):
        first = _FakeResponse([
            b"event: response.created",
            self._event("response.created", marker="first"),
            b"event: response.failed",
            self._event("response.failed", response={"error": None}),
        ])
        second = _FakeResponse([
            b"event: response.created",
            self._event("response.created", marker="second"),
            b"event: response.output_text.delta",
            self._event("response.output_text.delta", delta="ok"),
            b"event: response.completed",
            self._event("response.completed", response={"usage": {}}),
        ])
        retry_count = 0

        def retry():
            nonlocal retry_count
            retry_count += 1
            return second

        response = RetryingResponsesResponse(first, retry, 1, "req-retry")
        output = list(response.iter_lines())

        self.assertEqual(retry_count, 1)
        self.assertTrue(first.closed)
        self.assertNotIn(self._event("response.created", marker="first"), output)
        self.assertFalse(any(b"response.failed" in line for line in output))
        self.assertIn(self._event("response.created", marker="second"), output)
        self.assertIn(self._event("response.output_text.delta", delta="ok"), output)

    def test_does_not_retry_after_output_has_started(self):
        first = _FakeResponse([
            self._event("response.created"),
            self._event("response.output_text.delta", delta="partial"),
            self._event("response.failed", response={"error": None}),
        ])
        retry = mock.Mock()

        output = list(RetryingResponsesResponse(first, retry, 3, "req-partial").iter_lines())

        retry.assert_not_called()
        self.assertTrue(any(b"response.failed" in line for line in output))

    def test_retries_chunked_encoding_error_before_output_starts(self):
        first = _FakeResponse([
            self._event("response.created", marker="first"),
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
        ])
        second = _FakeResponse([
            self._event("response.created", marker="second"),
            self._event("response.output_text.delta", delta="ok"),
        ])
        retry = mock.Mock(return_value=second)

        output = list(
            RetryingResponsesResponse(first, retry, 1, "req-chunk-retry").iter_lines()
        )

        retry.assert_called_once_with()
        self.assertTrue(first.closed)
        self.assertNotIn(self._event("response.created", marker="first"), output)
        self.assertIn(self._event("response.created", marker="second"), output)
        self.assertIn(self._event("response.output_text.delta", delta="ok"), output)

    def test_retries_clean_eof_before_output_starts(self):
        first = _FakeResponse([
            self._event("response.created", marker="first"),
        ])
        second = _FakeResponse([
            self._event("response.created", marker="second"),
            self._event("response.output_text.delta", delta="ok"),
        ])
        retry = mock.Mock(return_value=second)

        output = list(
            RetryingResponsesResponse(first, retry, 1, "req-eof-retry").iter_lines()
        )

        retry.assert_called_once_with()
        self.assertTrue(first.closed)
        self.assertNotIn(self._event("response.created", marker="first"), output)
        self.assertIn(self._event("response.created", marker="second"), output)

    def test_does_not_retry_chunked_encoding_error_after_output_starts(self):
        partial = self._event("response.output_text.delta", delta="partial")
        first = _FakeResponse([
            self._event("response.created"),
            partial,
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
        ])
        retry = mock.Mock()
        stream = RetryingResponsesResponse(
            first, retry, 3, "req-partial-chunk"
        ).iter_lines()

        self.assertEqual(next(stream), self._event("response.created"))
        self.assertEqual(next(stream), partial)
        with self.assertRaises(requests.exceptions.ChunkedEncodingError):
            next(stream)
        retry.assert_not_called()

    def test_yields_buffered_preamble_before_raising_unretryable_transport_error(self):
        """When a pre-output transport failure cannot be retried (budget
        exhausted), the buffered preamble is still delivered before the error
        propagates -- matching the early_failure path's contract."""
        created = self._event("response.created")
        first = _FakeResponse([
            created,
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
        ])
        retry = mock.Mock(return_value=None)
        stream = RetryingResponsesResponse(
            first, retry, 0, "req-no-budget"
        ).iter_lines()

        self.assertEqual(next(stream), created)
        with self.assertRaises(requests.exceptions.ChunkedEncodingError):
            next(stream)

    def test_forwards_failure_after_retry_budget_is_exhausted(self):
        failed_line = self._event("response.failed", response={"error": None})
        first = _FakeResponse([self._event("response.created"), failed_line])
        second = _FakeResponse([self._event("response.created"), failed_line])
        retry = mock.Mock(return_value=second)

        output = list(RetryingResponsesResponse(first, retry, 1, "req-exhausted").iter_lines())

        retry.assert_called_once_with()
        self.assertIn(failed_line, output)

    def test_does_not_retry_a_standard_error_event(self):
        """The proxy emits ``error`` (not ``response.failed``) for upstream errors
        it observes after committing a 200. ``error`` is terminal: replaying it
        would turn one client request into ``max_retries + 1`` upstream calls for
        a request that can never succeed.
        """
        error_line = self._event(
            "error", code="upstream_error", message="bad tool schema",
            param=None, sequence_number=0,
        )
        first = _FakeResponse([b"event: error", error_line])
        retry = mock.Mock()

        output = list(RetryingResponsesResponse(first, retry, 3, "req-error").iter_lines())

        retry.assert_not_called()
        self.assertIn(error_line, output)

    def test_disconnect_while_retry_is_pending_closes_retry_response(self):
        first = _FakeResponse([
            self._event("response.created"),
            self._event("response.failed", response={"error": None}),
        ])
        second = _FakeResponse([])
        retry_started = threading.Event()
        release_retry = threading.Event()
        iteration_done = threading.Event()

        def retry():
            retry_started.set()
            release_retry.wait(1)
            return second

        wrapper = RetryingResponsesResponse(first, retry, 1, "req-cancel-retry")

        def consume():
            list(wrapper.iter_lines())
            iteration_done.set()

        thread = threading.Thread(target=consume)
        thread.start()
        self.assertTrue(retry_started.wait(1))

        wrapper.close()
        release_retry.set()
        self.assertTrue(iteration_done.wait(1))
        thread.join(1)

        self.assertTrue(first.closed)
        self.assertTrue(second.closed)


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
