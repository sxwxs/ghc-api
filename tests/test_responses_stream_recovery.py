"""Recover broken Responses streams only before downstream-visible output."""

import json
import threading
import time
import unittest
from unittest import mock

import requests

from ghc_api.cache import RequestCache
from ghc_api.sse import OpenAIResponsesStreamHandler, RetryingResponsesResponse
from ghc_api.sse import base as base_module


def _event(event_type, **fields):
    return f'data: {json.dumps({"type": event_type, **fields})}'.encode()


def _reasoning(marker="hidden", event_type="response.output_item.added", **fields):
    return _event(event_type, item={
        "type": "reasoning",
        "encrypted_content": marker,
        "content": [],
        "summary": [],
    }, **fields)


class _Response:
    def __init__(self, lines, status_code=200):
        self.lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.closed = False

    def iter_lines(self):
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        self.closed = True


class ResponsesStreamRetryTest(unittest.TestCase):
    def test_recovers_transport_errors_after_only_encrypted_reasoning(self):
        for error_type in (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ):
            with self.subTest(error_type=error_type):
                first = _Response([
                    b"event: response.created",
                    _event("response.created", marker="discarded"),
                    _reasoning("discarded-thinking"),
                    _reasoning("discarded-thinking", "response.output_item.done"),
                    error_type("Response ended prematurely"),
                ])
                second = _Response([
                    _event("response.created", marker="accepted"),
                    _reasoning("accepted-thinking"),
                    _event("response.output_text.delta", delta="ok"),
                    _event("response.completed", response={"usage": {}}),
                ])
                retry = mock.Mock(return_value=second)
                wrapper = RetryingResponsesResponse(first, retry, 1, "req-recover")

                self.assertEqual(list(wrapper.iter_lines()), second.lines)
                retry.assert_called_once_with()
                self.assertTrue(first.closed)

    def test_long_reasoning_stream_like_incident_fits_default_buffer(self):
        # The incident had 69 events: 2 preamble events, 33 reasoning pairs,
        # then an unfinished 34th reasoning item, totalling roughly 0.5 MiB.
        lines = [_event("response.created"), _event("response.in_progress")]
        for index in range(34):
            lines.append(_reasoning("x" * 4164, output_index=index))
            if index < 33:
                lines.append(_reasoning("x" * 10600, "response.output_item.done", output_index=index))
        first = _Response(lines + [requests.exceptions.ChunkedEncodingError("Response ended prematurely")])
        second = _Response([_event("response.completed", response={"usage": {}})])
        retry = mock.Mock(return_value=second)

        self.assertEqual(list(RetryingResponsesResponse(first, retry, 1, "req-long").iter_lines()), second.lines)
        retry.assert_called_once_with()

    def test_disconnect_while_buffering_reasoning_does_not_start_retry(self):
        hidden_seen = threading.Event()
        release = threading.Event()
        done = threading.Event()
        output = []
        errors = []

        def lines():
            yield _reasoning()
            hidden_seen.set()
            release.wait(1)
            raise requests.exceptions.ChunkedEncodingError("broken after cancellation")

        first = _Response(lines())
        retry = mock.Mock()
        wrapper = RetryingResponsesResponse(first, retry, 3, "req-cancel-buffer")

        def consume():
            try:
                output.extend(wrapper.iter_lines())
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=consume)
        thread.start()
        try:
            self.assertTrue(hidden_seen.wait(1))
            wrapper.close()
            release.set()
            self.assertTrue(done.wait(1))
        finally:
            wrapper.close()
            release.set()
            thread.join(2)

        retry.assert_not_called()
        self.assertTrue(first.closed)
        self.assertEqual(output, [])
        self.assertEqual(errors, [])

    def test_retries_response_failed_after_only_encrypted_reasoning(self):
        first = _Response([
            _event("response.created"),
            _reasoning(),
            _event("response.failed", response={"error": None}),
        ])
        second = _Response([_event("response.completed", response={"usage": {}})])
        retry = mock.Mock(return_value=second)

        output = list(RetryingResponsesResponse(first, retry, 1, "req-failed").iter_lines())

        self.assertEqual(output, second.lines)
        retry.assert_called_once_with()

    def test_healthy_stream_preserves_every_line(self):
        lines = [
            b": upstream keepalive",
            b"event: response.created",
            _event("response.created"),
            b"",
            _reasoning(),
            _reasoning(event_type="response.output_item.done"),
            _event("response.output_text.delta", delta="hello"),
            _event("response.completed", response={"usage": {}}),
            b"data: [DONE]",
        ]
        retry = mock.Mock()

        output = list(RetryingResponsesResponse(_Response(lines), retry, 3, "req-ok").iter_lines())

        self.assertEqual(output, lines)
        retry.assert_not_called()

    def test_does_not_retry_after_visible_or_unknown_output(self):
        visible_events = [
            _event("response.output_text.delta", delta="partial"),
            _event("response.output_item.added", item={"type": "function_call", "name": "bash"}),
            _event("response.reasoning_summary_text.delta", delta="thinking"),
            _event("response.output_item.done", item={
                "type": "reasoning", "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "visible thinking"}],
            }),
            _event("response.output_item.added", item={
                "type": "reasoning", "content": [{"type": "text", "text": "visible"}],
            }),
            b"data:not-json",
            b"data: []",
        ]
        for visible in visible_events:
            with self.subTest(visible=visible):
                lines = [_event("response.created"), _reasoning(), visible]
                response = _Response(lines + [requests.exceptions.ChunkedEncodingError("broken")])
                retry = mock.Mock()
                output = []

                with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                    output.extend(RetryingResponsesResponse(response, retry, 3, "req-visible").iter_lines())

                self.assertEqual(output, lines)
                retry.assert_not_called()

    def test_standard_error_event_remains_terminal_after_hidden_reasoning(self):
        lines = [_reasoning(), _event("error", code="upstream_error", message="bad schema")]
        retry = mock.Mock()

        output = list(RetryingResponsesResponse(_Response(lines), retry, 3, "req-error").iter_lines())

        self.assertEqual(output, lines)
        retry.assert_not_called()

    def test_disabled_retry_budget_does_not_delay_hidden_reasoning(self):
        reasoning = _reasoning()

        def lines():
            yield reasoning
            raise AssertionError("the first line must be forwarded before reading further")

        stream = RetryingResponsesResponse(_Response(lines()), mock.Mock(), 0, "req-off").iter_lines()
        try:
            self.assertEqual(next(stream), reasoning)
        finally:
            stream.close()

    def test_buffer_limit_commits_stream_and_counts_empty_lines(self):
        for lines, limit in (([_reasoning("x" * 200)], 100), ([b"", b"", b""], 2)):
            with self.subTest(lines=lines):
                first = _Response(lines + [requests.exceptions.ChunkedEncodingError("broken")])
                retry = mock.Mock()
                output = []
                wrapper = RetryingResponsesResponse(first, retry, 3, "req-limit", max_buffer_bytes=limit)

                with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                    output.extend(wrapper.iter_lines())

                self.assertEqual(output, lines)
                retry.assert_not_called()

    def test_exhausted_retry_budget_preserves_last_attempt_and_error(self):
        first = _Response([
            _event("response.created", marker="first"),
            _reasoning("first"),
            requests.exceptions.ChunkedEncodingError("first broken"),
        ])
        last_lines = [_event("response.created", marker="last"), _reasoning("last")]
        last = _Response(last_lines + [requests.exceptions.ChunkedEncodingError("last broken")])
        retry = mock.Mock(return_value=last)
        output = []

        with self.assertRaisesRegex(requests.exceptions.ChunkedEncodingError, "last broken"):
            output.extend(RetryingResponsesResponse(first, retry, 1, "req-exhausted").iter_lines())

        retry.assert_called_once_with()
        self.assertEqual(output, last_lines)

    def test_unsuccessful_retry_preserves_original_stream_error(self):
        for result in (requests.exceptions.ReadTimeout("retry slow"), _Response([], 503)):
            with self.subTest(result=result):
                lines = [_event("response.created"), _reasoning()]
                first = _Response(lines + [requests.exceptions.ChunkedEncodingError("original broken")])
                retry = mock.Mock(side_effect=result) if isinstance(result, Exception) else mock.Mock(return_value=result)
                output = []

                with self.assertRaisesRegex(requests.exceptions.ChunkedEncodingError, "original broken"):
                    output.extend(RetryingResponsesResponse(first, retry, 1, "req-retry-error").iter_lines())

                retry.assert_called_once_with()
                self.assertEqual(output, lines)
                if not isinstance(result, Exception):
                    self.assertTrue(result.closed)

    def test_accepts_sse_data_without_optional_space(self):
        hidden = _reasoning().replace(b"data: ", b"data:", 1)
        first = _Response([hidden, requests.exceptions.ChunkedEncodingError("broken")])
        second = _Response([_event("response.completed", response={"usage": {}})])
        retry = mock.Mock(return_value=second)

        self.assertEqual(list(RetryingResponsesResponse(first, retry, 1, "req-space").iter_lines()), second.lines)
        retry.assert_called_once_with()


class ResponsesStreamErrorTest(unittest.TestCase):
    def setUp(self):
        self.cache = RequestCache()
        self.cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self.cache_patch.start()
        self.saved_interval = base_module.state.sse_keepalive_interval
        base_module.state.sse_keepalive_interval = 0

    def tearDown(self):
        base_module.state.sse_keepalive_interval = self.saved_interval
        self.cache_patch.stop()

    @staticmethod
    def _handler(response):
        return OpenAIResponsesStreamHandler(
            response=response, request_id="req-diagnostics", request_size=42,
            start_time=time.time(), original_model="gpt-5", translated_model="gpt-5",
            request_body_for_cache={"model": "gpt-5"},
        )

    @staticmethod
    def _error_event(output):
        return next(event for line in output.splitlines() if line.startswith("data: ")
                    and (event := json.loads(line[6:])).get("type") == "error")

    def test_broken_stream_emits_standard_error_and_persists_diagnostics(self):
        raw_lines = [_event("response.created", sequence_number=0), _reasoning(sequence_number=1)]
        handler = self._handler(_Response(raw_lines + [
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
        ]))

        output = "".join(handler._generate())
        error = self._error_event(output)
        entry = self.cache.get_request("req-diagnostics")

        self.assertEqual(error["code"], "upstream_stream_error")
        self.assertEqual(error["message"], "Response ended prematurely")
        self.assertEqual(error["sequence_number"], 2)
        self.assertIn("event: error\n", output)
        self.assertEqual(entry["status_code"], 502)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)
        self.assertEqual(entry["response_body"]["error"]["type"], "ChunkedEncodingError")
        self.assertEqual(entry["response_body"]["error"]["message"], error["message"])
        self.assertEqual(entry["raw_events"], [line[6:].decode() for line in raw_lines])

    def test_timeout_and_connection_errors_emit_standard_error(self):
        for error_type in (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            with self.subTest(error_type=error_type):
                output = "".join(self._handler(_Response([error_type("upstream unavailable")]))._generate())
                error = self._error_event(output)
                entry = self.cache.get_request("req-diagnostics")

                self.assertEqual(error["code"], "upstream_connection_error")
                self.assertEqual(entry["status_code"], 504)
                self.assertEqual(entry["response_body"]["error"]["type"], error_type.__name__)

    def test_proxy_errors_remain_500_and_persist_diagnostics(self):
        output = "".join(self._handler(_Response([RuntimeError("proxy broken")]))._generate())

        self.assertEqual(self._error_event(output)["code"], "proxy_error")
        entry = self.cache.get_request("req-diagnostics")
        self.assertEqual(entry["status_code"], 500)
        self.assertEqual(entry["response_body"]["error"]["type"], "RuntimeError")

    def test_hidden_reasoning_keeps_client_alive_and_only_caches_accepted_attempt(self):
        base_module.state.sse_keepalive_interval = 0.01
        hidden_seen = threading.Event()
        release = threading.Event()

        def lines():
            yield _event("response.created", marker="discarded")
            yield _reasoning("discarded-thinking")
            hidden_seen.set()
            release.wait(1)
            raise requests.exceptions.ChunkedEncodingError("broken")

        accepted = [_event("response.created", marker="accepted"), _event("response.completed", response={
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })]
        retry = mock.Mock(return_value=_Response(accepted))
        wrapper = RetryingResponsesResponse(_Response(lines()), retry, 1, "req-diagnostics")
        stream = self._handler(wrapper)._generate()
        try:
            self.assertEqual(next(stream), ": keepalive\n\n")
            self.assertTrue(hidden_seen.wait(1))
            release.set()
            output = "".join(stream)
        finally:
            release.set()
            stream.close()

        retry.assert_called_once_with()
        self.assertNotIn("discarded", output)
        entry = self.cache.get_request("req-diagnostics")
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["input_tokens"], 7)
        self.assertEqual(entry["output_tokens"], 3)
        self.assertEqual(entry["raw_events"], [line[6:].decode() for line in accepted])
        self.assertIsNone(entry["response_body"])
