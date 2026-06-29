"""Tests for ghc_api.sse.keepalive.iter_lines_with_keepalive.

The helper reads upstream lines on a background thread and yields a KEEPALIVE
sentinel whenever the stream is idle longer than the configured interval, so the
proxy can emit a client-side ping instead of blocking until the next byte.
"""

import threading
import time
import unittest

import requests

from ghc_api.sse.keepalive import (
    BackgroundResult,
    KEEPALIVE,
    iter_lines_with_keepalive,
    wait_result_with_keepalive,
)


class _SlowResponse:
    """iter_lines() that sleeps before yielding its (single) line."""

    def __init__(self, delay, line=b"data: hello"):
        self._delay = delay
        self._line = line

    def iter_lines(self):
        time.sleep(self._delay)
        yield self._line

    def close(self):
        pass


class _ImmediateResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


class _RaisingResponse:
    def __init__(self, exc):
        self._exc = exc

    def iter_lines(self):
        raise self._exc
        yield  # pragma: no cover - makes this a generator

    def close(self):
        pass


class KeepaliveTest(unittest.TestCase):
    def _drain_with_return(self, gen):
        out = []
        while True:
            try:
                out.append(next(gen))
            except StopIteration as stop:
                return out, stop.value

    def test_idle_stream_yields_keepalive_before_real_line(self):
        # Line arrives after 0.3s but the keepalive interval is 0.1s, so at least
        # one KEEPALIVE must be emitted before the real line.
        resp = _SlowResponse(delay=0.3, line=b"data: hello")
        out = []
        for item in iter_lines_with_keepalive(resp, interval=0.1):
            out.append(item)
        self.assertIn(KEEPALIVE, out)
        # The real line is delivered, and it comes after the keepalive(s).
        self.assertEqual(out[-1], b"data: hello")
        self.assertIs(out[0], KEEPALIVE)

    def test_interval_zero_is_pure_passthrough(self):
        lines = [b"data: a", b"data: b", b"data: [DONE]"]
        resp = _ImmediateResponse(lines)
        out = list(iter_lines_with_keepalive(resp, interval=0))
        self.assertEqual(out, lines)
        self.assertNotIn(KEEPALIVE, out)

    def test_negative_interval_is_pure_passthrough(self):
        lines = [b"data: a"]
        out = list(iter_lines_with_keepalive(_ImmediateResponse(lines), interval=-5))
        self.assertEqual(out, lines)

    def test_fast_stream_yields_no_keepalive(self):
        lines = [b"data: a", b"data: b"]
        out = list(iter_lines_with_keepalive(_ImmediateResponse(lines), interval=5))
        self.assertEqual(out, lines)

    def test_upstream_exception_is_reraised_in_consumer(self):
        resp = _RaisingResponse(requests.exceptions.ReadTimeout("slow"))
        with self.assertRaises(requests.exceptions.ReadTimeout):
            list(iter_lines_with_keepalive(resp, interval=5))

    def test_reader_thread_is_daemon_and_terminates(self):
        before = threading.active_count()
        list(iter_lines_with_keepalive(_ImmediateResponse([b"x"]), interval=5))
        # Give the daemon thread a moment to wind down after the sentinel.
        time.sleep(0.05)
        self.assertLessEqual(threading.active_count(), before + 1)

    def test_background_result_yields_keepalive_until_result_is_ready(self):
        def slow_result():
            time.sleep(0.25)
            return "ready"

        pending = BackgroundResult(slow_result)
        out, result = self._drain_with_return(
            wait_result_with_keepalive(pending, interval=0.1)
        )

        self.assertIn(KEEPALIVE, out)
        self.assertEqual(result, "ready")

    def test_background_result_returns_without_keepalive_when_ready_fast(self):
        pending = BackgroundResult(lambda: "ready")
        out, result = self._drain_with_return(
            wait_result_with_keepalive(pending, interval=5)
        )

        self.assertEqual(out, [])
        self.assertEqual(result, "ready")

    def test_background_result_reraises_exception_in_consumer(self):
        pending = BackgroundResult(lambda: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("boom")
        ))

        with self.assertRaises(requests.exceptions.ConnectionError):
            list(wait_result_with_keepalive(pending, interval=5))


if __name__ == "__main__":
    unittest.main()
