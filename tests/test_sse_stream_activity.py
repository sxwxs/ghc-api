"""Downstream liveness and content-free diagnostics for shared SSE handlers."""

import http.client
import io
import json
import queue
import threading
from collections import deque
from unittest import mock

import pytest
import requests
import urllib3

from ghc_api.cache import RequestCache
from ghc_api.sse import base as base_module
from ghc_api.sse import keepalive as keepalive_module
from ghc_api.sse.anthropic_direct import AnthropicDirectStreamHandler
from ghc_api.sse.openai_responses import OpenAIResponsesStreamHandler, RetryingResponsesResponse
from ghc_api.state import state


class Clock:
    now = 0.0
    epoch = 1_800_000_000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.epoch + self.now


class FakeResponse:
    status_code = 200
    ok = True

    def __init__(self, lines=()):
        self.lines = lines
        self.closed = False

    def iter_lines(self):
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        self.closed = True


class ScheduledQueue:
    """Model Queue.get's deadline exactly, without sleeps or reader races.

    Entries represent lines already read by the producer at a given time. The
    real reader/observer path is tested separately below.
    """

    def __init__(self, clock, schedule):
        self.clock = clock
        self.items = deque(schedule)

    def get(self, timeout):
        at, value = self.items[0]
        if at > self.clock.now + timeout:
            self.clock.now += timeout
            raise queue.Empty
        self.clock.now = max(self.clock.now, at)
        self.items.popleft()
        return False, value


def data(event_type, **fields):
    return ("data: " + json.dumps({"type": event_type, **fields})).encode()


def handler(response=None, cls=OpenAIResponsesStreamHandler):
    return cls(
        response=response if response is not None else FakeResponse(),
        request_id="activity-test",
        request_size=0,
        start_time=Clock.epoch,
        original_model="test-model",
        translated_model="test-model",
        request_body_for_cache={},
    )


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(base_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(base_module.time, "time", clock.time)
    return clock


@pytest.fixture
def cache(monkeypatch):
    cache = RequestCache()
    monkeypatch.setattr(base_module, "cache", cache)
    monkeypatch.setattr(state, "save_request_to_file", False)
    monkeypatch.setattr(state, "sse_keepalive_interval", 30)
    return cache


def schedule_lines(monkeypatch, clock, schedule):
    scheduled = ScheduledQueue(clock, [
        *schedule,
        (schedule[-1][0], keepalive_module._SENTINEL),
    ])
    monkeypatch.setattr(keepalive_module.queue, "Queue", lambda **kw: scheduled)
    monkeypatch.setattr(keepalive_module.threading, "Thread", mock.Mock())
    return scheduled


@pytest.mark.parametrize("ignored_line", [b": upstream heartbeat", b"", b"retry: 1000"])
@pytest.mark.parametrize("cls", [OpenAIResponsesStreamHandler, AnthropicDirectStreamHandler])
def test_ignored_upstream_lines_cannot_starve_downstream_heartbeat(
    monkeypatch, clock, cache, ignored_line, cls,
):
    terminal = "message_stop" if cls is AnthropicDirectStreamHandler else "response.completed"
    schedule_lines(monkeypatch, clock, [
        (20, ignored_line), (40, ignored_line), (60, ignored_line), (80, ignored_line),
        (100, data(terminal, response={"usage": {}})),
    ])
    stream = handler(cls=cls)
    chunks = [(clock.now, chunk) for chunk in stream._generate()]

    assert [(at, chunk) for at, chunk in chunks if chunk == stream.keepalive_event()] == [
        (30, stream.keepalive_event()),
        (60, stream.keepalive_event()),
        (90, stream.keepalive_event()),
    ]
    assert stream.response.closed
    assert cache.get_request("activity-test")["stream_diagnostics"]["keepalives_sent"] == 3


def test_forwarded_data_resets_the_downstream_deadline(monkeypatch, clock, cache):
    lines = [(at, data("keepalive", sequence_number=i))
             for i, at in enumerate([20, 40, 60, 80])]
    lines.append((100, data("response.completed", response={"usage": {}})))
    schedule_lines(monkeypatch, clock, lines)

    chunks = list(handler()._generate())

    assert all(not chunk.startswith(":") for chunk in chunks)
    assert chunks == [line.decode() + "\n\n" for _, line in lines]


def test_keepalive_does_not_terminate_an_unfinished_event_header(monkeypatch, clock, cache):
    schedule_lines(monkeypatch, clock, [
        (0, b"event: response.output_text.delta"),
        (20, b": upstream heartbeat"),
        (40, b": upstream heartbeat"),
        (50, b'data: {"delta":"hello"}'),
    ])

    chunks = list(handler()._generate())

    # A blank line after the heartbeat would reset the SDK's pending event
    # name before the data arrived. Keep the comment inside the same frame.
    assert "".join(chunks) == (
        'event: response.output_text.delta\n'
        ': keepalive\n'
        'data: {"delta":"hello"}\n\n'
    )


def test_periodic_ticks_do_not_spin_if_the_consumer_ignores_them(monkeypatch, clock):
    schedule_lines(monkeypatch, clock, [(100, b"done")])
    response = FakeResponse()

    items = [(clock.now, item) for item in keepalive_module.iter_lines_with_keepalive(
        response, 30, last_activity=lambda: 0,
    )]

    assert [at for at, item in items if item is keepalive_module.KEEPALIVE] == [30, 60, 90]
    assert items[-1] == (100, b"done")
    assert response.closed


@pytest.mark.parametrize("interval", [0, 30])
def test_upstream_observer_sees_comments_and_blank_lines(interval):
    lines = [b": heartbeat", b"", data("keepalive")]
    response = FakeResponse(lines)
    observed = []

    assert list(keepalive_module.iter_lines_with_keepalive(
        response, interval, on_line=observed.append,
    )) == lines
    assert observed == lines
    assert response.closed


def test_observer_runs_before_the_consumer_drains_the_queue():
    observed = []
    all_read = threading.Event()
    lines = [b"one", b"two", b"three"]

    def observe(line):
        observed.append(line)
        if len(observed) == len(lines):
            all_read.set()

    response = FakeResponse(lines)
    stream = keepalive_module.iter_lines_with_keepalive(response, 30, on_line=observe)
    try:
        assert next(stream) == b"one"
        # Only one line has been consumed; all observations must already be
        # available, rather than being timestamped at downstream drain time.
        assert all_read.wait(2)
        assert observed == lines
    finally:
        stream.close()
    assert response.closed


def test_off_switch_preserves_wire_output_but_still_records_activity(monkeypatch, clock, cache):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)

    def lines():
        for at in [20, 40, 60, 80]:
            clock.now = at
            yield b": upstream heartbeat"
        clock.now = 100
        yield data("response.completed", response={"usage": {}})

    stream = handler(FakeResponse(lines()))
    chunks = list(stream._generate())

    assert chunks == [data("response.completed", response={"usage": {}}).decode() + "\n\n"]
    diagnostics = cache.get_request("activity-test")["stream_diagnostics"]
    assert diagnostics["upstream_comment_lines"] == 4
    assert diagnostics["keepalives_sent"] == 0
    assert diagnostics["last_upstream_line_at"] == Clock.epoch + 100
    assert diagnostics["last_downstream_yield_at"] == Clock.epoch + 100
    assert diagnostics["terminal_event_type"] == "response.completed"
    assert diagnostics["exception_type"] is None


@pytest.mark.parametrize("exception,status,code", [
    (requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
     502, "upstream_stream_interrupted"),
    (requests.exceptions.ReadTimeout("private URL / secret should not be persisted"),
     504, "upstream_connection_error"),
    (requests.exceptions.ConnectionError("private URL / secret should not be persisted"),
     504, "upstream_connection_error"),
])
def test_transport_error_has_diagnostics_and_never_retries_partial_output(
    monkeypatch, clock, cache, exception, status, code,
):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    raw = data("response.output_item.added", sequence_number=7, output_index=0,
               item={"type": "reasoning", "id": "r-1", "summary": []})

    def lines():
        clock.now = 1
        yield raw
        clock.now = 3
        yield b": private upstream comment"
        clock.now = 10
        raise exception

    upstream = FakeResponse(lines())
    retry = mock.Mock()
    stream = handler(RetryingResponsesResponse(upstream, retry, 3, "activity-test"))
    chunks = list(stream._generate())

    retry.assert_not_called()
    assert upstream.closed
    assert chunks[-1].startswith("event: error\ndata: ")
    error = json.loads(chunks[-1].split("data: ", 1)[1])
    assert error["type"] == "error"
    assert error["code"] == code
    assert error["sequence_number"] == 8
    assert error["param"] is None
    assert "private" not in chunks[-1]
    assert not any("response.failed" in chunk for chunk in chunks)
    entry = cache.get_request("activity-test")
    assert entry["status_code"] == status
    assert entry["raw_events"] == [raw.decode()[6:]]
    diagnostics = entry["stream_diagnostics"]
    assert diagnostics["exception_type"] == type(exception).__name__
    assert diagnostics["last_upstream_line_at"] == Clock.epoch + 3
    assert diagnostics["last_downstream_yield_at"] == Clock.epoch + 10
    assert diagnostics["finished_at"] == Clock.epoch + 10
    assert diagnostics["upstream_comment_lines"] == 1
    assert diagnostics["last_event_type"] == "response.output_item.added"
    assert diagnostics["terminal_event_type"] is None
    assert "private" not in json.dumps(diagnostics)


def test_disconnect_still_finalizes_once_and_records_last_yield(monkeypatch, clock, cache):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    stream = handler(FakeResponse([data("response.created")]))
    gen = stream._generate()
    next(gen)
    clock.now = 12
    gen.close()
    stream._complete_cache()

    assert stream.response.closed
    assert cache.request_count == 1
    entry = cache.get_request("activity-test")
    assert entry["status_code"] == 499
    assert entry["stream_diagnostics"]["exception_type"] == "GeneratorExit"
    assert entry["stream_diagnostics"]["last_downstream_yield_at"] == Clock.epoch
    assert entry["stream_diagnostics"]["finished_at"] == Clock.epoch + 12


@pytest.mark.parametrize("evict", [False, True])
def test_diagnostics_survive_body_truncation_daily_dump_and_import(
    monkeypatch, clock, cache, tmp_path, evict,
):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(state, "save_request_to_file", True)
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    cache.max_request_size = 1

    def lines():
        yield data("response.created")
        if evict:
            # A long-running request can be evicted before it completes.
            with cache.lock:
                cache.cache.clear()
        raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")

    list(handler(FakeResponse(lines()))._generate())

    paths = list((tmp_path / "requests").glob("*.jl"))
    assert len(paths) == 1
    dumped = json.loads(paths[0].read_text(encoding="utf-8"))
    assert json.loads(dumped["raw_events"][0])["_truncated"]
    assert dumped["stream_diagnostics"]["exception_type"] == "ChunkedEncodingError"
    imported = RequestCache()
    imported.import_request(dumped)
    assert imported.get_request("activity-test")["stream_diagnostics"] == dumped["stream_diagnostics"]


def test_generic_exception_is_distinct_from_upstream_disconnect(monkeypatch, clock, cache):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    list(handler(FakeResponse([RuntimeError("test failure")]))._generate())

    entry = cache.get_request("activity-test")
    assert entry["status_code"] == 500
    assert entry["stream_diagnostics"]["exception_type"] == "RuntimeError"
    assert entry["stream_diagnostics"]["last_upstream_line_at"] is None


@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete", "response.failed", "error"])
def test_terminal_event_is_recorded_separately_from_transport_exception(monkeypatch, clock, cache, terminal):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    list(handler(FakeResponse([
        data(terminal, response={"usage": {}}), b"data: [DONE]",
    ]))._generate())

    diagnostics = cache.get_request("activity-test")["stream_diagnostics"]
    assert diagnostics["terminal_event_type"] == terminal
    assert diagnostics["exception_type"] is None


def test_real_chunked_http_eof_is_reported_as_upstream_interruption(monkeypatch, clock, cache):
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    payload = data("response.created") + b"\n\n"
    wire = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        + format(len(payload), "x").encode() + b"\r\n" + payload + b"\r\n"
        # Deliberately omit HTTP's terminating zero-length chunk.
    )

    class Socket:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(wire)

    original = http.client.HTTPResponse(Socket(), method="POST")
    original.begin()
    response = requests.Response()
    response.status_code = 200
    response.raw = urllib3.response.HTTPResponse(
        body=original, headers=dict(original.headers), status=200,
        preload_content=False, original_response=original,
    )

    chunks = list(handler(response)._generate())

    assert '"code": "upstream_stream_interrupted"' in chunks[-1]
    entry = cache.get_request("activity-test")
    assert entry["status_code"] == 502
    assert entry["stream_diagnostics"]["exception_type"] == "ChunkedEncodingError"
    assert entry["raw_events"] == [payload.decode()[6:].strip()]
