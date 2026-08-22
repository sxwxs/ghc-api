"""Base SSE stream handler.

``SSEStreamHandler`` owns the SSE pipeline shared by every streaming route:

* Iterate upstream ``response.iter_lines()`` and parse ``event:``/``data:`` lines.
* Append every upstream ``data:`` payload verbatim to ``self.raw_events`` so the
  cache stores exactly what the server sent us -- no reconstruction, no field
  cherry-picking.
* Forward each line to the client untouched. Subclasses that need to translate
  override :meth:`forward_event` (called with the parsed event) or
  :meth:`forward_raw_line` (called with bytes that are not SSE ``data:`` lines).
* Drive the cache lifecycle (start_request -> SENDING -> RECEIVING -> complete_request).
* Handle GeneratorExit, upstream timeout/connection errors, and generic exceptions
  in the same shape every existing handler does today.

The base class deliberately does NOT import ``tool_call_recovery``; that lives
only in the recovery subclass so flipping the toggle off is truly a no-op.
"""

import json
import time
from typing import Any, Dict, Generator, Iterator, List, Optional

import requests
from flask import Response, stream_with_context

from ..cache import cache
from ..counters import counters
from ..state import state
from .keepalive import KEEPALIVE, iter_lines_with_keepalive


class UpstreamStreamProtocolError(Exception):
    """The upstream SSE body ended without a valid protocol terminator."""


class SSEStreamHandler:
    """Base class for an SSE stream from Copilot back to the API client.

    Subclasses must set:
      - ``endpoint``: the proxy-side endpoint string used for cache records
        (e.g. ``/v1/messages``).
      - ``log_prefix``: tag used in print() error logs (e.g. ``[Stream Direct Anthropic]``).

    Subclasses may override:
      - ``on_event(event_type, event)`` -- extract usage / accumulate state.
        Default: no-op.
      - ``forward_event(event_type, event, raw_data)`` -- yield ``(event_type, raw_data)``
        pairs to send to the client. Default: yields the upstream pair unchanged.
      - ``forward_raw_line(line)`` -- yield SSE-formatted strings for upstream lines
        that are neither ``event:`` nor ``data:`` headers. Default: passes through
        unchanged when upstream returned an error (matches existing behavior).
      - ``forward_malformed_data(data)`` -- handle a ``data:`` payload that is
        not JSON. Default: pass it through verbatim.
      - ``parse_event_data(data)`` -- decode one JSON data payload. Protocol
        adapters may override this with a stricter parser.
      - ``finalize_stream()`` -- emit any pending events after the upstream
        iterator ends. Default: no events.
      - ``extra_cache_fields()`` -- additional keys to pass to ``cache.complete_request``.
        Default: empty dict.

    Subclasses declare their protocol's terminal events via ``TERMINAL_EVENTS``;
    declaring any enables automatic clean-EOF truncation detection (see below).
    """

    endpoint: str = ""
    log_prefix: str = "[Stream]"
    # When True, ``forward_event`` output is emitted as ``event: TYPE\ndata: JSON\n\n``
    # (matches the existing direct-Anthropic handler). When False, only the
    # ``data:`` line is emitted and upstream ``event:`` lines are passed through
    # verbatim (matches the existing ``/v1/responses`` handler). The two
    # conventions differ; set per subclass.
    emit_event_header: bool = True
    # Whether to forward the OpenAI-style ``data: [DONE]`` sentinel to the
    # client. OpenAI streams use it; Anthropic ``/v1/messages`` does not (it
    # signals end-of-stream via ``message_stop`` events) and clients parsing
    # each line as Anthropic JSON would choke on a bare ``[DONE]``.
    emit_done_sentinel: bool = True
    capture_raw_sse_lines: bool = False
    # Event types that terminate the stream per the protocol spec (e.g.
    # Anthropic ``message_stop``, Responses ``response.completed``). The base
    # class tracks them automatically in :meth:`note_event`; a transport error
    # arriving after one is ignored, and a clean EOF without one raises
    # :class:`UpstreamStreamProtocolError`. Empty (default) means the protocol
    # has no required terminator: EOF is always accepted and truncation is only
    # detected via transport errors. Protocols whose termination cannot be
    # expressed as an event-type set should leave this empty-ish declaration to
    # the base class and override :meth:`has_terminal_event` instead.
    TERMINAL_EVENTS: frozenset = frozenset()

    def __init__(
        self,
        response: requests.Response,
        request_id: str,
        request_size: int,
        start_time: float,
        original_model: str,
        translated_model: str,
        request_body_for_cache: Dict,
        original_request_body: Optional[Dict] = None,
        request_headers: Optional[Dict] = None,
        client_ip: Optional[str] = None,
        user_id: str = "anonymous",
        max_raw_capture_bytes: int = 0,
    ) -> None:
        self.response = response
        self.request_id = request_id
        self.request_size = request_size
        self.start_time = start_time
        self.original_model = original_model
        self.translated_model = translated_model
        self.request_body_for_cache = request_body_for_cache
        self.original_request_body = original_request_body
        self.request_headers = request_headers
        self.client_ip = client_ip
        self.user_id = user_id

        # Every ``data:`` payload seen on the wire, in order. This is what gets
        # persisted to the cache for later inspection -- no transformation.
        self.raw_events: List[str] = []
        # Logical SSE lines, including event headers, comments, unknown fields
        # and blank frame separators. Protocol adapters may opt into this view
        # without changing the bounded dashboard cache contract.
        self.raw_sse_lines: List[str] = []
        self.max_raw_capture_bytes = max(0, int(max_raw_capture_bytes or 0))
        # Combined retained-content budget across raw_events and raw_sse_lines.
        # A data payload is charged for both copies when both views are enabled.
        self.raw_capture_bytes = 0
        self.raw_capture_truncated = False
        self.response_wire_bytes = 0

        # Bookkeeping for cache totals -- populated by ``on_event``.
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_creation_input_tokens: int = 0
        self.cache_read_input_tokens: int = 0

        # Stream-level state.
        self.status_code: int = response.status_code
        self.error_occurred: bool = False
        self.stream_error: Optional[Dict[str, str]] = None
        self._terminal_event_seen: bool = False

        # Whether the cache entry has been started. Seeded eagerly from
        # :meth:`stream` so the request shows up in /api/requests before the
        # first byte streams (Flask iterates the generator lazily).
        self._cache_seeded: bool = False
        self._cache_completed: bool = False

    # ------------------------------------------------------------------ hooks

    def on_event(self, event_type: str, event: Dict) -> None:
        """Subclass hook for usage extraction. No effect on the wire."""
        return None

    def forward_event(
        self, event_type: str, event: Dict, raw_data: str
    ) -> Iterator[tuple]:
        """Yield ``(out_event_type, out_data_json_string)`` pairs to send to the
        client. Default: pass the upstream event through unchanged.
        """
        yield (event_type, raw_data)

    def forward_raw_line(self, line: str) -> Iterator[str]:
        """Yield SSE-formatted strings for upstream lines that are not ``event:``/``data:``.
        Default: pass through when upstream returned an error status, otherwise drop.
        Matches the existing handlers' shape.
        """
        if self.status_code > 399:
            yield f"{line}\n\n"

    def forward_malformed_data(self, data: str) -> Iterator[str]:
        """Handle malformed JSON from an SSE ``data:`` line."""
        yield f"data: {data}\n\n"

    def parse_event_data(self, data: str) -> Any:
        return json.loads(data)

    def note_event(self, event_type: str) -> None:
        """Track protocol terminal events. Called by ``_generate`` for every
        parsed event, before :meth:`on_event`. Subclasses normally do not
        override this; they declare ``TERMINAL_EVENTS`` instead.
        """
        if event_type in self.TERMINAL_EVENTS:
            self._terminal_event_seen = True

    def has_terminal_event(self) -> bool:
        """Whether a protocol-defined terminal event has already been received.

        Override when termination is tracked outside the event-type set (e.g. a
        translator state machine). The default reads the flag maintained by
        :meth:`note_event` from declared ``TERMINAL_EVENTS``.
        """
        return self._terminal_event_seen

    def validate_stream_end(self) -> None:
        """Validate a clean EOF before finalizing protocol-specific state.

        Protocols that declare ``TERMINAL_EVENTS`` must see one before EOF;
        anything else is a truncated upstream body and raises
        :class:`UpstreamStreamProtocolError`. Protocols without a required
        terminator accept EOF (legacy behavior for handlers that have not
        opted into truncation detection).
        """
        if self.TERMINAL_EVENTS and not self.has_terminal_event():
            raise UpstreamStreamProtocolError(
                f"stream ended without a terminal event "
                f"(expected one of {sorted(self.TERMINAL_EVENTS)})"
            )

    def finalize_stream(self) -> Iterator[tuple]:
        """Emit any pending events after the upstream iterator ends. Default: none."""
        return iter(())

    def keepalive_event(self) -> str:
        """SSE payload emitted to the client when the upstream stream has been
        idle past ``state.sse_keepalive_interval``. Default: an SSE comment line,
        which every SSE client ignores. Subclasses that need a protocol-specific
        keepalive (e.g. Anthropic's ``ping`` event) override this.
        """
        return ": keepalive\n\n"

    def extra_cache_fields(self) -> Dict[str, Any]:
        """Extra keys for ``cache.complete_request``. Default: none."""
        return {}

    def raw_events_for_cache(self) -> List[str]:
        """Dashboard projection of captured data payloads."""

        return list(self.raw_events)

    def _capture_raw_value(self, target: List[str], value: str) -> None:
        size = len(value.encode("utf-8", errors="strict"))
        if (
            self.max_raw_capture_bytes > 0
            and self.raw_capture_bytes + size > self.max_raw_capture_bytes
        ):
            self.raw_capture_truncated = True
            return
        target.append(value)
        self.raw_capture_bytes += size

    # ----------------------------------------------------------- cache helpers

    def _seed_cache(self) -> None:
        if self._cache_seeded:
            return
        self._cache_seeded = True
        cache.start_request(
            self.request_id,
            {
                "request_headers": self.request_headers,
                "client_ip": self.client_ip,
                "original_request_body": self.original_request_body,
                "request_body": self.request_body_for_cache,
                "model": self.original_model,
                "translated_model": (
                    self.translated_model
                    if self.translated_model != self.original_model
                    else None
                ),
                "endpoint": self.endpoint,
                "request_size": self.request_size,
                "user_id": self.user_id,
            },
        )

    def _complete_cache(self) -> None:
        if self._cache_completed:
            return
        self._cache_completed = True
        duration = round(time.time() - self.start_time, 2)
        response_size = self.response_wire_bytes
        record = {
            "request_body": self.request_body_for_cache,
            "raw_events": self.raw_events_for_cache(),
            "model": self.original_model,
            "translated_model": (
                self.translated_model
                if self.translated_model != self.original_model
                else None
            ),
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "request_size": self.request_size,
            "response_size": response_size,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "duration": duration,
            "user_id": self.user_id,
        }
        if self.stream_error is not None:
            record["stream_error"] = dict(self.stream_error)
        record.update(self.extra_cache_fields())
        cache.complete_request(self.request_id, record)

    # ----------------------------------------------------------- main pipeline

    def stream(self) -> Response:
        """Return a Flask streaming response. The cache entry is seeded
        synchronously here so the request is visible in /api/requests even
        before Flask starts iterating the generator -- matching the original
        handlers' behavior.
        """
        self._seed_cache()
        return Response(
            stream_with_context(self._generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _generate(self) -> Generator[str, None, None]:
        # Idempotent -- normally seeded by stream(); kept here so tests that
        # drive _generate() directly still produce a complete cache entry.
        self._seed_cache()
        first_chunk_received = False
        try:
            cache.update_request_state(self.request_id, cache.STATE_SENDING)
            sse_event_type = ""

            for line in iter_lines_with_keepalive(self.response, state.sse_keepalive_interval):
                if line is KEEPALIVE:
                    counters.incr("ping_sent")
                    yield self.keepalive_event()
                    continue

                if not line:
                    if self.capture_raw_sse_lines:
                        self._capture_raw_value(self.raw_sse_lines, "")
                    continue

                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError:
                    if self.capture_raw_sse_lines:
                        self._capture_raw_value(
                            self.raw_sse_lines, "hex:" + bytes(line).hex()
                        )
                    raise
                if self.capture_raw_sse_lines:
                    self._capture_raw_value(self.raw_sse_lines, line)

                if line.startswith("event:"):
                    sse_event_type = line[6:]
                    if sse_event_type.startswith(" "):
                        sse_event_type = sse_event_type[1:]
                    if sse_event_type == "ping":
                        counters.incr("ping_received")
                    if not self.emit_event_header:
                        # Pass the event header through verbatim and let the
                        # next ``data:`` line emit only the data part. Matches
                        # the existing ``/v1/responses`` handler.
                        yield f"{line}\n"
                    # When ``emit_event_header`` is True the header is bundled
                    # with each ``data:`` line below; do not yield it here.
                    continue

                if line.startswith("data:"):
                    data = line[5:]
                    if data.startswith(" "):
                        data = data[1:]
                    if data == "[DONE]":
                        if self.emit_done_sentinel:
                            yield "data: [DONE]\n\n"
                        break

                    # Record the raw payload before anything else so even
                    # malformed JSON is preserved in the cache.
                    self.response_wire_bytes += len(data.encode("utf-8"))
                    self._capture_raw_value(self.raw_events, data)

                    try:
                        event = self.parse_event_data(data)
                    except (ValueError, TypeError, UnicodeError):
                        # The raw payload is already cached. Let protocol-aware
                        # subclasses decide whether passthrough is safe.
                        yield from self.forward_malformed_data(data)
                        sse_event_type = ""
                        continue

                    if not first_chunk_received:
                        first_chunk_received = True
                        cache.update_request_state(self.request_id, cache.STATE_RECEIVING)

                    event_type = sse_event_type or event.get("type", "")
                    sse_event_type = ""

                    self.note_event(event_type)
                    self.on_event(event_type, event)

                    for out_type, out_data in self.forward_event(event_type, event, data):
                        if self.emit_event_header:
                            yield f"event: {out_type}\ndata: {out_data}\n\n"
                        else:
                            yield f"data: {out_data}\n\n"

                    continue

                # Anything else (non-event, non-data line): defer to subclass.
                yield from self.forward_raw_line(line)

            self.validate_stream_end()
            for out_type, out_data in self.finalize_stream():
                if self.emit_event_header:
                    yield f"event: {out_type}\ndata: {out_data}\n\n"
                else:
                    yield f"data: {out_data}\n\n"

        except requests.exceptions.RequestException as e:
            # If the protocol terminal event arrived intact, a missing final
            # HTTP chunk carries no additional application data. Finish the
            # downstream stream normally instead of turning success into error.
            if self.has_terminal_event():
                print(
                    f"{self.log_prefix} Ignoring transport error after terminal "
                    f"event for request {self.request_id}: {type(e).__name__}: {e}"
                )
                return
            # requests raises ChunkedEncodingError (a RequestException, not a
            # ConnectionError) when a chunked body loses its terminating chunk.
            # ReadTimeout/ConnectionError mean the connection itself broke
            # (504); any other transport failure is an upstream gateway error
            # mid-body (502), not an internal 500.
            if isinstance(
                e,
                (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError),
            ):
                status_code, category = 504, "upstream_connection_error"
            else:
                status_code, category = 502, "upstream_stream_error"
            self._set_stream_error(e, status_code, category)
            print(
                f"{self.log_prefix} Upstream transport error for request "
                f"{self.request_id}: {type(e).__name__}: {e}"
            )
            formatted = self._format_transport_error(e)
            if formatted:
                yield formatted
        except UpstreamStreamProtocolError as e:
            self._set_stream_error(e, 502, "upstream_stream_error")
            print(
                f"{self.log_prefix} Incomplete upstream stream for request "
                f"{self.request_id}: {e}"
            )
            formatted = self._format_transport_error(e)
            if formatted:
                yield formatted
        except GeneratorExit:
            self.error_occurred = True
            self.status_code = 499
            print(f"{self.log_prefix} Client disconnected for request {self.request_id}")
            cache.update_request_state(self.request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            self.error_occurred = True
            self.status_code = 500
            print(f"{self.log_prefix} Error for request {self.request_id}: {type(e).__name__}: {e}")
            try:
                yield self._format_generic_error(e)
            except GeneratorExit:
                self.status_code = 499
                cache.update_request_state(self.request_id, cache.STATE_ERROR, status_code=499)
                return
        finally:
            # Preserve partial raw events and diagnostics even when the client
            # disconnects or the generator exits early.  Replay callbacks only
            # run on a validated terminal event, so partial reasoning is never
            # promoted to reusable state here.
            self._complete_cache()

    def _set_stream_error(self, exc: Exception, status_code: int, category: str) -> None:
        self.error_occurred = True
        self.status_code = status_code
        self.stream_error = {
            "category": category,
            "type": type(exc).__name__,
            "message": str(exc),
        }

    def _transport_failure(self, exc: Exception) -> tuple:
        """Facts about the recorded stream error, for protocol-specific
        rendering: ``(timed_out, detail)``. ``timed_out`` is True when the
        connection itself broke (read timeout / connection loss); ``detail``
        carries the exception class and message for client-side diagnostics.
        Must be called after :meth:`_set_stream_error`.
        """
        timed_out = (
            (self.stream_error or {}).get("category") == "upstream_connection_error"
        )
        return timed_out, f"{type(exc).__name__}: {exc}"

    def _format_generic_error(self, e: Exception) -> str:
        """SSE payload for the generic-Exception arm. Subclasses can override
        to emit an API-specific shape (e.g. Anthropic ``error`` event)."""
        return f"data: {json.dumps({'error': str(e)})}\n\n"

    def _format_transport_error(self, e: Exception) -> Optional[str]:
        """Optional protocol-specific SSE error for an upstream stream failure."""

        return None
