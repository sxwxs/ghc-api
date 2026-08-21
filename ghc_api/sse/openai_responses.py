"""OpenAI Responses (``/v1/responses``) SSE stream handler.

Pure passthrough. The upstream stream uses both ``event:`` and ``data:`` SSE
lines; we forward the ``event:`` line verbatim (single newline) and the
``data:`` JSON unchanged (double newline). Matches the wire format the
pre-refactor ``stream_responses`` produced.
"""

import json
import threading
from typing import Callable, Dict, Iterator, Optional

import requests

from .base import SSEStreamHandler, UpstreamStreamProtocolError


def format_responses_error_event(
    message: str,
    code: str = "upstream_error",
    param=None,
    sequence_number: int = 0,
) -> str:
    """Build a standard terminal Responses API ``error`` SSE event."""
    event = {
        "type": "error",
        "code": code,
        "message": message,
        "param": param,
        "sequence_number": sequence_number,
    }
    return f"event: error\ndata: {json.dumps(event)}\n\n"


class RetryingResponsesResponse:
    """Replay a Responses request when it fails before producing output.

    Copilot occasionally returns a successful HTTP response whose SSE body is
    just ``response.created`` followed by ``response.failed``, or whose body is
    interrupted before output starts. Buffer only that non-output preamble so a
    fresh attempt can replace it transparently. Once any substantive event is
    seen, lines pass through immediately and retries are disabled so text and
    tool calls cannot be duplicated.
    """

    _PRE_OUTPUT_EVENTS = {
        "response.created",
        "response.in_progress",
        "response.queued",
    }

    def __init__(
        self,
        response: requests.Response,
        response_factory: Callable[[], requests.Response],
        max_retries: int,
        request_id: str,
    ) -> None:
        self._response = response
        self._response_factory = response_factory
        self._max_retries = max(0, max_retries)
        self._request_id = request_id
        self._lock = threading.Lock()
        self._closed = False

        # SSEStreamHandler reads these attributes before iterating. ``text`` is a
        # property on purpose: on a ``stream=True`` response ``Response.text``
        # goes through ``Response.content``, which drains the socket and blocks
        # until upstream is done. Reading it here would turn every streaming
        # request into a buffered one.
        self.status_code = response.status_code
        self.ok = response.ok

    @property
    def text(self) -> str:
        with self._lock:
            response = self._response
        return response.text

    def _current_response(self):
        with self._lock:
            if self._closed:
                return None
            return self._response

    def _replace_response(self, expected, replacement) -> bool:
        with self._lock:
            if self._closed or self._response is not expected:
                accepted = False
            else:
                self._response = replacement
                accepted = True

        if accepted:
            expected.close()
        else:
            replacement.close()
        return accepted

    @staticmethod
    def _event_type(line) -> Optional[str]:
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(line, str) or not line.startswith("data: "):
            return None
        data = line[6:]
        if data == "[DONE]":
            return "response.done"
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # A malformed data payload is still downstream-visible output and
            # therefore commits the stream; represent it as an unknown event.
            return ""
        return event.get("type", "") if isinstance(event, dict) else ""

    def _retry(self, response, retries: int, reason: str) -> bool:
        try:
            retry_response = self._response_factory()
        except requests.exceptions.RequestException:
            return False

        if not retry_response.ok:
            retry_response.close()
            return False
        if not self._replace_response(response, retry_response):
            return False

        print(
            f"[Stream Responses] Retrying request {self._request_id} after "
            f"{reason} ({retries + 1}/{self._max_retries})"
        )
        return True

    def iter_lines(self) -> Iterator[bytes]:
        retries = 0

        while True:
            response = self._current_response()
            if response is None:
                return
            lines = iter(response.iter_lines())
            buffered = []
            output_started = False
            early_failure = False
            transport_error = None

            try:
                for line in lines:
                    if self._current_response() is None:
                        return
                    if output_started:
                        yield line
                        continue

                    buffered.append(line)
                    event_type = self._event_type(line)
                    if event_type == "response.failed":
                        early_failure = True
                        break
                    if event_type is not None and event_type not in self._PRE_OUTPUT_EVENTS:
                        output_started = True
                        yield from buffered
                        buffered.clear()
            except requests.exceptions.RequestException as exc:
                transport_error = exc

            # A transport failure is replay-safe only while the created / queued
            # preamble is still buffered. Once any substantive event was yielded,
            # retrying could duplicate text or execute a tool call twice.
            if transport_error is not None:
                if (
                    not output_started
                    and retries < self._max_retries
                    and self._retry(response, retries, "an early transport failure")
                ):
                    retries += 1
                    continue
                raise transport_error

            if early_failure and retries < self._max_retries:
                if self._retry(response, retries, "an early stream failure"):
                    retries += 1
                    continue

            if not output_started and not early_failure and retries < self._max_retries:
                if self._retry(response, retries, "an early incomplete stream"):
                    retries += 1
                    continue

            yield from buffered
            if early_failure:
                # Preserve any sentinel or diagnostic lines following the
                # terminal failure on the final attempt.
                yield from lines
            return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            response = self._response
        response.close()


class OpenAIResponsesStreamHandler(SSEStreamHandler):
    endpoint = "/v1/responses"
    log_prefix = "[Stream Responses]"
    # /v1/responses sends an ``event: TYPE`` line *before* each ``data:`` line.
    # We pass the event header through verbatim and emit only the data line
    # ourselves (the original handler's convention).
    emit_event_header = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._terminal_event_seen = False
        self._next_sequence_number = 0

    def on_event(self, event_type: str, event: Dict) -> None:
        sequence_number = event.get("sequence_number")
        if isinstance(sequence_number, int):
            self._next_sequence_number = max(
                self._next_sequence_number, sequence_number + 1
            )

        if event_type in (
            "response.completed",
            "response.incomplete",
            "response.failed",
            "error",
        ):
            self._terminal_event_seen = True

        if event_type in ("response.completed", "response.incomplete"):
            resp = event.get("response", {}) or {}
            usage = resp.get("usage", {}) or {}
            self.input_tokens = usage.get("input_tokens", 0)
            self.output_tokens = usage.get("output_tokens", 0)
            details = usage.get("input_tokens_details", {}) or {}
            self.cache_creation_input_tokens = details.get("cached_tokens", 0)
        elif event_type == "response.failed":
            # The HTTP status is already committed as 200, but request history
            # should still identify the terminal SSE failure.
            self.error_occurred = True
            self.status_code = 502
        elif event_type == "error":
            # Same for the standard Responses streaming ``error`` event, which
            # the proxy itself emits when an upstream error arrives after the
            # response headers were already committed. Without this a chained
            # ghc-api would record the failure as 200/completed.
            self.error_occurred = True
            self.status_code = 502

    def has_terminal_event(self) -> bool:
        return self._terminal_event_seen

    def validate_stream_end(self) -> None:
        if not self._terminal_event_seen:
            raise UpstreamStreamProtocolError(
                "Responses SSE stream ended without a terminal event"
            )

    def _format_transport_error(self, exc: Exception) -> str:
        category = (self.stream_error or {}).get("category")
        if category == "upstream_connection_error":
            code = "upstream_connection_error"
            message = (
                "Upstream Responses connection was interrupted: "
                f"{type(exc).__name__}"
            )
        else:
            code = "upstream_stream_error"
            message = (
                "Upstream Responses stream ended unexpectedly: "
                f"{type(exc).__name__}"
            )
        return format_responses_error_event(
            message,
            code=code,
            sequence_number=self._next_sequence_number,
        )

    def _format_generic_error(self, exc: Exception) -> str:
        return format_responses_error_event(
            "Internal Responses stream processing failed",
            code="proxy_error",
            sequence_number=self._next_sequence_number,
        )
