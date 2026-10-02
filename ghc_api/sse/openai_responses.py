"""OpenAI Responses (``/v1/responses``) SSE stream handler.

The upstream stream uses both ``event:`` and ``data:`` SSE lines. Preserve
their wire format, but keep output item IDs stable when Copilot rotates its
opaque IDs between events. Unchanged events remain byte-for-byte passthrough;
the base handler retains the original upstream payloads in the request cache.
"""

import json
import threading
from typing import Callable, Dict, Iterator, Optional

import requests

from .base import SSEStreamHandler


class RetryingResponsesResponse:
    """Replay a Responses request when it fails before producing output.

    Buffer the non-output preamble and encrypted-only reasoning with empty
    content/summary so failures and transport errors can be replaced before
    visible output. Text, visible reasoning, tool calls, unknown events, or the
    1 MiB buffer limit commit the stream and permanently disable retries.
    The outer handler continues client keepalives while this iterator buffers.

    An optional error_response_factory can recover a specific upstream error
    independently of the transient retry budget. It must bound its own retries.
    Keep a trailing error diagnostic with the failed preamble so invalid input
    is never blindly retried, and the original error survives failed recovery.
    """

    _PRE_OUTPUT_EVENTS = {
        "response.created",
        "response.in_progress",
        "response.queued",
    }
    _INVALID_REQUEST_CODES = ("invalid_request_body", "invalid_request_error")
    _MAX_BUFFERED_BYTES = 1024 * 1024
    _TRANSPORT_ERRORS = (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.ReadTimeout,
    )

    def __init__(
        self,
        response: requests.Response,
        response_factory: Callable[[], requests.Response],
        max_retries: int,
        request_id: str,
        error_response_factory: Optional[Callable[[Dict], Optional[requests.Response]]] = None,
        max_buffer_bytes: int = 1024 * 1024,
    ) -> None:
        self._response = response
        self._response_factory = response_factory
        self._max_retries = max(0, max_retries)
        self._request_id = request_id
        self._error_response_factory = error_response_factory
        self._max_buffer_bytes = max(0, max_buffer_bytes)
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
    def _event(line) -> Optional[Dict]:
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                return {}
        if not isinstance(line, str) or not line.startswith("data:"):
            return None
        data = line[5:].lstrip(" ")
        if data == "[DONE]":
            return {"type": "response.done"}
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # A malformed data payload is still downstream-visible output and
            # therefore commits the stream; represent it as an unknown event.
            return {}
        return event if isinstance(event, dict) else {}

    @staticmethod
    def _error_payload(event: Dict) -> Dict:
        """Return the error object of a standalone ``error`` event.

        The API reference documents a flat event, but the live service wraps
        the error in a nested ``error`` object instead. Both shapes occur in
        practice, which is why ``anthropic_error_from_responses`` has unwrapped
        them the same way since the Anthropic bridge shipped. Matching only the
        flat shape here would silently skip recovery on the nested one.
        """
        nested = event.get("error")
        return nested if isinstance(nested, dict) else event

    def _recover_from_http_error(self, response) -> Optional[requests.Response]:
        """Offer error recovery a replay that upstream rejected in HTTP form.

        The same rejection arrives either as a pre-output SSE failure or as an
        HTTP 4xx, and the two can interleave across attempts. Dropping the HTTP
        body here would leave recovery unattempted and send the client the
        earlier, diagnostic-free ``response.failed`` instead.
        """
        if self._error_response_factory is None:
            return None
        try:
            error = response.json().get("error")
        except (AttributeError, TypeError, ValueError):
            return None
        if not isinstance(error, dict):
            return None
        try:
            return self._error_response_factory(error)
        except Exception as exc:
            print(
                f"[Stream Responses] Error recovery failed for request "
                f"{self._request_id}: {type(exc).__name__}: {exc}"
            )
            return None

    @staticmethod
    def _is_hidden_reasoning(event: Dict) -> bool:
        if event.get("type") not in (
            "response.output_item.added", "response.output_item.done",
        ):
            return False
        item = event.get("item")
        return (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and isinstance(item.get("encrypted_content"), str)
            and bool(item["encrypted_content"])
            and item.get("content") in (None, [])
            and item.get("summary") in (None, [])
        )

    def iter_lines(self) -> Iterator[bytes]:
        retries = 0

        while True:
            response = self._current_response()
            if response is None:
                return
            lines = iter(response.iter_lines())
            if retries >= self._max_retries and self._error_response_factory is None:
                terminal_failure_seen = False
                try:
                    for line in lines:
                        if self._current_response() is None:
                            return
                        event = self._event(line)
                        if event is not None and event.get("type") == "response.failed":
                            terminal_failure_seen = True
                        yield line
                except requests.exceptions.RequestException:
                    if not terminal_failure_seen:
                        raise
                return
            buffered = []
            buffered_bytes = 0
            output_started = False
            early_failure = False
            terminal_error = False
            error = None
            transport_error = None

            while True:
                try:
                    line = next(lines)
                except StopIteration:
                    break
                except requests.exceptions.RequestException as exc:
                    if self._current_response() is None:
                        return
                    if output_started:
                        raise
                    if not early_failure:
                        if not isinstance(exc, self._TRANSPORT_ERRORS) or retries >= self._max_retries:
                            raise
                        transport_error = exc
                        early_failure = True
                    # A broken connection while reading the optional diagnostic
                    # must not discard the terminal failure already received.
                    # Mid-stream breaks usually surface as ChunkedEncodingError
                    # rather than ConnectionError, so catch the common base.
                    lines = iter(())
                    break
                if self._current_response() is None:
                    return
                if output_started:
                    yield line
                    continue

                buffered.append(line)
                buffered_bytes += len(line if isinstance(line, bytes) else line.encode("utf-8")) + 1
                event = self._event(line)
                event_type = event.get("type", "") if event is not None else None
                # Bound the preamble, including comments/diagnostics after a
                # failure. Once anything is forwarded it cannot be replaced.
                if buffered_bytes > self._max_buffer_bytes:
                    output_started = True
                    early_failure = False
                    yield from buffered
                    buffered.clear()
                    continue
                if event_type == "error":
                    error = self._error_payload(event)
                    terminal_error = True
                    break
                if early_failure:
                    # Copilot can put error=null on response.failed and send
                    # the actual rejection in a following standalone error.
                    # Read that diagnostic before deciding to replay history.
                    if event_type is None:
                        continue
                    if event_type == "response.done":
                        break
                    # Unexpected content after failure must remain visible.
                    early_failure = False
                    output_started = True
                    yield from buffered
                    buffered.clear()
                    continue
                if event_type == "response.failed":
                    failed = event.get("response") or {}
                    if isinstance(failed, dict) and not failed.get("output"):
                        early_failure = True
                        error = failed.get("error")
                        if isinstance(error, dict) and error:
                            break
                        continue
                if (event_type is not None and event_type not in self._PRE_OUTPUT_EVENTS
                        and not self._is_hidden_reasoning(event)):
                    output_started = True
                    yield from buffered
                    buffered.clear()

            retry_response = None
            if self._current_response() is None:
                return
            if isinstance(error, dict) and self._error_response_factory is not None:
                try:
                    retry_response = self._error_response_factory(error)
                except Exception as exc:
                    # Recovery is best effort. A transport failure or a failed
                    # token refresh (RuntimeError) must not discard the buffered
                    # upstream diagnostic, which is the more useful answer --
                    # and token trouble is exactly what triggers this path.
                    print(
                        f"[Stream Responses] Error recovery failed for request "
                        f"{self._request_id}: {type(exc).__name__}: {exc}"
                    )
                    retry_response = None

            invalid_request = (
                isinstance(error, dict)
                and (
                    error.get("code") in self._INVALID_REQUEST_CODES
                    or error.get("type") in self._INVALID_REQUEST_CODES
                )
            )
            # A diagnostic means upstream stated a reason, so replaying the same
            # input mostly burns quota on a request that cannot succeed; only an
            # unexplained failure (``error: null`` and no diagnostic) is replayed.
            generic_retry = (
                retry_response is None and early_failure and not terminal_error
                and not invalid_request and retries < self._max_retries
            )
            if generic_retry:
                try:
                    retry_response = self._response_factory()
                except Exception as exc:
                    # The original response.failed is more useful to the
                    # downstream client than replacing it with an empty 504.
                    print(
                        f"[Stream Responses] Early-failure retry failed for request "
                        f"{self._request_id}: {type(exc).__name__}: {exc}"
                    )
                    retry_response = None

                if retry_response is not None and not retry_response.ok:
                    recovered = self._recover_from_http_error(retry_response)
                    if recovered is not None:
                        retry_response.close()
                        retry_response = recovered
                        # Recovery has its own one-shot budget; this attempt
                        # must not be charged to the connection retries.
                        generic_retry = False

            if retry_response is not None and retry_response.ok:
                if not self._replace_response(response, retry_response):
                    return
                if generic_retry:
                    retries += 1
                    print(
                        f"[Stream Responses] Retrying request {self._request_id} "
                        f"after an early stream failure ({retries}/{self._max_retries})"
                    )
                continue

            # The retry could not establish a valid SSE stream. Preserve the
            # original failure and diagnostic for the downstream client.
            if retry_response is not None:
                retry_response.close()

            yield from buffered
            if transport_error is not None:
                raise transport_error
            if early_failure or terminal_error:
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
        self._item_ids: Dict[int, str] = {}
        self._next_sequence_number = 0
        self._error_response_body: Optional[Dict] = None

    def _normalize_id(self, output_index, value: Dict, field: str) -> Dict:
        """Keep the first upstream ID for an output slot, without inventing IDs.

        The index correlates Copilot's independently wrapped IDs. Retain a real
        upstream ID rather than generating one: clients may replay output items
        in later requests. Do not touch call_id or encrypted reasoning content.
        """
        if type(output_index) is not int or output_index < 0:
            return value
        item_id = value.get(field)
        if not isinstance(item_id, str) or not item_id:
            return value
        canonical_id = self._item_ids.setdefault(output_index, item_id)
        if item_id == canonical_id:
            return value
        return {**value, field: canonical_id}

    def forward_event(self, event_type: str, event: Dict, raw_data: str) -> Iterator[tuple]:
        normalized = event
        if event_type.startswith("response."):
            output_index = event.get("output_index")
            if event_type in ("response.output_item.added", "response.output_item.done"):
                item = event.get("item")
                if isinstance(item, dict):
                    normalized_item = self._normalize_id(output_index, item, "id")
                    if normalized_item is not item:
                        normalized = {**normalized, "item": normalized_item}
            normalized = self._normalize_id(output_index, normalized, "item_id")

        if event_type in ("response.completed", "response.incomplete", "response.failed"):
            response = event.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                normalized_output = [
                    self._normalize_id(index, item, "id") if isinstance(item, dict) else item
                    for index, item in enumerate(output)
                ]
                if any(new is not old for new, old in zip(normalized_output, output)):
                    normalized = {
                        **normalized,
                        "response": {**response, "output": normalized_output},
                    }

        # Yield immediately, including deltas. Copy-on-write preserves both the
        # parsed upstream event and the exact wire bytes for healthy streams.
        yield (event_type, raw_data if normalized is event else json.dumps(normalized, ensure_ascii=False))

    def on_event(self, event_type: str, event: Dict) -> None:
        sequence_number = event.get("sequence_number")
        if type(sequence_number) is int:
            self._next_sequence_number = max(self._next_sequence_number, sequence_number + 1)
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

    def _format_error(self, exc: Exception, code: str) -> str:
        message = str(exc)
        self._error_response_body = {"error": {
            "type": type(exc).__name__, "code": code, "message": message,
        }}
        event = {
            "type": "error",
            "code": code,
            "message": message,
            "param": None,
            "sequence_number": self._next_sequence_number,
        }
        return f"event: error\ndata: {json.dumps(event)}\n\n"

    def _format_transport_error(self, exc: Exception) -> str:
        return self._format_error(exc, "upstream_connection_error")

    def _format_generic_error(self, exc: Exception) -> str:
        code = (
            "upstream_stream_error"
            if isinstance(exc, requests.exceptions.ChunkedEncodingError)
            else "proxy_error"
        )
        return self._format_error(exc, code)

    def extra_cache_fields(self) -> Dict:
        # Keep upstream raw_events untouched; store proxy-generated diagnostics
        # separately so request JSONL logs retain the actual exception.
        if self._error_response_body is not None:
            return {"response_body": self._error_response_body}
        return {}
