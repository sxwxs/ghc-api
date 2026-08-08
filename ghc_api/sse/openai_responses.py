"""OpenAI Responses (``/v1/responses``) SSE stream handler.

Pure passthrough. The upstream stream uses both ``event:`` and ``data:`` SSE
lines; we forward the ``event:`` line verbatim (single newline) and the
``data:`` JSON unchanged (double newline). Matches the wire format the
pre-refactor ``stream_responses`` produced.
"""

import json
from typing import Callable, Dict, Iterator, Optional

import requests

from .base import SSEStreamHandler


class RetryingResponsesResponse:
    """Replay a Responses request when it fails before producing output.

    Copilot occasionally returns a successful HTTP response whose SSE body is
    just ``response.created`` followed by ``response.failed``. Buffer only that
    non-output preamble so a fresh attempt can replace it transparently. Once
    any substantive event is seen, lines pass through immediately and retries
    are disabled so text and tool calls cannot be duplicated.
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

        # SSEStreamHandler reads these attributes before iterating. ``text`` is a
        # property on purpose: on a ``stream=True`` response ``Response.text``
        # goes through ``Response.content``, which drains the socket and blocks
        # until upstream is done. Reading it here would turn every streaming
        # request into a buffered one.
        self.status_code = response.status_code
        self.ok = response.ok

    @property
    def text(self) -> str:
        return self._response.text

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

    def iter_lines(self) -> Iterator[bytes]:
        retries = 0

        while True:
            response = self._response
            lines = iter(response.iter_lines())
            buffered = []
            output_started = False
            early_failure = False

            for line in lines:
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

            if early_failure and retries < self._max_retries:
                try:
                    retry_response = self._response_factory()
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                    # The original response.failed is more useful to the
                    # downstream client than replacing it with an empty 504.
                    retry_response = None

                if retry_response is not None and retry_response.ok:
                    response.close()
                    self._response = retry_response
                    retries += 1
                    print(
                        f"[Stream Responses] Retrying request {self._request_id} "
                        f"after an early stream failure ({retries}/{self._max_retries})"
                    )
                    continue

                # The retry could not establish a valid SSE stream. Preserve
                # the original response.failed event for the downstream client.
                if retry_response is not None:
                    retry_response.close()

            yield from buffered
            if early_failure:
                # Preserve any sentinel or diagnostic lines following the
                # terminal failure on the final attempt.
                yield from lines
            return

    def close(self) -> None:
        self._response.close()


class OpenAIResponsesStreamHandler(SSEStreamHandler):
    endpoint = "/v1/responses"
    log_prefix = "[Stream Responses]"
    # /v1/responses sends an ``event: TYPE`` line *before* each ``data:`` line.
    # We pass the event header through verbatim and emit only the data line
    # ourselves (the original handler's convention).
    emit_event_header = False

    def on_event(self, event_type: str, event: Dict) -> None:
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
