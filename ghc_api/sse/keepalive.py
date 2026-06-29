"""Keepalive wrapper for upstream SSE streams.

``response.iter_lines()`` has no per-iteration timeout, so when the Copilot
upstream goes silent (e.g. while the model "thinks" before the first token) the
proxy writes nothing to the client and the client's own read timeout fires.

``iter_lines_with_keepalive`` reads the upstream lines on a background daemon
thread feeding a queue; the consumer blocks on ``queue.get(timeout=interval)``
and yields the ``KEEPALIVE`` sentinel whenever the stream has been idle for
longer than ``interval`` seconds. Callers translate ``KEEPALIVE`` into the
endpoint-appropriate keepalive payload (Anthropic ``ping`` event or an SSE
comment). Upstream exceptions are re-raised in the consumer so existing
``ReadTimeout`` / ``ConnectionError`` handling is unchanged.
"""

import queue
import threading

# Yielded to the consumer when the stream has been idle for ``interval`` seconds.
KEEPALIVE = object()
# Internal marker: the upstream iterator finished normally.
_SENTINEL = object()


class BackgroundResult:
    """Run a blocking operation on a daemon thread and expose its result.

    This is used before an upstream SSE response object exists: the foreground
    generator can wait with a timeout and emit a client keepalive only when the
    upstream call has made no progress.
    """

    def __init__(self, fn):
        self._q: "queue.Queue" = queue.Queue(maxsize=1)

        def _runner():
            try:
                self._q.put((False, fn()))
            except Exception as exc:  # propagate to the consumer thread
                self._q.put((True, exc))

        threading.Thread(target=_runner, daemon=True).start()

    def get(self, timeout=None):
        is_exc, item = self._q.get(timeout=timeout)
        if is_exc:
            raise item
        return item


def wait_result_with_keepalive(pending_result, interval):
    """Return a background result; yield ``KEEPALIVE`` while it is idle.

    This covers the phase before ``requests.post(..., stream=True)`` has
    returned response headers. Once the response object exists,
    ``iter_lines_with_keepalive`` covers idle gaps in the SSE body.
    """
    if not interval or interval <= 0:
        result = pending_result.get()
        if False:
            yield KEEPALIVE
        return result

    while True:
        try:
            return pending_result.get(timeout=interval)
        except queue.Empty:
            yield KEEPALIVE


def iter_lines_with_keepalive(response, interval):
    """Yield raw lines from ``response.iter_lines()``; yield ``KEEPALIVE`` when
    the stream has been idle for more than ``interval`` seconds.

    ``interval`` <= 0 disables keepalive entirely and is a pure passthrough to
    ``response.iter_lines()`` (byte-identical to the old behavior).
    """
    if not interval or interval <= 0:
        yield from response.iter_lines()
        return

    q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    def _reader():
        try:
            for line in response.iter_lines():
                if stop.is_set():
                    break
                q.put((False, line))
        except Exception as exc:  # propagate to the consumer thread
            q.put((True, exc))
        finally:
            q.put((False, _SENTINEL))

    threading.Thread(target=_reader, daemon=True).start()

    try:
        while True:
            try:
                is_exc, item = q.get(timeout=interval)
            except queue.Empty:
                yield KEEPALIVE
                continue
            if is_exc:
                raise item
            if item is _SENTINEL:
                return
            yield item
    finally:
        # On early consumer exit (e.g. client disconnect -> GeneratorExit),
        # signal the reader to stop and close the upstream response so the
        # thread's blocking ``iter_lines()`` unblocks instead of lingering.
        stop.set()
        response.close()
