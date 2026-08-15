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

from ..counters import counters

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

    def __init__(self, fn, label: str = "other"):
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._cancelled = False
        self._done = threading.Event()
        self._label = label

        # In-flight upstream work started here lives outside the WSGI thread
        # pool, so a cancelled client no longer bounds it. These counters are
        # what makes that occupancy visible (new keys render automatically in
        # the dashboard's Proxy Activity panel).
        counters.incr(f"bg.{label}.started")
        counters.incr(f"bg.{label}.inflight")

        def _runner():
            try:
                try:
                    result = (False, fn())
                except Exception as exc:  # propagate to the consumer thread
                    result = (True, exc)

                discard = False
                with self._lock:
                    if self._cancelled:
                        discard = True
                    else:
                        self._q.put_nowait(result)

                if discard:
                    counters.incr(f"bg.{label}.orphan_closed")
                    self._close_result(result)
            finally:
                counters.incr(f"bg.{label}.inflight", -1)
                self._done.set()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    @staticmethod
    def _close_result(result):
        is_exc, item = result
        if is_exc:
            return
        close = getattr(item, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Cancellation is best-effort cleanup and must not mask the
                # client disconnect or other error that initiated it.
                pass

    def get(self, timeout=None):
        is_exc, item = self._q.get(timeout=timeout)
        if is_exc:
            raise item
        return item

    def cancel(self):
        """Discard and close a queued or eventual result.

        Python cannot interrupt a blocking ``requests.post`` safely, but a
        cancelled consumer must not leave the response it eventually returns
        open. The lock makes cancellation atomic with the producer's enqueue.
        """
        queued_result = None
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            counters.incr(f"bg.{self._label}.cancelled")
            try:
                queued_result = self._q.get_nowait()
            except queue.Empty:
                pass

        if queued_result is not None:
            counters.incr(f"bg.{self._label}.orphan_closed")
            self._close_result(queued_result)

    @property
    def done(self):
        """True once the background call finished (delivered, failed or discarded)."""
        return self._done.is_set()


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
        try:
            yield from response.iter_lines()
        finally:
            response.close()
        return

    # Bound producer lead so a fast or adversarial upstream cannot queue an
    # unbounded response in memory while the downstream client is slow.
    q: "queue.Queue" = queue.Queue(maxsize=128)
    stop = threading.Event()

    def _put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _reader():
        try:
            for line in response.iter_lines():
                if stop.is_set():
                    break
                if not _put((False, line)):
                    break
        except Exception as exc:  # propagate to the consumer thread
            _put((True, exc))
        finally:
            _put((False, _SENTINEL))

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
