# Async JSONL logging (design, not yet implemented)

Status: **agreed, not implemented.** This file is the plan of record so the
next change does not have to re-derive it.

## The problem

Two code paths append a JSON Lines record to disk from inside a request
thread, both with the same shape:

| Caller | File | Guarded by |
|---|---|---|
| `cache._append_request_to_daily_file()` | `<config_dir>/requests/YYYY-MM-DD.jl` | `save_request_to_file` |
| `webiq_log.record_search_to_file()` | `<config_dir>/webiq/YYYY-MM-DD.jl` | `log_webiq_requests` (on by default) |

Both do `open(path, "a")` / `write(one_line)` / `close()` **outside any lock**,
once per request, on the request thread. Two problems:

1. **Interleaving.** Concurrent appends from several threads can interleave and
   produce a line that is not valid JSON, which corrupts the whole file for
   `/api/requests/import` and for the request-statistics indexer. A single small
   `write()` is *usually* atomic, but a Web IQ search with `contentFormat: html`
   and `maxLength: 500000` is megabytes — far past any atomicity you can rely on.
2. **Latency on the request path.** A slow or full disk stalls a paid upstream
   call that has already succeeded.

## The design

New module `ghc_api/jsonl_writer.py`, used by both callers.

```python
class JsonlWriter:
    def __init__(self, queue_maxsize: int = 10000): ...
    def write(self, path: str, line: str) -> bool:   # producer; never raises
    def flush(self, timeout: float = 5.0) -> bool:   # drain; for tests + atexit
    def stats(self) -> dict:                         # queued/written/dropped/errors
```

- One module-level instance. The consumer thread is **started lazily** on the
  first `write()` (double-checked under a `threading.Lock`), so a process that
  only runs a CLI subcommand does not grow a thread for nothing.
- `queue.Queue` is thread-safe on its own (internal lock + condition variables);
  no extra locking is needed around it.
- Consumer loop: block on `q.get()`, then **batch-drain** up to N (≈256) more
  with `get_nowait()`, group by path, and do one `open`/`write*`/`close` per
  path per batch. The batching is the point: under a burst this turns one
  `open()` per record into one per batch.
- Shutdown: `atexit` sends a sentinel and calls `flush(timeout=5)`. The thread
  is a daemon, so Ctrl-C still exits promptly; atexit just makes a best effort
  not to lose the tail.
- Error handling: on write failure, print — but **rate-limit per path** (e.g. at
  most one message per 60s per path), otherwise a full disk turns the console
  into a scrolling wall and hides everything else.

### Backpressure — decided

Use `put_nowait()`; if the queue is full, **drop the record and increment
`dropped`**. Logging must never slow down a paid upstream call.

Because a silently dropped log is more dangerous than a slow one, `dropped`
must be *visible*: expose it via `stats()` and surface it on the dashboard
(next to the existing `webiq.search` counters). A non-zero `dropped` means the
disk cannot keep up and the operator needs to know.

## Migration cost (accept this before starting)

Writes become asynchronous, so any test that asserts "the file has the record
immediately after the request returned" becomes flaky. `flush()` is a public
API for exactly this reason. Known call sites to update:

- `tests/test_webiq_log.py` — `FileLoggingTest`, `RouteLoggingTest` (they read
  `<config_dir>/webiq/*.jl` right after the call)
- any request-file test that reads `requests/*.jl` after a proxied request

## Rollout order

1. Add `jsonl_writer.py` with its own unit tests (ordering, batching, drop
   accounting, `flush()` semantics, per-path error rate limiting).
2. Route `webiq_log.record_search_to_file()` through it — small blast radius,
   and it is the caller with the multi-megabyte lines.
3. Route `cache._append_request_to_daily_file()` through it — larger blast
   radius (`save_request_to_file`, export/import, the request-statistics
   indexer all read those files).

Do **not** do 3 in the same commit as 2.
