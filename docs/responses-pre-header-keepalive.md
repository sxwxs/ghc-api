# Responses pre-header keepalive — review notes and decisions

Record of the review of [PR #42](https://github.com/sxwxs/ghc-api/pull/42)
(`/v1/responses` recorded as 499 with zero response bytes, then retried by the
client). It keeps the reasoning that is not visible in the diff: what we decided
and why, what we deliberately did not do, and what is still open.

Shipped as four commits: `Fix Responses pre-header stream keepalive`,
`Handle fast Responses connection failures`,
`Preserve Responses errors and close cancelled streams`,
`Restore the keepalive off switch and stop replaying permanent errors`.

## 1. Problem

`requests.post(..., stream=True)` blocks the Flask route until Copilot returns
**response headers**. During that window the client receives no HTTP bytes at
all — not even a status line — so a client whose read timeout is shorter than the
upstream header latency hangs up and retries. Measured against a fake upstream
that stalls 3 s before headers, with the client giving up after 1 s: time to
first downstream byte 3.01 s before, 0.51 s after (the configured grace).

The fix runs the upstream POST on a background thread; if headers have not
arrived within `responses_pre_header_grace`, the route commits to an SSE
response, emits `: keepalive`, and keeps waiting inside the generator.

**The premise is still not proven from production data.** `response_size` counts
only upstream `data:` payloads, so a `499 + 0 bytes` record cannot by itself
distinguish "stalled before headers" from "headers arrived, then 30 s of
silence" — the second case was already covered by `iter_lines_with_keepalive`.
The available discriminator is the `duration` of those 499 records: on the old
code the 499 was only recorded once `requests.post` returned and the first write
failed, so `duration` approximates the header latency. Add explicit
header-latency instrumentation before tuning the grace (§5).

## 2. Decisions

**D1. Split on "did we get an HTTP response", not on "was it fast".**
The first version took the streaming path for every non-OK upstream response, so
a 400 or 429 that was available immediately still became `HTTP 200` plus a
synthetic SSE event. Clients lost `429` (no backoff — they retry harder,
amplifying the very loop this PR targets), lost `401` (no re-auth), and the
dashboard disagreed with what the client saw. The rule that dissolved most of
the design debate: **an upstream HTTP response, whatever its status, goes back
through the normal route loop; only the case where no response exists yet
commits to SSE.** A `ConnectionError` means no response exists, so it correctly
belongs on the streaming path.

**D2. Emit `error`, never a synthetic `response.failed`.**
`RetryingResponsesResponse` treats an early `response.failed` as a transient
failure that is safe to replay, so synthesising it for permanent errors made a
chained ghc-api replay a request that can never succeed: 4 upstream calls for one
client request (measured). Switching to the standard Responses `error` event
takes that to 1 with no change to the retry wrapper. Use the **documented flat
shape** (`type/code/message/param/sequence_number`); the live service sends a
nested `error` object instead, which breaks SDK deserialization
([openai-dotnet#881](https://github.com/openai/openai-dotnet/issues/881)), so it
is not imitated. `OpenAIResponsesStreamHandler.on_event()` had to learn that
`error` means failure too — without it a chained proxy records the failure as
`200/completed` and success-rate metrics silently lie.

**D3. The grace is a real knob, and 50 ms was too small to matter.**
The grace decides what fraction of upstream errors keep their real HTTP status.
Cross-region RTT alone is often 50–200 ms, so a 50 ms grace would make D1 a no-op
in exactly the deployments that need it. The constraint from §1 is
`grace < min(client read timeout)` — the observed clients give up after ~1 s — so
the default is **0.5 s**, config-only, clamped to `[0, 5]` and capped at the use
site by `min(grace, sse_keepalive_interval)`.

The failure modes are asymmetric, which is why the placeholder errs large: too
small fails **silently** (the status-preserving path quietly never runs), too
large fails **visibly and boundedly** (client-side timeouts).

Validation is one expression rather than a table of rejected values:

```python
state.responses_pre_header_grace = min(max(0.0, float(value)), 5.0)
```

`-1` and `nan` clamp to `0.0` (a non-blocking poll), `inf` clamps to `5.0`. Left
unclamped these are not benign: a negative raises `ValueError` and `inf` raises
`OverflowError` inside `queue.Queue.get(timeout=...)` — turning *every* streaming
request into a 500 — while `nan` silently disables the timeout and degrades back
to the old blocking behavior with no error anywhere. **Argument order matters**:
`max(0.0, nan)` is `0.0`, but `max(nan, 0.0)` is `nan`.

**D4. `sse_keepalive_interval: 0` stays the only runtime off switch.**
It is what the README already documents as "inject nothing", and the direct
Anthropic path already gates on it. The new path was initially entered on
`use_streaming` alone, leaving no way to turn the behavior off in production.
The grace deliberately did **not** get a dashboard field: one runtime switch that
disables the whole thing is enough, and a second knob would need its own
validation surface.

**D5. An in-flight response needs a deterministic owner.**
If the client hangs up after the first keepalive but before upstream headers
arrive, the generator returns while the POST is still running, and the response
that arrives later is handed to nobody — measured: 20 cancelled requests produced
20 responses on which `iter_lines` was never called and `close` never ran.
`BackgroundResult.cancel()` decides ownership under one lock: whoever loses
closes the result. It is called from `finally:`, not from the `GeneratorExit`
arm, so every exit path is covered, and the generator hands ownership of a
response to the stream handler explicitly (`response = None`) so cleanup never
closes a live stream.

**D6. Make the new resource visible instead of capping it.**
Upstream work now lives outside the waitress thread pool: on the old code a
blocked POST occupied one of `server_threads` (16), which was itself a failure
mode but did bound in-flight work. A cancelled client now frees the WSGI thread
while the background POST keeps running for up to `upstream_read_timeout`
(default **1800 s**), so in-flight work is bounded only by
`cancellation rate × remaining header latency`. `cancel()` stops the *leak* but
cannot shorten the *occupancy*.

Rather than build a semaphore for a risk nobody has observed, the occupancy is
now counted: `bg.<label>.{started,inflight,cancelled,orphan_closed}` (new counter
keys render automatically in the dashboard's Proxy Activity panel). The decision
rule, agreed in advance:

| observed | action |
|---|---|
| peak `inflight` < `server_threads` (16) | risk is theoretical, do nothing |
| peak > 64, or `cancelled` > 10/min | add a dedicated pre-header timeout plus a concurrency cap |

Note the frequency of this risk equals the 499 rate the PR exists to fix — the two
are the same event — so it is estimable from existing dashboard data before the
counters have collected anything.

## 3. Costs accepted by merging

* Errors arriving after the grace still return `200` plus an `error` event; the
  true status survives only in the cache record. Inherent to committing response
  headers early — not an implementation gap.
* One extra daemon thread per streaming request (~31 µs, ~19 kB each).
* In-flight upstream work is no longer bounded by the WSGI thread pool; bounded
  by observation (D6) until the counters say otherwise.
* Retry / encrypted-content logic is duplicated between the route loop and the
  pending generator. Drift has already started: the generator dropped the
  `Received error response ...` and encrypted-content warning prints.
* The grace default is a placeholder, not a measurement.

## 4. Deliberately not done

* No dashboard field or `generate_config.py` entry for the grace (D4).
* No concurrency cap or dedicated pre-header timeout (D6).
* No shared helper for the duplicated retry logic — a separate refactor, not
  something to land inside a behavior fix.
* `cancel()` was applied to `openai.py` only. The two `anthropic.py` pending
  paths have the identical ownership gap, but it is pre-existing there.

## 5. Open items

1. **Instrument upstream header latency** (counter buckets plus a warning line
   with `request_id` above a threshold, ~8 lines, ~1 µs per request). It answers
   two questions at once: whether the production 499s really are pre-header
   stalls, and what the grace default should be. Compare the header-latency
   distribution *of the requests that 499'd* against the overall one; a low p99
   is not sufficient, since 499s are a tail phenomenon.
2. **Watch the `bg.responses.*` counters** against the D6 thresholds.
3. **Backoff sleeps are silent**: connection-retry backoff inside the generator
   sleeps up to 8 s without emitting a keepalive, which breaks the promise of a
   short configured interval.
4. **Three streaming paths, three error semantics**: `/v1/messages` direct waits
   a full keepalive interval, the translated path commits immediately,
   `/v1/responses` waits the grace. Converge them or document why they differ.
5. **`handle_direct_anthropic_request` replays every non-OK response**: in the
   non-OK branch an error that is neither web-search-unsupported nor
   orphaned-tool-result falls through to the next `for attempt in
   range(max_retries + 1)` iteration instead of breaking, so any 400 costs 4
   upstream calls. One-line fix, unrelated to this PR.
6. **README links `ANTHROPIC_RESPONSES_WARNING_RUNBOOK.md`, which does not exist.**
7. Minor: the synthetic `error` message is the raw upstream body and may be an
   HTML error page (truncate it); `ping_sent` conflates pre-header keepalives
   with in-stream pings, which is exactly the distinction §1 needs.

## 6. Reproducing and testing

`/v1/responses` streaming behavior is decided entirely by
`responses_pre_header_grace` and `sse_keepalive_interval` against the upstream
delay, so **every test must pin both** rather than rely on defaults; a test that
does not will silently start exercising a different path when a default changes.
Prefer a `threading.Event` handshake over sleeps raced against deadlines.

The cancellation test needs three preconditions or the leak is invisible — an
earlier review round got all three wrong and wrongly cleared the issue:

* the upstream must **close** the stream, otherwise neither version ever finishes;
* `sse_keepalive_interval` must be **shorter than the upstream header delay**, so
  a keepalive write actually fails while the client is gone;
* the client must disconnect **after** the first keepalive and **before** headers.

Assert on the `iter_lines` call count as well as `close`: a response that was
never read is unambiguously ownerless, whereas `close` counts alone can be
explained by GC timing.

End-to-end time-to-first-byte check (expect ~3 s before the fix, ~grace after):

```python
# Fake upstream that stalls before sending response headers; full script in the
# PR discussion. The essentials:
c.recv(65536); time.sleep(3.0)                       # stall BEFORE headers
c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" + SSE)
```

## 7. Process notes

* **The retracted measurement.** The ownership problem (D5) was first measured as
  "no difference" and dismissed. Both branches showed identical `close()` counts
  because the fake upstream held the connection open after the SSE body and the
  keepalive interval was left at its 30 s default, so no write ever failed while
  the client was gone. The lesson that generalises: when a measurement says "no
  difference", verify the harness can actually observe the difference before
  believing it.
* **Mutation-test the regression tests.** Each new test was run against the code
  with its fix reverted; a test that still passes is not guarding anything.
* **Defensive prose should become executable.** The review notes originally
  carried a table of what `Queue.get(timeout=...)` does for `-1`, `0`, `nan` and
  `inf`. The table was not the deliverable — the one-line clamp in D3 is, and it
  is shorter than the table it replaced.
* **Measure before building the mitigation.** The concurrency cap (D6) was
  specified, then deferred behind four counters costing ~0.3 µs per request. The
  deferral is only defensible because the observation landed with it.
