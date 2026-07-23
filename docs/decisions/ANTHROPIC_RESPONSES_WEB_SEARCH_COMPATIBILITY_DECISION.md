# Anthropic Responses Web Search Compatibility Decision

- **Status:** Deferred
- **Decision date:** July 15, 2026
- **Scope:** Claude Code requests served through the OpenAI `/responses` compatibility path

## Summary

When Claude Code uses a model exposed through the OpenAI `/responses` API, web searches are executed successfully, but Claude Code may display an incorrect summary such as:

```text
Did 0 searches in 12s
```

The captured requests contained 1, 2, and 1 completed searches respectively. The final answers and upstream usage data were otherwise correct.

A possible workaround is to project each completed Responses `web_search_call` into an Anthropic `server_tool_use` content block. This would expose the search invocation and query to Claude Code and may allow the client to display the correct search count.

We have decided **not to implement this workaround for now**. The proposed projection is incomplete relative to Anthropic's native web-search protocol and introduces replay, streaming, privacy, and forward-compatibility risks. The existing sidecar-only behavior is safer until the client behavior and protocol consequences are validated more thoroughly.

## Observed behavior

The Responses provider returns the following information:

- one or more `web_search_call` output items;
- the search action and query;
- lifecycle events such as `in_progress`, `searching`, and `completed`;
- the final generated text;
- terminal web-search usage;
- limited citation or source metadata.

The compatibility layer currently preserves `web_search_call` items in the encrypted replay/audit sidecar but does not include them in the Anthropic-visible `content` array:

- non-streaming conversion skips visible web-search blocks in `ghc_api/anthropic_responses.py`;
- streaming conversion skips visible web-search blocks in `ghc_api/sse/anthropic_responses.py`;
- terminal usage is still mapped to `usage.server_tool_use.web_search_requests` when supplied by the upstream response.

As a result, the search is performed and billed, but Claude Code sees only the final text. The current evidence suggests that the client derives its search card or summary from visible Anthropic content blocks rather than solely from the usage field. This client behavior has not yet been established as a stable public contract.

## Proposed compatibility workaround

The proposed workaround would introduce a shared mapping for completed Responses web-search calls.

For each eligible `web_search_call`, the compatibility layer would emit an Anthropic block similar to:

```json
{
  "type": "server_tool_use",
  "id": "srvtoolu_<stable-id>",
  "name": "web_search",
  "input": {
    "query": "example query"
  }
}
```

The streaming form would emit one synthetic block lifecycle after `response.output_item.done`:

1. `content_block_start` with an empty input object;
2. `content_block_delta` containing an `input_json_delta` for the query;
3. `content_block_stop`.

The implementation would also need to:

- emit exactly one block per completed search call, not one block per provider lifecycle event;
- preserve the original Responses output order;
- generate a bounded, deterministic `srvtoolu_` ID from the response identity and output index instead of using the provider's opaque item IDs;
- allocate contiguous Anthropic content indexes rather than reusing sparse Responses output indexes;
- leave the final `stop_reason` as `end_turn` because the server-side search has already completed;
- keep the original provider item in the encrypted sidecar;
- keep terminal and streaming projections identical;
- retain `usage.server_tool_use.web_search_requests` and only warn when usage and completed-call counts disagree;
- emit no block when the call is incomplete, failed, has an unsupported action, or lacks a valid query;
- avoid generating a fabricated `web_search_tool_result`.

## Why the workaround is risky

### 1. It creates an incomplete Anthropic server-tool sequence

Native Anthropic web search normally represents both the invocation and its result:

```text
server_tool_use
web_search_tool_result
text
```

The Responses data currently available to this compatibility layer does not contain enough information to faithfully construct an Anthropic `web_search_tool_result`, including all result content and encrypted citation fields. Emitting only `server_tool_use` would therefore be an approximation, not a lossless protocol translation.

Claude Code may currently tolerate the unpaired block for display purposes, but a current or future client could interpret it as an unfinished server-tool operation. This could affect subsequent turns, history validation, caching, or replay.

### 2. The search-count assumption requires real client validation

The proposed fix assumes that Claude Code counts visible `server_tool_use` blocks. This is consistent with the observed `Did 0 searches` behavior, but it is not a documented client contract.

Before accepting the workaround, it must be tested against real Claude Code versions with at least:

- `server_tool_use` only;
- `server_tool_use` plus usage;
- multiple searches;
- a subsequent conversation turn;
- streaming and non-streaming responses.

Without this validation, the implementation could add protocol complexity without correcting the user-visible count.

### 3. Provider IDs are unstable

Captured Copilot Responses Lite events use different opaque item IDs for the same search across lifecycle events. These IDs may also be long or unsuitable for Anthropic tool identifiers.

The translator cannot use those IDs directly. It must correlate Lite events by `output_index` and generate its own stable Anthropic ID. Public Responses profiles may require stricter upstream identity validation, so the behavior must be profile-aware.

### 4. Streaming, terminal responses, replay, and caches can diverge

Changing only the SSE translator would make the streamed content differ from the terminal response generated by the non-streaming converter. Changing only the terminal converter would leave live Claude Code behavior unchanged.

A safe implementation must update all related projections together:

- streamed Anthropic events;
- terminal Anthropic response content;
- cached response bodies;
- visible assistant-message reconstruction;
- encrypted replay items;
- replay matching and restoration.

The current SSE translator also has broader content-index and item-identity limitations that can merge multiple text parts even though terminal conversion keeps them separate. That pre-existing behavior makes a general streaming-versus-terminal equivalence guarantee difficult without additional state-machine work.

### 5. Search queries become more widely visible

Today, the search action and query are retained primarily in the protected sidecar and redacted diagnostics. Adding the query to `server_tool_use.input` intentionally makes it visible to Claude Code, but it may also expand its presence in:

- response caches;
- logs and diagnostics;
- dashboard exports;
- replay-visible message projections;
- test failure output.

Any implementation would require a complete review of redaction, cache retention, warning sanitation, and tenant isolation. Compatibility warnings must remain value-free.

### 6. Incorrect calls could be reported as successful searches

The mapping must not generate a visible search block merely because an item has type `web_search_call`. It must validate that:

- the item is completed;
- the action is a supported search action;
- a non-empty query is present.

Falling back from an unsupported multi-query structure to the first query could misrepresent both semantics and search count. Unsupported or incomplete cases should remain sidecar-only and produce a value-free compatibility warning.

### 7. Usage and visible call counts can disagree

The upstream terminal usage and the number of completed output items are separate observations. Either may be missing or malformed. The compatibility layer must not fabricate searches to force them to agree.

If implemented later, the completed call count should control visible blocks, while upstream usage should remain the authoritative billing/statistics value. A mismatch should be reported, not repaired heuristically.

## Decision

We will retain the current behavior:

- preserve Responses `web_search_call` items in the encrypted sidecar;
- return the final generated text;
- preserve upstream web-search usage when available;
- do not emit synthetic `server_tool_use` or `web_search_tool_result` blocks;
- accept that Claude Code may temporarily display `Did 0 searches` even when searches were performed.

This is a deliberate safety trade-off. An incorrect UI count is preferable to introducing an incomplete server-tool history that may affect replay or future conversation turns.

The behavior should continue to be classified as a compatibility limitation rather than lossless support.

## Conditions for revisiting the decision

We may revisit the mapping when all of the following are available:

1. Real Claude Code testing confirms that one synthetic `server_tool_use` block produces exactly one displayed search and does not require a result block.
2. A subsequent user turn succeeds when the assistant history contains the synthetic, unpaired block.
3. Streaming, terminal response, cache, and replay projections can be made equivalent.
4. Stable profile-aware identity and content-index rules are implemented.
5. Query redaction and retention behavior is reviewed and covered by tests.
6. The approximation is explicitly represented in the conversion report and remains rejected by `lossless_required` mode.
7. The change can be scoped behind a compatibility capability or feature flag so it can be disabled without affecting standard profiles.

## Required tests for a future implementation

### Non-streaming conversion

- One completed call produces one `server_tool_use` block.
- Multiple calls produce distinct blocks in the original output order.
- IDs are legal, stable, bounded, and unique by output index.
- The final stop reason remains `end_turn`.
- Upstream usage remains unchanged.
- Missing, invalid, incomplete, or failed actions remain sidecar-only.
- Warnings and redacted cache fields do not expose query values.

### Streaming conversion

- Rotating Copilot Lite lifecycle IDs still produce exactly one block.
- The block is emitted only after the complete item is available.
- The query delta parses as valid JSON.
- Anthropic content indexes are contiguous and stable.
- Two searches produce exactly two blocks.
- Public profiles retain strict identity validation where required.
- Terminal content matches the content accumulated from SSE events.

### Route and replay integration

- `/v1/messages` completes without error events or 502 responses.
- The completed cache record has status 200.
- Replay and visible-message matching remain valid.
- A subsequent conversation turn succeeds.
- Private queries do not appear in warnings or unauthorized diagnostic fields.

### End-to-end Claude Code validation

The original cases should display:

```text
Did 1 search
Did 2 searches
Did 1 search
```

The final answers must remain unchanged, lifecycle events must not be double-counted, and no URLs or result contents may be fabricated.

## Evidence handling

The original request dump is private diagnostic evidence and must not be committed. Any future regression fixture must be synthetic and sanitized while preserving only the relevant event shapes, lifecycle ordering, rotating IDs, annotations, and terminal usage.
