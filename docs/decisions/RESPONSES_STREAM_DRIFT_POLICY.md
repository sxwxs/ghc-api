# Responses stream drift policy

## Context

`/v1/messages` requests for Responses-only models are translated by a strict
state machine (`ghc_api/sse/anthropic_responses.py`) and a strict wire auditor
(`ghc_api/compat_profiles.py`). Both were written to fail closed: any deviation
from the observed contract produced an Anthropic `error` event and a logged 502.

That policy destroyed live traffic. A `copilot_responses_lite` backend
re-encrypts every identifier per frame, so `response.created`,
`response.in_progress`, and `response.completed` each carry a different
`response.id`. The translator required them to match and rejected every single
stream *after* the answer had already been delivered, which Claude Code shows as
`API Error: Server error mid-response`.

Two properties made this expensive:

- the check ran at the terminal frame, so a complete answer was thrown away;
- the fixture sanitizer replaced every id with one constant placeholder, so no
  test could observe the rotation the backend actually performs.

## Decision

Drift is classified by what it can corrupt, not by how unusual it looks.

**Fatal (error event, 502) — the delivered content itself would be wrong:**

- content-bearing contradictions: `text_done_mismatch`, `arguments_mismatch`,
  `item_type_mutation`, `content_part_reordering`, invalid function arguments;
- lifecycle contradictions that make the stream unreadable: events after a
  closed item, interleaved open blocks, a terminal frame that claims `failed`
  or a non-terminal status, an `error` object on a success terminal;
- identity mismatches on profiles whose ids are documented stable
  (`stable_ids=True`), because there a foreign id can mean foreign content;
- anything at all under `mode=lossless_required`.

**Recoverable (recorded, exchange continues) — cosmetic or already-delivered:**

- rotating identifiers on profiles with `stable_ids=False`;
- unknown lifecycle events and unknown output item types (the upstream surface
  keeps growing; a new event must not break every request);
- terminal model change, unknown terminal status, terminal output that omits an
  item that was already streamed, items that never became projectable;
- unknown/missing `incomplete_details.reason`, projected as `max_tokens`.

Every recoverable case is still recorded as a compatibility warning and in the
conversion report, so the dashboard shows exactly what was skipped.

**Non-streaming exception.** Nothing has been sent yet on that transport, so a
body that projects to *no* content while recording an unsupported output item
still fails closed: an empty assistant message would be less honest than a 502.

## Upstream error classes

Upstream failures keep their HTTP class (`responses_error_status`). Collapsing a
429 into 502 prevents clients from backing off correctly.

## Test obligations

- Every recoverable rule has a paired test: recorded and skipped in
  `compatibility` mode, fatal in `lossless_required` mode.
- Fixture sanitizing must preserve identifier equality/inequality and structural
  indices; see `scripts/generate_anthropic_responses_fixtures.py`.
- At least one captured stream per wire profile is replayed **in order**
  (`tests/fixtures/anthropic_responses/coherent_stream_lite.json`). Shape-only
  coverage cannot express cross-frame rules and is what let this ship.
