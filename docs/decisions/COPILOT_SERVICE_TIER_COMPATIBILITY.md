# Copilot `service_tier` stripping — review notes and open gaps

- **Status:** Merged, with follow-ups open
- **Review date:** October 25, 2026
- **Scope:** [PR #61](https://github.com/sxwxs/ghc-api/pull/61)
  (`Fix Codex compatibility by omitting unsupported service_tier`), merged as
  `142eca9`
- **Shipped in:** 1.0.31

Record of the review of PR #61. The diff is three lines of behavior; everything
worth remembering is *around* it: why the stripping is necessary, what the merged
change does **not** cover, and which assumption behind it is still unverified.

## 1. Problem

Codex injects `service_tier` into requests on its own — it is not something the
user configures. Upstreams that do not know the field answer with
`HTTP 400 Unknown parameter: 'service_tier'`, which ghc-api passes straight back
to the client (`ghc_api/routes/openai.py` returns the upstream body and status
verbatim on non-2xx).

The failure is reported upstream in
[openai/codex#22876](https://github.com/openai/codex/issues/22876):

```text
Error running remote compact task: {"error":{"code":"invalid_request_error",
"message":"Pipeline execution failed: provider error (HTTP 400):
Unknown parameter:'service_tier'.", ...}}
```

> "No service_tier is set anywhere in config.toml — it is injected by the compact
> request path itself."

Per that issue, Codex decides whether to send the field from the session auth
mode (`auth_mode() == Some(AuthMode::ApiKey)` → omit, otherwise send), so the
bearer-token setup documented in the README for Codex is exactly the shape that
triggers it.

This is a hard failure, not a degradation. If the field rides on every turn,
Codex is unusable against ghc-api; if it only rides on the compaction path, the
session works until auto-compaction fires and then dies with an error that points
at nothing the user did.

## 2. What was merged

`ghc_api/routes/openai.py`: `payload.pop("service_tier", None)` immediately after
the `copy.deepcopy(payload)` that captures `original_request_body`, in both
`chat_completions()` (line 1461) and `responses()` (line 1760). The deep copy
happens first, so request history keeps the client's original body while the
forwarded payload and the cached `request_body` do not carry the field.

`tests/test_openai_service_tier.py` covers four route aliases × six tier values
plus the streaming and chat→Responses delegation paths. Verified during review:
29 passed; reverting only `ghc_api/routes/openai.py` turns that into 25 failed /
4 passed, so the tests do hold the regression. Full suite at merge time: 605
passed, 421 subtests passed.

## 3. Open gaps

### G1. The Anthropic → Responses path still forwards `service_tier`

`ghc_api/anthropic_responses.py:1390-1397` maps the Anthropic tier into the
translated Responses payload:

```python
if "service_tier" in payload:
    value = payload["service_tier"]
    mapping = {"auto": "auto", "standard_only": "default",
               "default": "default", "priority": "priority"}
    if value in mapping:
        responses["service_tier"] = mapping[value]
```

That payload is posted to `{copilot_base_url}/v1/responses` from
`ghc_api/routes/anthropic.py` (lines 1345, 1701, 1719, 1763), which never passes
through the `openai.py` routes, so the merged fix does not apply to it. A
`/v1/messages` client that sends `service_tier` while
`anthropic_responses_compat_enabled` is on (default `True`,
`ghc_api/state.py:55`) reproduces the original 400.

Current real-world impact is **low**: the captured client fixtures
`tests/fixtures/anthropic_responses/claude_cli_2_1_197.json` and
`claude_cli_2_1_207.json` contain no `service_tier`. The value in
`tests/test_anthropic_responses_translation.py` is a synthetic kitchen-sink
payload, not evidence of client behavior. But the Anthropic API does define
`service_tier` (`auto` / `standard_only`), `compat_profiles.py:150,1405` already
validates it as an accepted request field, so a future client version is enough
to surface this.

The trap is the documentation, not the frequency: README now states the field is
omitted, so whoever debugs that 400 will rule this out first and lose time.

### G2. The README claim is broader than the evidence

README `### OpenAI service tiers` says "Copilot does not support the OpenAI
`service_tier` request parameter", unqualified. Two things in the tree disagree:

- `ghc_api/routes/anthropic.py:233` lists `service_tier` in
  `COPILOT_SUPPORTED_FIELDS` — "Fields supported by Copilot's Anthropic API
  endpoint" — so the claim is at least endpoint-specific.
- `tests/fixtures/anthropic_responses/gpt_5_6_responses.json` (captured traffic,
  `source: "2026-07-10.jl"`) contains `"service_tier": "default"` in the upstream
  **response** object, and `anthropic_responses.py:1489` carries it as a known
  response sidecar field. The upstream Responses backend is not ignorant of the
  concept.

The claim should be scoped to "the Copilot Responses and Chat Completions
endpoints reject this request parameter".

### G3. The unconditional strip is unverified, unobservable and unswitchable

PR #61 did not record the upstream 400 body, so we cannot tell whether the
endpoint rejects the **parameter** or only some **values** (for example
accepting `auto`/`default` and rejecting `priority`). Dropping everything is the
safe default, but it is currently:

- **unobservable** — no log line, no counter, unlike the comparable
  `auto_remove_encrypted_content_on_parse_error` path which emits
  `counters.incr("mod.encrypted_content_removal")` and prints a warning;
- **unswitchable** — `state.copilot_api_base_url` (`ghc_api/api_helpers.py:23`)
  lets these same two routes point at a non-Copilot upstream. There, `priority`
  is a paid tier (third-party model catalogs advertise it as
  `{"id": "priority", "name": "Fast", "description": "1.5x speed"}`, see
  [CLIProxyAPI#6460](https://github.com/router-for-me/CLIProxyAPI/issues/6460))
  and `flex` is a discounted one. Those users now silently lose the tier they are
  paying for, with no signal and no opt-out.

Silently discarding a billable client parameter is the part of this change that
deserves a switch, not the stripping itself.

### G4. "The original request remains available in request history" is conditional

`ghc_api/cache.py:58-67` replaces **both** `request_body` and
`original_request_body` with a `_truncated` placeholder when `request_size`
exceeds `cache_max_request_size`. For large agent requests — the ones most likely
to hit auto-compaction, i.e. exactly the Codex path from §1 — the original body
is not retained. The README sentence needs that qualifier.

### G5. Test and code nits

- `tests/test_openai_service_tier.py` asserts on positional arguments —
  `factory.call_args.args[7]`, `forward.call_args.args[8]`. Any signature change
  in `_build_responses_stream_handler` / `stream_chat_completions` /
  `chat_completions_via_responses` breaks these with an unrelated-looking error.
  Bind by name instead (`inspect.signature(...).bind(*args, **kwargs).arguments`).
- The README states that configured `/proxy/<profile>/v1/...` routes keep
  forwarding `service_tier` unchanged, but no test pins that behavior. Documented
  behavior without a test will drift.
- The tests assert equality against an expected dict; an explicit
  `assert "service_tier" not in sent` makes failures readable.
- The `pop` in `chat_completions()` is partly redundant on the
  chat→Responses delegation path: `_chat_completions_to_responses_payload()` is a
  field whitelist and never copies `service_tier` through. Keeping it is correct
  for cache consistency (`request_body` matches what was sent), but the
  redundancy is not obvious and invites a "dead code" cleanup later.

## 4. Follow-ups, in priority order

1. **Scope the README claim** to the Responses and Chat Completions endpoints,
   and add the `cache_max_request_size` qualifier (G2, G4). Documentation-only,
   but it is what misleads the next debugging session.
2. **Record the upstream 400 body** in this document when it is next observed.
   It answers G3 — parameter-level versus value-level rejection — and determines
   whether the strip should stay unconditional.
3. **Make the strip observable**: one counter (`mod.service_tier_removed`) plus a
   log line, following the encrypted-content-removal precedent.
4. **Cover the Anthropic → Responses path** (G1), ideally by extracting a shared
   helper rather than adding a third ad-hoc `pop`:

   ```python
   UNSUPPORTED_COPILOT_FIELDS = ("service_tier",)

   def strip_unsupported_copilot_fields(payload: Dict) -> list[str]:
       """Remove params Copilot rejects as unknown. Returns dropped keys."""
   ```

   The existing `pop_legacy_option()` (`ghc_api/webiq.py:161`) is the house
   pattern for this. When done, `anthropic_responses.py` should mark the field
   `PRESERVATION_UNSUPPORTED` instead of translating it.
5. **Add a configured-proxy regression test** pinning the documented passthrough
   (G5), and de-brittle the positional-argument assertions.

## 5. Conditions for revisiting the stripping itself

Drop or narrow the strip if any of these becomes true:

- the Copilot Responses endpoint starts accepting `service_tier` (verify by
  sending `auto` and `priority` directly and capturing the response);
- the 400 turns out to be value-specific, in which case filter values rather
  than the field;
- `copilot_api_base_url` deployments against tier-aware upstreams become a
  supported configuration, in which case the strip must be gated on targeting
  Copilot rather than applied to the route unconditionally.
