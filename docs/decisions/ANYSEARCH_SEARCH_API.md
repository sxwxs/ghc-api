# AnySearch REST APIs and Chat tools

## Decision and scope

AnySearch is independent of Web IQ. Proxy all three documented REST endpoints
and expose matching client-executed tools when AnySearch is selected in Chat:

| Method/path | Chat tool | Meaning |
| --- | --- | --- |
| POST /v1/search | anysearch_search | Ranked search results |
| POST /v1/extract | anysearch_extract | Clean content from a public HTTP(S) URL |
| GET /v1/sub-domains | anysearch_sub_domains | Search capability definitions |

The original integration only implemented Search. This extension fills the two
missing REST contracts and lets the Chat tool loop compose all three. Existing
Web IQ routes, tool names, and native model-side web_search behavior are unchanged.
The LLM proxy never secretly executes searches or injects results into prompts.
`ghc_api/web_search.py`, the older Anthropic rejection fallback, is unrelated.

Authoritative contracts (checked together, rather than just the Search page):

- https://www.anysearch.com/docs/api-endpoints
- https://www.anysearch.com/docs/api-endpoints/v1-search
- https://www.anysearch.com/docs/api-endpoints/v1-extract
- https://www.anysearch.com/docs/api-endpoints/v1-sub-domains
- https://www.anysearch.com/docs/auth

## Native contracts

Search forwards bytes, including unknown/additive parameters. The proxy does
not validate search fields, clamp counts, insert defaults, or rewrite tag,
zone, language, params, or format. Markdown changes content, not the outer JSON.

Extract's upstream contract is strict: exactly one UTF-8 JSON object containing
one nonempty url string, at most 16 KiB, no unknown or duplicate fields, a public
absolute HTTP(S) URL, application/json with no charset or UTF-8, and no content
encoding or identity. The proxy preserves bytes, Content-Type (including its
absence), and Content-Encoding so **upstream**, not an approximate local parser,
validates them. Extraction is performed by AnySearch: ghc-api must never issue
an HTTP request directly to the caller's url. Ordinary errors including 422 and
error_code remain intact. The project-wide JSON nesting safety guard still applies.

**Sub-domains is not DNS enumeration.** Despite some marketing wording, the API
returns capability tags and parameter definitions for search categories such as
code and finance. The request is `?domain=code&domain=finance`, not `domains`, a
comma-separated string, or a website hostname. Keep repeated query parameters
and unknown fields; do not deduplicate or invent aliases locally. Upstream owns
missing-domain errors and unknown-domain empty successes. Definitions include
full tags such as code.doc and named params with description, required, and
sort_order. This operation consumes no search quota and is not search usage.

## Transport, authentication, and configuration

Only the three fixed paths and their methods are reachable through the server
key. All require enable_anysearch and a nonempty server key. All are protected
by ghc-api's optional user-token gate, including the quota-free catalog endpoint.
Client Authorization/x-api-key headers are never forwarded; upstream sees only
the server's Bearer key. The catalog's raw query string retains repeats/encoding,
except client api_key fields (including percent-encoded names) are removed before
forwarding and auditing. Do not turn the proxy's query-string auth into a leak
to the provider or the shared request cache.

Configuration: enable_anysearch, anysearch_api_key OR anysearch_api_key_file,
anysearch_base_url, anysearch_timeout (30s), anysearch_extract_timeout (60s), and
log_anysearch_requests. The existing enable/key/base settings cover all three
services; existing installations need no new setting merely to enable them.
Base paths support trusted private reverse proxies; arbitrary caller URLs do not.
Key files are read at startup, not discovered/created. They should be outside the
repository with mode 0600. No key goes into Chat template data or runtime-config
responses. Invalid settings fail startup without printing secret values.

There is no anonymous mode, auto-registration, automatic retry, provider fallback,
result fusion, or upstream redirect following. Missing configuration is 503;
timeout is 504; other transport failure is 502. Exceptions are not interpolated
into client errors or routine logs. Normal responses retain the native body,
status, Retry-After and X-Request-ID, using the existing HTTP header filter.

## Sensitive-response boundary

Search AND Extract may return generated credentials in anonymous HTTP 402
messages. Even though this proxy only uses keys, **every 402 on every endpoint**
is sensitive regardless of content type, parseability, or regex matches.

The transport consumes/closes the upstream response and returns only a safe
SearchResponse (historical class name retained for compatibility):

- 402: fixed quota error, HTTP 402.
- 401/403: fixed server-credential/access error, HTTP 503.
- Redirect: fixed HTTP 502, no Location forwarded.
- These responses have only safe fixed headers/envelopes and Cache-Control:
  no-store. Only a validated UUID v4 upstream request ID survives; otherwise use
  the local UUID. Preserve original HTTP status as safe audit metadata.

Raw sensitive bytes/headers never enter Flask, stdout, audit files, request cache,
exports, browser history, or later model requests. The browser additionally does
not read non-success response bodies, even if a misconfigured intermediary fails
to sanitize them. No generated credentials are saved or used to resubmit a call.

## Chat tool layer

One visible selector: Off / Native / Web IQ / AnySearch. Existing localStorage
preferences are migrated; unavailability never chooses another provider silently.
Native retains its Copilot GPT Responses restrictions. A provider can expose
multiple function tools through the common registry. Only a tool actually declared
for the turn may execute, even if the selection changes during an in-flight call.

AnySearch declares:

- anysearch_search: query and max_results plus optional tag, params, zone,
  language, format. Usually use query alone. Specialized searches should first
  discover a valid tag and its required params, not invent values.
- anysearch_extract: url only. Fetching remains upstream, never in the browser.
- anysearch_sub_domains: domain as a nonempty string array, encoded as repeated
  query parameters. Its description explicitly rules out DNS interpretation.

A typical sequence is capability lookup -> tagged search -> extraction -> answer.
All tools share the existing three-round budget and the final forced-answer round.
Multiple calls can occur in a round; the budget is not a hard request-count cap.
Keep cancellation, call-ID correlation, history, and citation deduplication across
Chat Completions, Responses, and Messages. Reject undeclared tools and tool calls
in the forced-answer round. Search/extraction failures return safe tool results,
not an automatic switch to another provider or a failed chat turn.

Success requires HTTP success, code 0, and the endpoint's expected data shape.
Empty search/catalog arrays are valid; malformed bodies are not empty successes.
The UI defaults to five search results/Markdown; search content is clipped to
3000 characters, snippets to 1000, and extracted content to 12000, with truncation
markers. Native HTTP responses are never trimmed by these context policies.
Web content is explicitly untrusted. Extraction contributes its URL to citations;
capability metadata does not manufacture web citations.

## Observability and testing

All three operations use `<config_dir>/anysearch/YYYY-MM-DD.jl`, independently
controlled by log_anysearch_requests. Disabling this dedicated log does not disable
the shared request cache, whose existing body-size limits apply. Normal dedicated
audit records are untruncated; sensitive errors are always sanitized first.

Use distinct model labels/counters: anysearch_search / anysearch.search,
anysearch_extract / anysearch.extract, anysearch_sub_domains / anysearch.sub_domains.
LLM tokens stay zero. Catalog records have uses_search_quota false and never
increment search counters; all API traffic still appears in the request browser.
Record safe query parameters, extraction URL, local/upstream IDs, status, business
code, duration, and applicable result count. Business failures/malformed HTTP 200s
retain the wire status but are marked errors in audit/cache and counters.

Tests cover all native paths/methods, query repeats and credential stripping,
Extract's strict-body passthrough, no local fetch, auth separation, 402 across all
sinks, status/header handling, failures, configuration, and no retries. Browser-free
JS tests compose all three tools in each protocol, including errors, cancellation,
round limits, and unselected-tool rejection. Real-browser layout tests guard the
selector's visibility. Existing Web IQ regressions remain in place. Live HTTP
smoke checks, when explicitly run, are separate from CI; they do not prove live
model reasoning/tool choice. The demo CLI covers all three native endpoints.

The extension's live check made exactly three requests: a code/finance catalog
lookup (2 domains, 8 capabilities), a one-result search, and extraction of
https://example.com/ (156 characters). All returned HTTP 200 with valid native
data shapes. No retries, model calls, service restarts, or runtime config changes
were made. The earlier Search-only integration was also tested independently.

AnySearch MCP proxying, DNS subdomain enumeration, anonymous credential management,
automatic retries/fallback and fusion are not implemented by this change.
