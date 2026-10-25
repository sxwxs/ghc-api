# Model metadata

Clients can discover token limits and capabilities in the normal model listing:

```text
GET /v1/models
GET /models
GET /proxy/<profile>/v1/models
```

Retrieve one entry using its exact public model ID:

```text
GET /v1/models/<model_id>
GET /models/<model_id>
GET /proxy/<profile>/v1/models/<model_id>
```

Model IDs containing `/` are supported. Detail responses contain the same object
as the corresponding list entry. An unknown model returns HTTP 404 with
`error.code: "model_not_found"`; an unknown proxy profile returns
`"proxy_not_found"`. All these routes require the usual user token when
`enable_auth: true`.

The existing OpenAI/Anthropic listing fields (`id`, `object`, `type`, `created`,
`created_at`, `owned_by`, `display_name`) are accompanied by the following fields:

| Field | Type | Meaning / Copilot source |
| --- | --- | --- |
| `model_type` | string or null | `capabilities.type`, e.g. `chat` or `embeddings`; configured proxies use `decision` for System One only models and `chat` otherwise |
| `context_window` | integer or null | Total context budget, from `capabilities.limits.max_context_window_tokens` |
| `max_input_tokens` | integer or null | Separate input limit, from `capabilities.limits.max_prompt_tokens` |
| `max_output_tokens` | integer or null | Output limit, from `capabilities.limits.max_output_tokens` |
| `input` | string array or null | `text`, plus `image` when `capabilities.supports.vision` is true |
| `reasoning` | boolean or null | `capabilities.supports.reasoning`; inferred as true if reasoning effort choices exist |
| `reasoning_efforts` | string array or null | `capabilities.supports.reasoning_effort`, preserving the advertised values |
| `tool_calls` | boolean or null | `capabilities.supports.tool_calls` |
| `streaming` | boolean or null | `capabilities.supports.streaming` |
| `supported_endpoints` | string array or null | The model's advertised endpoints, including locally configured Chat Completions support |
| `capabilities` | object or null | Original Copilot capability object, including fields without a normalized equivalent |

An absent or invalid limit is `null`. Missing capabilities are `null`, while an
explicit `false` remains false. Clients should apply their own fallback only for
`null`; there is no universal default context size. Chat and embedding models
advertise text input; a missing vision flag does not advertise image support.
The context budget and maximum output are independent constraints: the actual
output must also fit the remaining context budget. The input limit is never
derived by subtracting the maximum output from the context window.

For configured proxies, `context_window`, `max_output_tokens`, `input`, and
`reasoning` come from `upstream-proxies.yaml`, including its existing defaults.
Fields without a configuration source are `null`, except `streaming: false` for
models that only expose the nonstreaming System One API. Proxy catalogs publish only
public metadata; upstream URLs, model mappings, credentials, and routing headers
are excluded. `/proxy/models` uses the same metadata and additionally includes
the existing first-party profile routing information.

Example list entry (illustrative limits):

```json
{
  "id": "example-model",
  "object": "model",
  "type": "model",
  "created": 0,
  "created_at": "1970-01-01T00:00:00Z",
  "owned_by": "Example",
  "display_name": "Example Model",
  "model_type": "chat",
  "context_window": 200000,
  "max_input_tokens": 180000,
  "max_output_tokens": 32000,
  "input": ["text", "image"],
  "reasoning": true,
  "reasoning_efforts": ["low", "medium", "high"],
  "tool_calls": true,
  "streaming": true,
  "supported_endpoints": ["/responses", "/v1/chat/completions"],
  "capabilities": {
    "type": "chat",
    "limits": {
      "max_context_window_tokens": 200000,
      "max_prompt_tokens": 180000,
      "max_output_tokens": 32000
    },
    "supports": {
      "vision": true,
      "reasoning_effort": ["low", "medium", "high"],
      "tool_calls": true,
      "streaming": true
    }
  }
}
```

## Using the metadata in pi

A pi extension or configuration generator can map each chat entry to its model
definition using the following fields:

| ghc-api | pi model definition |
| --- | --- |
| `id` | `id` |
| `display_name` | `name` |
| `context_window` | `contextWindow` |
| `max_output_tokens` | `maxTokens` |
| `input` | `input` |
| `reasoning` | `reasoning` |

Select pi's API adapter from `supported_endpoints` and use the same base URL as
the catalog (`/v1` or `/proxy/<profile>/v1`). Filter out `model_type: "embeddings"`
and `"decision"` for chat model selection, and select only entries advertising a
supported chat API. Handle `null` before constructing a pi model definition
and supply client defaults for required values that are unknown. These endpoints
make the metadata available to clients; a pi integration must read and apply it.

### Run the pi extension

The [example extension](../examples/pi/ghc-api.ts) fetches the catalog during
startup and registers the `ghc-api` provider with each model's advertised limits.
Run from the repository root with a server running the metadata implementation:

```bash
export GHC_API_BASE_URL=http://localhost:8313/v1
# If ghc-api authentication is enabled, also set GHC_API_KEY to your user token.
pi -e ./examples/pi/ghc-api.ts --list-models ghc-api
pi -e ./examples/pi/ghc-api.ts --provider ghc-api --model gpt-6-luna
```

For a configured upstream profile, set `GHC_API_BASE_URL` to
`http://localhost:8313/proxy/<profile>/v1`. The extension selects Responses,
Chat Completions, or Messages according to the advertised endpoints. It skips
embedding and decision models, entries without a supported chat API, and entries
with unknown context or output limits. Limits are refreshed when pi starts or
reloads the extension. The example supplies zero price placeholders because the
catalog does not publish prices.

Verified with pi 1.4.0 (`@earendil-works/pi-coding-agent`): its RPC model registry
received the exact context and output limits from a live Copilot catalog, and
real Chat Completions (`gemini-3.8-flash`) and Responses (`gpt-6-luna`) prompts
both completed successfully. The check used an isolated server running the
workspace code and forwarded inference through the existing ghc-api instance.

## Cache and raw metadata

Copilot listings use the in-memory catalog, loading it from Copilot if it has not
been initialized. A cached listing does not require a Copilot token refresh.
Failure to load the catalog returns HTTP 502 with
`error.code: "model_catalog_unavailable"`; a successfully loaded empty catalog
returns HTTP 200 with `data: []`.

`GET /v1/models/full/` and `GET /models/full/` return the original full catalog
with all upstream fields. Both also accept a URL without the trailing slash;
`full` is reserved for this catalog route. They load the catalog on first use
with the same error handling as the normalized listing.

The existing `POST /v1/models/full/refresh` (or `/models/full/refresh`) explicitly
refreshes the Copilot catalog and returns the updated raw metadata. Configured
proxy metadata follows the existing automatic reload of `upstream-proxies.yaml`.
