# GitHub Copilot API Proxy (ghc-api)

A Python Flask application that serves as a proxy server for GitHub Copilot API, providing OpenAI and Anthropic API compatibility with caching and monitoring capabilities.

## Features

- **OpenAI API Compatibility**: `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings` endpoints
- **Anthropic API Compatibility**: `/v1/messages` endpoint with automatic request/response translation
- **Model Listing**: `/v1/models` endpoint listing available models
- **Model Name Mapping**: Translate model names with exact and prefix-based matching
- **Token Management**: Automatic GitHub Copilot token refresh
- **Vision Support**: Handle image inputs and enable vision capabilities
- **Memory Caching**: Cache all requests and responses (up to 1000 entries)
- **Web Dashboard**: Real-time statistics and request browser
- **Request Details**: View full request/response bodies with JSON formatting
- **Export/Import**: Export and import request history as JSON Lines files
- **Optional Request File Logging**: Save completed requests to daily JSON Lines files
- **Request File Statistics**: Build reusable line-level indexes, inspect size/latency/token distributions, and drill into exact historical requests
- **Content Filtering**: Remove or add content from system prompts and tool results
- **Code Agent Manager UI**: Install Codex/Claude/Copilot CLI and manage config sync from dashboard
- **Code Agent Interaction**: Web UI to create and interact with Claude Code, Codex, and Copilot CLI agents via the Agent Client Protocol (ACP)
- **Config Sync**: Sync Claude Code, Codex, and ghc-api config files with OneDrive
- **Safe Backups**: Auto backup overwritten config files as `*.YYYYMMDD_HHMMSS.bak`
- **Machine Token Usage Logs**: Periodic token usage JSONL per machine with cross-machine overview in dashboard
- **Optional User-Token Auth**: Opt-in middleware gates LLM endpoints behind self-signup + admin-approved tokens; requests, stats, and token usage are then grouped per user
- **Configured Upstream Proxy**: Isolated `/proxy/<profile>/v1/...` routes for config-driven OpenAI Responses and Chat Completions upstreams, with private auth commands, model/header mapping, and persisted affinity routing
- **Microsoft Web IQ Search**: `/v3/search/web`, a transparent proxy for the official Web Search v3 API, backed by a server-held API key

## Maintenance Guides

- [Anthropic Messages to Responses compatibility warning runbook](ANTHROPIC_RESPONSES_WARNING_RUNBOOK.md)
- [Async JSONL logging](ASYNC_JSONL_LOGGING.md) - agreed design for moving request/search file appends off the request thread (not yet implemented)

## Installation

Install the package using pip:

```bash
pip install ghc-api
```

Or install from source:

```bash
pip install .
```

## Usage

Start the server with the `ghc-api` command:

```bash
ghc-api
```

By default, the server will start on `http://localhost:8313`.

### Command Line Options

- `-p PORT` or `--port PORT`: Specify the port to listen on (default: 8313)
- `-a ADDRESS` or `--address ADDRESS`: Specify the address to listen on (default: localhost)
- `-c` or `--config`: Generate a YAML config file in `~/.ghc-api/config.yaml`
- `--ghe-endpoint HOST`: Configure both GHE data-residency endpoints and exit
- `--delete-github-token`: Delete the locally saved `github_token.txt` and exit
- `--github-device-login`: Run GitHub Device Flow, replace the locally saved token, and exit
- `-v` or `--version`: Show version (for example `ghc-api 1.0.24`)
- `--help`: Show help message

### Configuration

The application looks for a configuration file at `~/.ghc-api/config.yaml`. You can generate this file using:

```bash
ghc-api --config
```

The config file contains:
```yaml
# Server Settings
address: localhost
port: 8313
debug: false

# GitHub Copilot Account Type
# Options: "individual", "business", "enterprise"
account_type: individual

# Optional upstream endpoint overrides. Leave empty for github.com.
github_api_base_url: ""
copilot_api_base_url: ""

# GitHub Enterprise Cloud with data residency example:
# github_api_base_url: "https://api.octocorp.ghe.com"
# copilot_api_base_url: "https://copilot-api.octocorp.ghe.com"

# Version settings (used to build request headers)
vscode_version: "1.93.0"
api_version: "2025-04-01"
copilot_version: "0.26.7"

# Model Name Mappings
model_mappings:
  # Exact match mappings
  exact:
    opus: claude-opus-4.5
    sonnet: claude-sonnet-4.5
    haiku: claude-haiku-4.5
  # Prefix match mappings
  prefix:
    claude-sonnet-4-: claude-sonnet-4
    claude-opus-4.5-: claude-opus-4.5

# Chat completions endpoint overrides
chat_completions_model_support:
  exact: []
  prefix:
    - gpt-
    - mai-code-

# Content Filtering
system_prompt_remove: []    # Strings to remove from system prompts
system_prompt_add: []       # Strings to append to system prompts
tool_result_suffix_remove: [] # Strings to remove from end of tool results

# Optional request persistence
save_request_to_file: false # If true, save completed requests to requests/YYYY-MM-DD.jl

# Optional OneDrive access gate
disable_onedrive_access: true # If true, skip all OneDrive detection/sync/shared reads

# Optional leaked tool-call recovery (direct Anthropic /v1/messages streaming)
enable_tool_call_recovery: false # If true, recover tool calls Copilot leaks as plain text

# Retry /v1/responses streams that fail before any output (enabled by default)
enable_responses_early_failure_retry: true # Transparently retry a stream that returns
                              # HTTP 200 then response.failed before any text, reasoning,
                              # or tool call. Retries stop once real output has been sent
                              # (never duplicates content) and are capped by
                              # max_connection_retries. Each retry costs upstream quota.
                              # Set false to disable.

# Streaming reliability
upstream_read_timeout: 1800   # Read timeout (seconds) for each upstream Copilot request
sse_keepalive_interval: 30    # Send a keepalive ping to the client when a stream is idle
                              # this many seconds (keeps clients like Claude Code from
                              # timing out while the model "thinks"). Set 0 to disable.
responses_pre_header_grace: 0.5 # How long /v1/responses waits for upstream response
                              # headers before committing to a streaming response and
                              # sending keepalives. Errors that arrive within this window
                              # keep their real HTTP status; later ones can only be
                              # reported as an SSE error event, so keep this below the
                              # shortest client read timeout. Clamped to [0, 5] seconds;
                              # sse_keepalive_interval: 0 disables the behavior entirely.

# Recover from undecryptable encrypted content (disabled by default)
auto_remove_encrypted_content_on_parse_error: false # If /v1/responses returns HTTP 400
                              # because encrypted reasoning or tool output content cannot
                              # be decrypted, clean request.input and retry once.
                              # Lossy; see "Encrypted Content Recovery" below.
```

### Anthropic Messages → Responses Reasoning Continuity

When `/v1/messages` is routed to a Responses-only model, ghc-api carries each
Responses reasoning item in a namespaced Anthropic `thinking` block. The
visible reasoning summary is stored in `thinking`, while the opaque
`encrypted_content` is encoded in `thinking.signature`. Clients such as Claude
Code echo that assistant block on the next turn, allowing ghc-api to reconstruct
the original Responses `reasoning` input item without a replay database,
session identifier, or encryption key.

The carrier includes its source model and wire profile. If either changes, or
if the carrier is malformed or oversized, ghc-api keeps the visible summary but
drops the opaque state and reports a compatibility warning. Synthetic carrier
blocks are removed before forwarding history to a native Anthropic model so a
forged signature never reaches `/v1/messages` upstream.

### Encrypted Content Recovery

Copilot's `/v1/responses` sometimes rejects a conversation with HTTP 400 because encrypted
reasoning blobs (`encrypted_content` on a `reasoning` item) or encrypted tool output can no
longer be decrypted — typically after a token rotation or a server-side key change. The
conversation is then permanently stuck: every follow-up turn replays the same history and
fails again.

Set `auto_remove_encrypted_content_on_parse_error: true` to let ghc-api react to such a 400
exactly once per request:

- Items whose encrypted payload *is* the content (reasoning items, messages) are dropped.
- Tool output items (`function_call_output`, `custom_tool_call_output`, ...) are **kept** with
  their encrypted blocks stripped and a placeholder body, so the paired `function_call` is not
  orphaned (removing them would make the retry fail with "No tool output found for function call").
- If a tool *call* itself must be dropped, its paired output is dropped with it.
- The cleaned request is retried once; the retry does not consume the connection-retry budget,
  and a second identical failure is returned to the client as-is.

The option is **off by default** because it is lossy — the model loses that reasoning/tool
context — and costs one extra upstream request. Every recovery is counted
(`mod.encrypted_content_removal`) and logged.

### Token Management

For GitHub Enterprise Cloud with data residency, switch both upstream endpoints with one command:

```bash
ghc-api --ghe-endpoint https://octocorp.ghe.com
```

The command accepts the tenant web host, GitHub API host, or Copilot API host, and normalizes all of these forms to the same configuration:

```text
octocorp.ghe.com
https://octocorp.ghe.com
https://api.octocorp.ghe.com
https://copilot-api.octocorp.ghe.com
```

It updates `config.yaml` while preserving its other settings and comments:

```yaml
github_api_base_url: "https://api.octocorp.ghe.com"
copilot_api_base_url: "https://copilot-api.octocorp.ghe.com"
```

GitHub API and Copilot requests use these values directly. Device Flow derives the OAuth origin by removing the `api.` prefix, so the example above signs in through `https://octocorp.ghe.com`. Restart a running server after switching; run `ghc-api --github-device-login` if the new tenant requires a different token. Invalid or non-HTTPS GHE URLs fail explicitly rather than falling back to github.com.

The application follows this priority for getting the GitHub token:

1. `GITHUB_TOKEN` environment variable
2. Token file at `~/.ghc-api/github_token.txt`
3. Interactive GitHub Device Flow authentication

To discard only the local token file, or explicitly sign in again without starting the server:

```bash
ghc-api --delete-github-token
ghc-api --github-device-login
```

The Code Agent Manager shows the latest Copilot token refresh attempt/result and can start a new Device Flow. The UI displays GitHub's short user code and verification URL; the secret device code and resulting access token remain server-side. If `GITHUB_TOKEN` is set, deleting the local file does not remove that environment variable, and it will take priority again after restart.

Copilot token refresh failures are appended as structured JSON lines to `error.log` in the ghc-api config directory. The upstream response body is retained for debugging up to 64 KiB; authentication headers and tokens are not logged.

### Config Sync and OneDrive

`ghc-api` can manage and sync these files:

- Claude Code: `~/.claude/settings.json`
- Codex: `~/.codex/config.toml`
- ghc-api: `~/.ghc-api/config.yaml` (or `%APPDATA%/ghc-api/config.yaml` on Windows)

OneDrive detection priority:

1. `~/OneDrive - *`
2. `~/OneDrive`
3. In WSL: `/mnt/c/Users/<username>/OneDrive - *` then `/mnt/c/Users/<username>/OneDrive`

To disable all OneDrive-dependent operations, set `disable_onedrive_access: true` in `config.yaml`.
When enabled, ghc-api skips OneDrive detection, config sync actions, and shared OneDrive hash reads.

Sync target folder:

- `.ghc-api/configSync` under detected OneDrive root

Machine folder:

- `.ghc-api/agents/{hostname}_{os}` where `os` is `Win`, `Linux`, or `WSL`

Hash files:

- `.ghc-api/configSync/config.sha1`
- `.ghc-api/agents/{hostname}_{os}/ghc-api/config.sha1`

Hashes are recalculated when local config file timestamp is newer than the hash file.
On startup, ghc-api checks synced files and prints config differences to stdout (and UI indicator if different).

### Token Usage Logging

Every 5 minutes, ghc-api writes token usage delta (if non-zero) to:

- OneDrive mode: `.ghc-api/agents/{hostname}_{os}/token_usage.jl`
- Fallback when OneDrive is unavailable: `~/.ghc-api/token_usage.jl`

Also flushes pending usage on shutdown (`Ctrl+C`/termination/normal exit).

Each JSONL line includes:

- `timestamp` (unix seconds)
- `models` list with:
  - `model`
  - `request_count`
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`

### Request File Logging

When `save_request_to_file: true`, ghc-api appends each completed request to:

- `<ghc-api config dir>/requests/YYYY-MM-DD.jl`

The saved `.jl` line format is the same as dashboard export (`/api/requests/export`) and can be imported by dashboard import (`/api/requests/import`).

Web IQ searches are recorded as requests too (model name `webiq_search`, zero tokens), so they appear in these files, in `/api/stats` and in the request statistics alongside LLM traffic. That copy obeys `cache_max_request_size` like any other; the untruncated record of a search lives in `<ghc-api config dir>/webiq/YYYY-MM-DD.jl`.

Open `/request-stats` to select one or more daily files and generate request-size, request-duration, and billing-token distributions overall, by model, or by exact HTTP response code. Scans run asynchronously and write one lightweight JSONL sidecar plus metadata per request file under `requests/.request-stats-index/`. Each sidecar row stores scalar metrics and the source byte offset/length/hash, never request/response bodies or headers. If a source file is unchanged its sidecar is reused without reopening the source; append-only growth is indexed incrementally, while truncated, replaced, corrupt, or incompatible files are rebuilt safely.

Histogram bars are interactive: selecting a bucket shows the exact indexed requests that contributed to it. Each result opens a stable `/request-file-detail` link which seeks directly to the original `.jl` line and verifies its SHA-256 before returning the complete persisted record. These links remain valid while the source file is unchanged; a changed source reports that the index must be rebuilt. Detail rendering is capped at 4 MiB per JSONL line.

The token views follow the fields persisted by ghc-api: input not cached (`input_tokens`), cache write (`cache_creation_input_tokens`), cache read (`cache_read_input_tokens`), output (`output_tokens`), and their total. Provider compatibility paths use the same existing field mappings as the request log itself. Distributions include every persisted request attempt, including failed attempts; failures commonly record zero usage tokens.

### Code Agent Interaction

The Code Agent page (`/agent`) provides a web interface for interacting with AI coding agents via the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/). Supported agents:

| Agent | Package | Install |
|-------|---------|---------|
| Claude Code | `@agentclientprotocol/claude-agent-acp` | `npm install -g @agentclientprotocol/claude-agent-acp` |
| Codex | `codex-acp` | Download from [GitHub releases](https://github.com/zed-industries/codex-acp/releases) |
| Copilot CLI | `@github/copilot` | `npm install -g @github/copilot` |

Agent binaries are resolved in order: environment variable override (`CLAUDE_ACP_BINARY`, `CODEX_ACP_BINARY`, `COPILOT_CLI_BINARY`), then PATH lookup, then npm global packages.

Session data is stored in:

- OneDrive mode: `.ghc-api/agents/{hostname}_{os}/sessions/`
- Fallback: `~/.ghc-api/sessions/` (or `%APPDATA%/ghc-api/sessions/` on Windows)

Recent working directories are persisted to `workdirs.json` in the same location. Sessions from other machines are browsable via the machine selector dropdown when OneDrive is enabled.

### User-Token Authentication (Optional)

When you want to share a single ghc-api instance among multiple people without giving everyone unrestricted access to the deployer's Copilot quota, enable token auth:

```bash
ghc-api --enable-auth
# or set in ~/.ghc-api/config.yaml:
#   enable_auth: true
```

Once enabled, LLM endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, their non-`/v1` aliases, `/v3/search/web`, and configured `/proxy/<profile>/v1/...` routes) require an approved user token. Dashboard and admin endpoints stay open at the Flask layer — they're expected to be gated by a reverse proxy in production (see [Production Deployment](#production-deployment)).

**Self-signup flow**:

1. User opens `http://<host>:8313/signup`, fills `user_id` (letters/digits/`_-.`, max 64 chars) and an optional display name, submits.
2. Server generates a token of the form `gha_<43 url-safe chars>`, returns it once. Status is `pending`.
3. Admin opens the dashboard → **Code Agent Manager** → **Users** section → clicks **Approve** next to the new user. (Or `curl -X POST http://localhost:8313/api/users/<id>/approve`.)
4. The user can now use the token. Revocation and deletion are available from the same panel.

**Token presentation** (middleware accepts any of these, first match wins):

1. `Authorization: Bearer <token>` — OpenAI SDK, Claude Code (`ANTHROPIC_AUTH_TOKEN`), Codex, curl
2. `x-api-key: <token>` — Anthropic SDK (`ANTHROPIC_API_KEY`)
3. `?api_key=<token>` query parameter — curl one-liners

**Client configuration examples** (assuming server at `localhost:8313` and an approved token `gha_abc...xyz`):

*Claude Code* — `~/.claude/settings.json`:
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8313",
    "ANTHROPIC_AUTH_TOKEN": "gha_abc...xyz"
  }
}
```
Note: `ANTHROPIC_BASE_URL` is **without** `/v1`. Prefer `ANTHROPIC_AUTH_TOKEN` over `ANTHROPIC_API_KEY` for proxies.

*Codex* — `~/.codex/config.toml`:
```toml
model_provider = "ghc-api"
model = "gpt-4o"

[model_providers.ghc-api]
name = "GHC API Proxy"
base_url = "http://localhost:8313/v1"
env_key = "GHC_API_TOKEN"
wire_api = "chat"   # or "responses"
```
Then `export GHC_API_TOKEN=gha_abc...xyz`. Note: Codex's `base_url` **includes** `/v1`.

*OpenAI Python SDK*:
```python
client = OpenAI(base_url="http://localhost:8313/v1", api_key="gha_abc...xyz")
```

*Anthropic Python SDK*:
```python
client = anthropic.Anthropic(base_url="http://localhost:8313", api_key="gha_abc...xyz")
```

*curl*:
```bash
curl http://localhost:8313/v1/chat/completions \
  -H "Authorization: Bearer gha_abc...xyz" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

**Per-user dashboard views**: with auth on, the request browser, statistics, and cross-machine token-usage overview all gain a "Filter by user" dropdown. Requests issued before auth was enabled (and any anonymous calls when auth is off) show up under a single `anonymous` bucket.

**Token registry storage**:
- If OneDrive is detected and `disable_onedrive_access: false`: `{OneDrive}/.ghc-api/configSync/users.json` (shared across machines — register once, use anywhere).
- Otherwise: `~/.ghc-api/users.json` (local-only).

The registry file is re-read whenever its mtime changes (checked every 5 seconds), so approval / revocation on one machine propagates to others as soon as OneDrive syncs the file.

## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Chat completions
- `POST /chat/completions` - Chat completions (without v1 prefix)
- `POST /v1/embeddings` - Create embeddings
- `POST /embeddings` - Create embeddings (without v1 prefix)
- `GET /v1/models` - List available models
- `GET /models` - List available models (without v1 prefix)

### Anthropic Compatible

- `POST /v1/messages` - Messages API (Anthropic format)

### Microsoft Web IQ Search

- `POST /v3/search/web` - Web Search v3, backed by the server-held Web IQ API key

A transparent proxy for the [official Microsoft Web Search v3
API](https://webiq.microsoft.ai/documentation/api-reference/web/). The request
body is forwarded as the bytes the client sent, and the upstream status, headers
and body come back verbatim. A client written against `api.microsoft.ai` works
here by changing only the base URL — the same deal the OpenAI- and
Anthropic-shaped endpoints offer.

```bash
curl -X POST http://localhost:8313/v3/search/web \
  -H "content-type: application/json" \
  -d '{"query": "latest trends in LLM RAG", "maxResults": 10, "contentFormat": "passage"}'
```

What the proxy adds is key custody (`webiq_api_key` never leaves the server), the
optional user-token auth gate, and logging. It adds nothing to the search itself:

- **There are no server-side search settings.** Every parameter and every default
  is Microsoft's, including the defaults for what a request omits (`maxResults`
  10, `contentFormat` html, `maxLength` 10000). A server-side default would
  silently make this endpoint disagree with the API it claims to be. If you want
  passage format, ask for it in the request — that is what `/chat` and
  `scripts/webiq_search_demo.py` do.
- **There is no parameter whitelist and no local validation.** A parameter
  Microsoft adds tomorrow works here today, and an invalid request gets the
  authoritative upstream error instead of an imitation of it.
- **Errors and error headers are passed through too**, so `Retry-After` on a 429
  reaches the client that has to back off. The single exception is upstream
  401/403: those mean *this server's* key was rejected, so they surface as 503
  with an explicit message rather than being confused with this proxy rejecting
  the caller's token.
- **A client's own `x-apikey` is ignored**, never forwarded, and redacted before
  the request is logged. Searches always spend this server's key and quota.

The proxy never searches on a model's behalf. Clients declare the `webiq_search`
function tool, the model decides whether and what to search, and the client
executes the tool call against this endpoint; `/chat` does that automatically
when the Web IQ toggle is on. That tool schema stays narrow (`query`,
`max_results`) on purpose — it is a prompt surface, not the API — so the client
is what turns those arguments into a full official request. See
`scripts/webiq_search_demo.py`.

Every call is written to `<ghc-api config dir>/webiq/YYYY-MM-DD.jl`
(`log_webiq_requests`, on by default), which is the only untruncated record of a
search, and added to the shared request cache so it appears in the request list,
full-text search, detail view and export under the model name `webiq_search`.

### Configured Upstream Proxy

Optional private upstream profiles expose isolated routes without changing the existing Copilot endpoints:

- `POST /proxy/<profile>/v1/responses`
- `POST /proxy/<profile>/v1/chat/completions`
- `GET /proxy/<profile>/v1/models`
- `GET /proxy/models` - First-party model catalog used by the built-in Chat page

Configured models are selectable from `/chat` and participate in the same request cache, request browser, token statistics, per-user usage reporting, and persisted request-file statistics as existing endpoints. Configuration lives in a separate private `upstream-proxies.yaml` file and supports independent upstream authentication, profile/API/model headers, public-to-upstream model mapping, and response-header affinity persistence. See [Configured Upstream Proxy](UPSTREAM_PROXY.md).

### Dashboard & Monitoring

- `GET /` - Web dashboard with statistics
- `GET /requests` - Request browser page
- `GET /request-stats` - Multi-file persisted request statistics page
- `GET /api/request-stats/files` - List daily request files and index state without scanning contents
- `POST /api/request-stats/jobs` - Start an asynchronous statistics/index job for selected files
- `GET /api/request-stats/jobs/<id>` - Read job progress and completed statistics
- `POST /api/request-stats/jobs/<id>/cancel` - Cancel an active statistics job
- `GET /api/request-stats/datasets/<id>/requests` - Paginate requests contributing to a selected histogram bucket
- `GET /request-file-detail` - Stable historical request detail page
- `GET /api/request-stats/request-detail` - Read and hash-verify one indexed JSONL request by file offset
- `GET /api/runtime-config` - Read in-memory runtime config
- `POST /api/runtime-config` - Update in-memory runtime config (no file write)
- `GET /api/stats` - JSON statistics endpoint
- `GET /api/requests` - Paginated list of requests
- `GET /api/requests/search` - Full-text search in request/response bodies
- `GET /api/requests/export` - Export all requests as JSON Lines file
- `POST /api/requests/import` - Import requests from JSON Lines file
- `GET /api/request/<id>` - Individual request details
- `GET /api/request/<id>/request-body` - Request body only
- `GET /api/request/<id>/response-body` - Response body only
- `GET /api/config-manager/status` - Config manager status and diff info
- `POST /api/config-manager/install-tools` - Install Codex/Claude/Copilot CLI
- `POST /api/config-manager/sync-to-onedrive` - Sync local config to OneDrive
- `POST /api/config-manager/sync-from-onedrive` - Copy OneDrive config to local machine with backups
- `GET /api/config-manager/token-usage?range=all|day|week|month` - Cross-machine token usage overview
- `GET /api/config-manager/config-hashes` - Config hash overview for shared OneDrive and each machine (with create time)

### User Authentication

Active only when `enable_auth: true`. See [User-Token Authentication](#user-token-authentication-optional) above.

- `GET /signup` - Self-signup form (public)
- `POST /signup` - Create a pending user, return token (public)
- `GET /api/users-list` - User list without tokens, for filter dropdowns (public)
- `GET /api/users` - Full user list including tokens (admin: gate behind reverse proxy)
- `POST /api/users/<user_id>/approve` - Mark a pending user as approved (admin)
- `POST /api/users/<user_id>/revoke` - Revoke an approved user (admin)
- `DELETE /api/users/<user_id>` - Remove a user from the registry (admin)

Per-user filtering is also available on existing endpoints via the `?user=<user_id>` query parameter: `/api/stats`, `/api/requests`, `/api/requests/search`, `/api/config-manager/token-usage`.

### Code Agent (ACP)

- `GET /agent` - Code agent interaction page
- `POST /api/agent/sessions` - Create a new agent session
- `GET /api/agent/sessions` - List sessions (paginated, filterable by machine)
- `GET /api/agent/sessions/<id>` - Get session detail with message history
- `POST /api/agent/sessions/<id>/prompt` - Send a prompt (returns SSE stream)
- `POST /api/agent/sessions/<id>/cancel` - Cancel the current prompt
- `DELETE /api/agent/sessions/<id>` - Terminate a session
- `GET /api/agent/machines` - List available machine names
- `GET /api/agent/workdirs` - List recent working directories

## Example Usage

### With OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8313/v1",
    api_key="not-needed"  # Token is managed by the proxy
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### With Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8313",
    api_key="not-needed"  # Token is managed by the proxy
)

message = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

### With cURL

```bash
# Chat completions
curl http://localhost:8313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# List models
curl http://localhost:8313/v1/models
```

## Dashboard

Access the web dashboard at `http://localhost:8313/` to:

- View overall statistics (total requests, data transfer)
- See per-model usage statistics
- See per-endpoint analytics
- Browse recent requests
- View detailed request/response bodies
- Use Request Stats (`/request-stats`) to multi-select persisted daily files, compare overall/model/response-code distributions, click histogram buckets for matching requests, and open hash-verified historical details
- Use Code Agent Manager to:
  - Install code-agent CLIs
  - Sync config files to/from OneDrive
  - See config mismatch alerts
  - View token usage overview by machine/model with time-range and machine filters
  - View config hash overview by machine and shared OneDrive hash with create times
- Use Code Agent page (`/agent`) to:
  - Create interactive sessions with Claude Code, Codex, or Copilot CLI
  - Send prompts and receive real-time streaming responses (text, tool calls, thinking)
  - Toggle verbose mode for detailed tool inputs/outputs, stdout/stderr, and token usage
  - Browse sessions across machines via OneDrive
  - Resume viewing past session history

## Production deployment

When you expose ghc-api beyond `localhost` (sharing a single instance with other people, putting it on a VPS, etc.), put a reverse proxy in front to authenticate admin paths. ghc-api intentionally does **not** authenticate dashboard pages or admin APIs at the Flask layer — that responsibility is delegated to your reverse proxy.

### Path classification

| Category | Paths | How to gate |
|---|---|---|
| **Public — LLM API** | `POST /v1/chat/completions`, `/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses`, `/responses`, `/v1/embeddings`, `/embeddings`, `/v3/search/web`, configured `/proxy/<profile>/v1/responses`, `/proxy/<profile>/v1/chat/completions`, `GET /v1/models`, `/models`, `/v1/models/full/`, `/models/full/`, `/proxy/<profile>/v1/models` | No basic-auth (clients send `Authorization: Bearer <user-token>`); ghc-api's own middleware checks the user token when `enable_auth=true` |
| **Public — signup** | `GET /signup`, `POST /signup`, `GET /api/users-list` (token-redacted) | No basic-auth — anyone may request an account |
| **Admin — user mgmt** | `GET /api/users`, `POST /api/users/<id>/approve`, `POST /api/users/<id>/revoke`, `DELETE /api/users/<id>` | basic-auth — `GET /api/users` returns plaintext tokens |
| **Admin — config & data** | `POST /api/runtime-config`, `POST /api/config-manager/install-tools`, `POST /api/config-manager/sync-to-onedrive`, `POST /api/config-manager/sync-from-onedrive`, `POST /api/requests/import` | basic-auth — affect global state |
| **Admin — dashboard & inspection** | `GET /`, `/requests`, `/request-stats`, `/request-file-detail`, `/code-agent-manager`, `/chat`, `/agent`, all `/api/request-stats/*`, `GET /api/stats`, `/api/requests*`, `/api/request/<id>*`, `/api/config-manager/*`, `/api/agent/*` | basic-auth — request data and usage aggregates may expose other users' activity |

### Sample nginx config

Default-deny strategy: protect everything with basic-auth, then explicitly allow the public paths.

```nginx
server {
    listen 443 ssl http2;
    server_name ghc.example.com;

    # ssl_certificate / ssl_certificate_key go here

    # Default for the whole server: admin basic-auth required.
    auth_basic "ghc-api admin";
    auth_basic_user_file /etc/nginx/ghc-api.htpasswd;

    # ---- Public: LLM API (auth is enforced by ghc-api itself via user tokens) ----
    location /v1/ {
        auth_basic off;
        proxy_pass http://127.0.0.1:8313;
        proxy_buffering off;          # SSE / streaming responses
        proxy_read_timeout 1200s;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    # Aliases without the /v1 prefix
    location ~ ^/(chat/completions|responses|models)(/|$) {
        auth_basic off;
        proxy_pass http://127.0.0.1:8313;
        proxy_buffering off;
        proxy_read_timeout 1200s;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Configured upstream proxy routes use the same client bearer-token model.
    location ^~ /proxy/ {
        auth_basic off;
        proxy_pass http://127.0.0.1:8313;
        proxy_buffering off;
        proxy_read_timeout 1200s;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # ---- Public: signup page and token-redacted user list ----
    location = /signup {
        auth_basic off;
        proxy_pass http://127.0.0.1:8313;
    }
    location = /api/users-list {
        auth_basic off;
        proxy_pass http://127.0.0.1:8313;
    }

    # ---- Everything else: admin basic-auth applies ----
    location / {
        proxy_pass http://127.0.0.1:8313;
        proxy_buffering off;
        proxy_read_timeout 1200s;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Create the password file (use bcrypt via `-B`):

```bash
sudo htpasswd -cB /etc/nginx/ghc-api.htpasswd admin
# add more admins later without -c:
sudo htpasswd -B /etc/nginx/ghc-api.htpasswd alice
```

### Critical caveats

- **Never apply `auth_basic` to LLM API paths.** Clients like Codex, Claude Code, and the OpenAI SDK send `Authorization: Bearer <token>`, not HTTP basic. nginx would 401 the request before ghc-api ever sees it.
- **Always set `proxy_buffering off;` and a long `proxy_read_timeout`** for any location that forwards LLM traffic — otherwise streamed responses stall or get truncated.
- **The two `Authorization` schemes don't conflict**: basic-auth lives in admin `location` blocks (`Authorization: Basic ...`), user tokens live in LLM `location` blocks (`Authorization: Bearer ...`). They never coexist on the same request.
- **For local-only single-user use without nginx**, bind ghc-api to localhost so the admin endpoints aren't reachable from the network: `ghc-api --enable-auth -a 127.0.0.1`.

## Architecture

- **Modular Design**: Organized into separate modules for maintainability
  - `main.py` - Entry point and configuration loading
  - `app.py` - Flask application factory
  - `config.py` - Configuration constants and model mappings
  - `cache.py` - Request caching and statistics
  - `translator.py` - OpenAI/Anthropic format translation
  - `streaming.py` - Streaming response handling
  - `token_manager.py` - GitHub token management
  - `routes/` - API endpoint handlers (openai, anthropic, dashboard, agent)
  - `acp/` - Agent Client Protocol implementation (JSON-RPC 2.0 over subprocess stdio)
- **Thread-Safe Caching**: Uses threading locks for concurrent access
- **Memory-Based Storage**: No external database dependencies
- **RESTful API Design**: Follows REST conventions

## License

MIT License
