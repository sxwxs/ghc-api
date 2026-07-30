# GitHub Copilot API Proxy (ghc-api)

A Python Flask application that serves as a proxy server for GitHub Copilot API, providing OpenAI and Anthropic API compatibility with caching and monitoring capabilities.

## Features

- **OpenAI API Compatibility**: `/v1/chat/completions` endpoint
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

## Maintenance Guides

- [Anthropic Messages to Responses compatibility warning runbook](ANTHROPIC_RESPONSES_WARNING_RUNBOOK.md)

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
- `--delete-github-token`: Delete the locally saved `github_token.txt` and exit
- `--github-device-login`: Run GitHub Device Flow, replace the locally saved token, and exit
- `-v` or `--version`: Show version (for example `ghc-api 1.0.22`)
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

# Streaming reliability
upstream_read_timeout: 1800   # Read timeout (seconds) for each upstream Copilot request
sse_keepalive_interval: 30    # Send a keepalive ping to the client when a stream is idle
                              # this many seconds (keeps clients like Claude Code from
                              # timing out while the model "thinks"). Set 0 to disable.
```

### Token Management

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

### Email and API-Token Authentication (Optional)

Authentication is disabled by default for local use. The legacy top-level `enable_auth: true` setting and `--enable-auth` CLI flag remain API-token-only for backward compatibility; they do not require MailDispatch or protect dashboard routes, and the original direct `/signup` plus administrator-approval flow remains available. The new nested `auth.enabled: true` mode is for nginx + HTTPS public deployments and uses two complementary mechanisms:

- maglink email sessions protect the dashboard and management routes;
- approved `gha_...` API tokens continue to protect LLM endpoints used by SDKs and CLI tools.

Configure `~/.ghc-api/config.yaml`:

```yaml
auth:
  enabled: true
  hostname: "ghc.example.com"
  secret_key: "replace-with-a-strong-random-secret"
  allow_public_registration: false
  admin_emails:
    - "admin@example.com"
  trust_proxy_headers: true
  maildispatch:
    endpoint: "https://mail.example.com/api/v1/messages"
    api_key: "md_live_..."
    sender_id: "system"
```

`hostname` is a hostname only; maglink builds HTTPS verification URLs from it. The MailDispatch key needs `mail:send` and `mail:authentication`. Bind the backend privately behind nginx before enabling `trust_proxy_headers`.

**Email registration and approval**:

1. The user opens `/signup`, solves the CAPTCHA, and requests email verification.
2. The waiting browser displays a device code. The email contains only a private verification link.
3. Opening the link is side-effect-free; the user must POST the device code on the confirmation page.
4. The waiting browser consumes the verified email once and creates a `pending` account.
5. An administrator approves the account from Code Agent Manager.
6. Only an email-verified, approved account can sign in or use its API token.

If final registration data such as `user_id` is invalid or already used, the verified email remains in the waiting browser session so the form can be corrected without sending another verification email. Eligibility is checked again when the user record is actually created.

When `allow_public_registration: false`, an administrator must add the email user first. Configured `admin_emails` can bootstrap the management UI without a `users.json` approval record. Multiple administrators are supported.

Existing records without an email remain valid legacy API-token users. New email records and legacy records coexist in the same `users.json`.

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

**Per-user dashboard views**: with auth on, the request browser, statistics, and cross-machine token-usage overview retain their user filter. Requests issued while auth is off remain under `anonymous`.

**Storage**:

- `users.json` remains under the existing OneDrive/local registry selection and is backward compatible with legacy users;
- maglink's atomic pending requests and rate counters use local SQLite at `<config dir>/maglink.db` by default;
- do not place the SQLite database in OneDrive;
- Flask `secret_key` and the MailDispatch API key are excluded from OneDrive config sync and preserved locally when synced configuration is restored.

## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Chat completions
- `POST /chat/completions` - Chat completions (without v1 prefix)
- `GET /v1/models` - List available models
- `GET /models` - List available models (without v1 prefix)

### Anthropic Compatible

- `POST /v1/messages` - Messages API (Anthropic format)

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

Active only when `auth.enabled: true`. Dashboard and management endpoints require a maglink administrator session unless noted as public.

- `GET /login` - Email login waiting page (public)
- `GET /account` - Current signed-in email user's account and API token
- `GET /signup` - Verified registration page (public)
- `POST /signup` - Consume the waiting browser's verified email and create/complete a pending user
- `/api/auth/*` - maglink CAPTCHA, login request, confirmation, status, state, and logout endpoints
- `/api/register/*` - maglink email-only verification endpoints
- `GET /api/users-list` - Token-redacted list for authenticated dashboard filters
- `GET /api/users` - Full user list (administrator)
- `POST /api/users` - Add an invited email user (administrator)
- `POST /api/users/<user_id>/approve` - Approve a user (administrator)
- `POST /api/users/<user_id>/revoke` - Revoke a user (administrator)
- `DELETE /api/users/<user_id>` - Remove a user (administrator)

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

Leave `auth.enabled: false` for the default local-only use case. Public auth is designed for an nginx + HTTPS deployment: nginx terminates TLS and forwards all routes, while ghc-api enforces maglink sessions on dashboard/management routes and API tokens on LLM routes.

### Sample nginx config

```nginx
server {
    listen 443 ssl http2;
    server_name ghc.example.com;

    # ssl_certificate / ssl_certificate_key go here

    location / {
        proxy_pass http://127.0.0.1:8313;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_read_timeout 1800s;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

Bind ghc-api to a private interface such as `127.0.0.1`; do not expose the backend port directly. Set `auth.hostname` to the nginx `server_name`. Enable `auth.trust_proxy_headers` only when nginx is the exclusive trusted ingress and overwrites forwarding headers.

### Security notes

- Keep HTTPS enabled because auth sessions always use `Secure`, `HttpOnly`, `SameSite=Lax` cookies.
- Do not add nginx Basic Auth in front of LLM routes: SDK clients need their Bearer API Token to reach ghc-api unchanged.
- Keep `proxy_buffering off` and a long read timeout for SSE responses.
- The MailDispatch API key must have only `mail:send` and `mail:authentication` and should be restricted to the configured sender.
- MailDispatch recipient policy must allow every application user who may receive registration or login mail.
- `GET` confirmation pages never approve a login or registration; approval requires the emailed token plus the device code in a deliberate POST.

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
- **Lightweight Storage**: In-memory request cache, JSON user registry, and SQLite maglink state when public auth is enabled
- **RESTful API Design**: Follows REST conventions

## License

MIT License
