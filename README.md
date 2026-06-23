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
- **Content Filtering**: Remove or add content from system prompts and tool results
- **Code Agent Manager UI**: Install Codex/Claude/Copilot CLI and manage config sync from dashboard
- **Code Agent Interaction**: Web UI to create and interact with Claude Code, Codex, and Copilot CLI agents via the Agent Client Protocol (ACP)
- **Config Sync**: Sync Claude Code, Codex, and ghc-api config files with OneDrive
- **Safe Backups**: Auto backup overwritten config files as `*.YYYYMMDD_HHMMSS.bak`
- **Machine Token Usage Logs**: Periodic token usage JSONL per machine with cross-machine overview in dashboard
- **Optional User-Token Auth**: Opt-in middleware gates LLM endpoints behind self-signup + admin-approved tokens; requests, stats, and token usage are then grouped per user

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
- `-v` or `--version`: Show version (for example `ghc-api 1.0.15`)
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
```

### Token Management

The application follows this priority for getting the GitHub token:

1. `GITHUB_TOKEN` environment variable
2. Token file at `~/.ghc-api/github_token.txt`
3. Interactive GitHub Device Flow authentication

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

Once enabled, LLM endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/models`, plus their non-`/v1` aliases) require an approved user token. Dashboard and admin endpoints stay open at the Flask layer — they're expected to be gated by a reverse proxy in production (see [Production Deployment](#production-deployment)).

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
- `GET /v1/models` - List available models
- `GET /models` - List available models (without v1 prefix)

### Anthropic Compatible

- `POST /v1/messages` - Messages API (Anthropic format)

### Dashboard & Monitoring

- `GET /` - Web dashboard with statistics
- `GET /requests` - Request browser page
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
| **Public — LLM API** | `POST /v1/chat/completions`, `/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses`, `/responses`, `GET /v1/models`, `/models`, `/v1/models/full/`, `/models/full/` | No basic-auth (clients send `Authorization: Bearer <user-token>`); ghc-api's own middleware checks the user token when `enable_auth=true` |
| **Public — signup** | `GET /signup`, `POST /signup`, `GET /api/users-list` (token-redacted) | No basic-auth — anyone may request an account |
| **Admin — user mgmt** | `GET /api/users`, `POST /api/users/<id>/approve`, `POST /api/users/<id>/revoke`, `DELETE /api/users/<id>` | basic-auth — `GET /api/users` returns plaintext tokens |
| **Admin — config & data** | `POST /api/runtime-config`, `POST /api/config-manager/install-tools`, `POST /api/config-manager/sync-to-onedrive`, `POST /api/config-manager/sync-from-onedrive`, `POST /api/requests/import` | basic-auth — affect global state |
| **Admin — dashboard & inspection** | `GET /`, `/requests`, `/code-agent-manager`, `/chat`, `/agent`, all `GET /api/stats`, `/api/requests*`, `/api/request/<id>*`, `/api/config-manager/*`, `/api/agent/*` | basic-auth — request bodies expose other users' prompts |

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
