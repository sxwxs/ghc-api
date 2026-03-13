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
- **Config Sync**: Sync Claude Code, Codex, and ghc-api config files with OneDrive
- **Safe Backups**: Auto backup overwritten config files as `*.YYYYMMDD_HHMMSS.bak`
- **Machine Token Usage Logs**: Periodic token usage JSONL per machine with cross-machine overview in dashboard

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
- `-v` or `--version`: Show version (for example `ghc-api 1.0.8`)
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

# Content Filtering
system_prompt_remove: []    # Strings to remove from system prompts
system_prompt_add: []       # Strings to append to system prompts
tool_result_suffix_remove: [] # Strings to remove from end of tool results

# Optional request persistence
save_request_to_file: false # If true, save completed requests to requests/YYYY-MM-DD.jl

# Optional OneDrive access gate
disable_onedrive_access: true # If true, skip all OneDrive detection/sync/shared reads
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

## Architecture

- **Modular Design**: Organized into separate modules for maintainability
  - `main.py` - Entry point and configuration loading
  - `app.py` - Flask application factory
  - `config.py` - Configuration constants and model mappings
  - `cache.py` - Request caching and statistics
  - `translator.py` - OpenAI/Anthropic format translation
  - `streaming.py` - Streaming response handling
  - `token_manager.py` - GitHub token management
  - `routes/` - API endpoint handlers (openai, anthropic, dashboard)
- **Thread-Safe Caching**: Uses threading locks for concurrent access
- **Memory-Based Storage**: No external database dependencies
- **RESTful API Design**: Follows REST conventions

## License

MIT License
