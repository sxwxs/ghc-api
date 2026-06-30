# fix_token_usage

One-shot recovery script for `token_usage.jl`.

## Problem

The token-usage reporter previously emitted per-tick rows that omitted
`cache_creation_input_tokens` and `cache_read_input_tokens`. On Anthropic
endpoints, `usage.input_tokens` only counts the *uncached-new* slice of the
prompt — when prompt caching is active, the cached portions live in the two
cache fields. Dropping them made the dashboard's "Total Token" column
under-report by orders of magnitude on any traffic that hit the cache.

## Recovery source

`cache.complete_request` always wrote the full per-request breakdown
(including cache fields) to `<config_dir>/requests/YYYY-MM-DD.jl`. The
script re-aggregates those files into `token_usage.jl`, matching the
reporter's wire format and 5-minute bucket cadence.

## Usage

```bash
# Preview without writing anything
python rebuild_token_usage.py --dry-run

# Rebuild (server must be stopped first)
python rebuild_token_usage.py --force
```

Defaults follow the same path resolution as `TokenUsageReporter`:

- `--requests-dir`: `%APPDATA%/ghc-api/requests` on Windows,
  `~/.ghc-api/requests` elsewhere
- `--output`: OneDrive agent root's `token_usage.jl` if available,
  otherwise `~/.ghc-api/token_usage.jl`
- `--bucket-seconds`: `300` (matches the reporter)

## Safety

- **Stop the ghc-api server first.** The live reporter holds an in-memory
  snapshot; if it ticks while a rebuilt file is on disk, it will append a
  delta on top and double-count the current session. The script refuses to
  overwrite without `--force`, and prompts for confirmation even then.
- The existing `token_usage.jl` is copied to `token_usage.jl.bak.<ts>`
  before being overwritten.
- `--dry-run` is always safe and writes nothing.

## Limitations

- Old request lines that pre-date cache-field tracking simply contribute
  zeros to the cache columns — there's no way to reconstruct what isn't on
  disk. New entries (post-fix) and lines from `ghc_api.cache` after the
  cache fields were added will rebuild accurately.
- The script only knows about request files on the local machine. Other
  machines' `token_usage.jl` files in the OneDrive `agents/` tree are
  untouched; run the script once per machine if you want them all fixed.
