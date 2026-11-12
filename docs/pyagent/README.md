# pyagent

A lightweight Python coding agent, modeled after [pi](https://github.com/earendil-works/pi), living in
`ghc_api/pyagent/`. Every user interface drives it through one JSON-RPC protocol; ghc-api's `/agent` page is one such
interface. Design: [design.md](design.md); protocol reference: [protocol.md](protocol.md).

It started as a separate repository (commit 3dd6e04, now archived); all changes happen here. It stays self-contained:
nothing in `ghc_api/pyagent/` imports the rest of `ghc_api` (`tests/pyagent/test_independence.py` checks this).

Status: M2. One-shot runs, a REPL, saved sessions, and the protocol server on stdio.

## Command line

```bash
export DEEPSEEK_API_KEY=...                     # or ANTHROPIC_API_KEY, OPENAI_API_KEY, MOONSHOT_API_KEY, OPENROUTER_API_KEY
python -m ghc_api.pyagent                       # REPL; /help lists commands, Ctrl+C aborts a reply
python -m ghc_api.pyagent "list the python files and summarize them"
python -m ghc_api.pyagent -c "now add tests"    # continue the latest session in this directory
python -m ghc_api.pyagent --session 8d0723428ae1
python -m ghc_api.pyagent --provider anthropic --thinking high "fix the failing test"
python -m ghc_api.pyagent --mode json "say hi"  # one event envelope per line
python -m ghc_api.pyagent --mode rpc            # serve the protocol on stdin/stdout
```

Sessions are saved under `~/.pyagent/sessions/<working directory>/`; `--no-session` keeps one in memory only.
Configuration lives in `~/.pyagent/config.json` (`PYAGENT_HOME` changes the directory); see the design doc. A local
keyless Responses API endpoint, for example:

```json
{
  "defaultProvider": "local",
  "defaultModel": "gpt-6.1-sol",
  "providers": {
    "local": {
      "api": "openai_responses",
      "baseUrl": "http://127.0.0.1:8313",
      "models": { "gpt-6.1-sol": { "contextWindow": 922000, "maxTokens": 128000, "reasoning": true, "images": true } }
    }
  }
}
```

The `/agent` page does not read this file: ghc-api runs pyagent in its own process with a fixed home,
`<ghc-api config dir>/pyagent`, and a built-in `ghc-api` provider that points back at ghc-api itself.

`AGENTS.md` (or `CLAUDE.md`) files in the pyagent home and in every directory from the root to the working directory
are added to the system prompt.

From Python, drive a server in a child process:

```python
from ghc_api.pyagent.client.transports import ChildProcess

child = await ChildProcess.start()
child.client.on_event(lambda envelope: print(envelope.event.type))
await child.client.initialize("my-app")
state = await child.client.new_session(cwd="/repo")
await child.client.prompt(state.session_id, "run the tests")
# ... wait for a run_end event ...
await child.close()
```

## Development

```bash
ruff check ghc_api/pyagent tests/pyagent && ruff format --check ghc_api/pyagent tests/pyagent
pyright                     # strict on ghc_api/pyagent; see pyrightconfig.json
python -m pytest tests/pyagent
```
