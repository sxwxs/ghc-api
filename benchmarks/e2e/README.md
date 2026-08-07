# ghc-api E2E performance benchmark

This suite starts three real HTTP participants:

```text
benchmark client -> fake backend
benchmark client -> ghc-api -> fake backend
```

The fake backend implements deterministic, synthetic response structures for:

- `fake-opus`: Anthropic `/v1/messages` and OpenAI `/v1/responses`
- `fake-gpt`: OpenAI `/v1/responses`

It does not read or echo local request dumps. All IDs, text, paths, URLs, token
counts, encrypted-content placeholders, and usage fields are synthetic.

## Run

```bash
python -m benchmarks.e2e.runner --suite smoke
python -m benchmarks.e2e.runner --suite full
```

Compare with a locally built `puxu-msft/copilot-api-js` checkout:

```bash
python -m benchmarks.e2e.compare_copilot_api_js \
  --js-repo build/copilot-api-js \
  --output benchmarks/results/copilot-api-js-comparison \
  --requests 60 --warmup 5 --trials 3
```

Use `--wait-for-idle` to require a stable low-CPU preflight and
`--js-runtime <path>` when Node is not on `PATH`.

Results are written under `benchmarks/results/<UTC timestamp>/` as JSON and
Markdown. Generated results and the local `opus.jl`/`gpt.jl` dumps are ignored
by Git.

The current CLI uses Flask's threaded development server, so that server is
part of the measured stack. Results are best used to compare variants or code
revisions on the same otherwise-idle machine.
