"""Optional local check that synthetic leaf values were not copied from dumps.

The dump text is read only in memory and no dump value is printed or persisted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from benchmarks.e2e.fake_backend.app import app


def _leaf_strings(value) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_strings(child)
    elif isinstance(value, str):
        yield value


def _synthetic_events():
    client = app.test_client()
    for path, model in (("/v1/messages", "fake-opus"), ("/v1/responses", "fake-gpt")):
        response = client.post(path, json={
            "model": model,
            "stream": True,
            "metadata": {"ghc_benchmark": {"profile": "full", "text_bytes": 1024, "text_chunks": 16}},
        })
        for line in response.get_data(as_text=True).splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


def verify(samples: Iterable[Path]) -> int:
    # Only compare distinctive long generated values; short protocol enums and
    # field names are expected to overlap with real protocol records.
    generated = {
        value for event in _synthetic_events() for value in _leaf_strings(event)
        if len(value) >= 32 and not value.startswith((
            "resp_fake_", "msg_fake_", "rs_fake_", "fc_fake_", "call_fake_",
            "toolu_fake_", "ws_fake_", "response.",
        ))
    }
    overlaps = 0
    for sample in samples:
        dump_text = sample.read_text(encoding="utf-8")
        for value in generated:
            if value in dump_text:
                overlaps += 1
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="+", type=Path)
    args = parser.parse_args()
    overlaps = verify(args.samples)
    if overlaps:
        raise SystemExit(f"FAILED: {overlaps} long synthetic values overlap sample leaf content")
    print("PASS: no long synthetic response value was found in the supplied dumps")


if __name__ == "__main__":
    main()
