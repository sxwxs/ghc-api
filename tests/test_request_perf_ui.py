"""Run the actual dashboard scripts with a fake DOM and local-job API."""

import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

import pytest

from ghc_api.app import create_app
from ghc_api.request_file_stats import _extract_index_row, _new_quality
from ghc_api.request_perf import build_performance_cube


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is required for dashboard UI checks")
def test_performance_ui_and_math():
    root = Path(__file__).resolve().parent.parent
    with create_app().test_client() as client:
        response = client.get("/request-perf")
    assert response.status_code == 200
    rows = []
    for day in (1, 20, 28, 31):
        for model in ("measured", 'legacy <img src=x onerror="alert(1)">', "__proto__"):
            for i in range(24):
                output = 100 + i * 20
                entry = {
                    "timestamp": datetime(2026, 7, day, 12).timestamp(), "model": model,
                    "status_code": 200, "state": "completed", "duration": 2 + output / (10 if day >= 28 else 20),
                    "ttft": 2 if model == "measured" else None,
                    "output_tokens": output, "input_tokens": [1000, 8000, 200000][i % 3],
                    "upstream_profile": '<svg onload="alert(2)">', "endpoint": "/v1/messages",
                }
                rows.append(_extract_index_row(entry, f"2026-07-{day:02d}.jl", 0, 1, b"\n"))
    cube = build_performance_cube(iter(rows))
    result = {
        "kind": "performance", "generated_at": "2026-08-01T00:00:00Z", "files": ["test"],
        "quality": {**_new_quality(), "invalid_json_lines": 2, "records_indexed": len(rows)},
        "warnings": [], "skipped_files": [], "perf": cube,
    }
    payload = {
        "html": response.get_data(as_text=True), "result": result,
        "math": (root / "ghc_api/static/request_perf_math.js").read_text(encoding="utf-8"),
        "script": (root / "ghc_api/static/request_perf.js").read_text(encoding="utf-8"),
    }
    execution = subprocess.run(
        ["node", str(root / "tests/fixtures/request_perf_ui.js")],
        input=json.dumps(payload), text=True, encoding="utf-8", capture_output=True, timeout=30,
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
