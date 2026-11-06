"""Historical metrics must be consistent without retaining a drill-down dataset."""

import json
from datetime import datetime
from pathlib import Path
import threading
import time
from unittest import mock

import pytest

from ghc_api import request_file_stats as stats
from ghc_api import request_perf as perf


DAY = "2026-07-14.jl"


def indexed(**fields):
    entry = {
        "timestamp": datetime(2026, 7, 14, 12).timestamp(), "model": "model-a",
        "status_code": 200, "state": "completed", "duration": 10,
        "output_tokens": 100, "input_tokens": 20,
        **fields,
    }
    raw = json.dumps(entry).encode() + b"\n"
    return stats._extract_index_row(entry, DAY, 0, len(raw), raw)


def test_index_extracts_new_scalars_and_no_bodies():
    row = indexed(ttft=1.234, upstream_provider="proxy", upstream_profile="private",
                  upstream_api="messages", request_body={"secret": "body"})
    assert row["ttft_ms"] == 1234
    assert row["state"] == "completed"
    assert row["upstream_provider"] == "proxy"
    assert row["upstream_profile"] == "private"
    assert row["upstream_api"] == "messages"
    assert "secret" not in json.dumps(row)


@pytest.mark.parametrize("value,expected", [
    (None, None), (0, 0), (False, None), (-1, None), ("bad", None),
    (float("inf"), None), (float("nan"), None), (1e308, None), (10**400, None), ("1.25", 1250),
])
def test_timing_validation(value, expected):
    row = indexed(duration=value, ttft=value)
    assert row["duration_ms"] == row["ttft_ms"] == expected


@pytest.mark.parametrize("fields,expected", [
    ({"input_tokens": 8000}, 8000),  # omitted optional cache fields
    ({"input_tokens": 8000, "cache_read_input_tokens": None}, 8000),
    ({"input_tokens": 0}, 0),
    ({"input_tokens": None}, None),
    ({"input_tokens": 8000, "cache_read_input_tokens": -1}, None),
    ({"input_tokens": -1, "cache_read_input_tokens": 100}, None),
    ({"cache_creation_input_tokens": "invalid"}, None),
    ({"cache_read_input_tokens": False}, None),
    ({"input_tokens": stats.MAX_METRIC_VALUE, "cache_read_input_tokens": 1}, None),
    ({"output_tokens": -1}, 20),  # invalid output does not invalidate context
])
def test_index_validates_context_without_confusing_invalid_and_missing(fields, expected):
    assert indexed(**fields)["context_tokens"] == expected


def test_invalid_context_is_excluded_from_all_context_views_not_speed_metrics():
    cube = perf.build_performance_cube([
        indexed(input_tokens=8000, cache_read_input_tokens=2000, ttft=2),
        indexed(input_tokens=8000, cache_read_input_tokens=-1, ttft=2),
        indexed(input_tokens=-1, cache_read_input_tokens=2000),
        indexed(input_tokens=None),
    ])
    cell = cube["cells"][0]
    assert cell["context_count"] == 1
    assert cell["context_sum"] == 10000
    assert cell["input"] == 8000
    assert cell["cache_read"] == 2000
    assert cell["cache_write"] == 0
    assert [sample[2] for sample in cell["samples"]] == [10000, None, None, None]
    assert sum(group["count"] for group in cube["contexts"]) == 1
    assert cube["quality"]["missing_context"] == 3
    assert cell["count"] == cell["gen_count"] == 4
    assert cell["gen_output"] == 400
    assert cell["gen_duration"] == 40
    assert cell["ttft_count"] == 2
    for dimension in ("hourly", "upstream", "endpoints"):
        assert sum(group["input"] + group["cache_read"] + group["cache_write"]
                   for group in cube[dimension]) == 10000


def test_disjoint_failures_and_speed_denominators():
    rows = [
        indexed(ttft=2, cache_read_input_tokens=30, cache_creation_input_tokens=40),
        indexed(duration=20, output_tokens=300),
        indexed(duration=1, output_tokens=0, ttft=.5),
        indexed(status_code=500, state="error", output_tokens=900, duration=4, ttft=1),
        indexed(status_code=499, state="error", output_tokens=1000, duration=3),
        indexed(status_code=200, state="cancelled", duration=2),
        indexed(status_code=None, duration=None, output_tokens=None, input_tokens=None),
        indexed(status_code=200, state="error", duration=5),
    ]
    cube = perf.build_performance_cube(iter(rows))
    cell = cube["cells"][0]
    assert cube["total_rows"] == cell["count"] == 8
    assert cell["errors"] == 2
    assert cell["cancelled"] == 2
    assert cell["unknown_status"] == 1
    assert cell["gen_count"] == 2
    assert cell["gen_output"] == 400
    assert cell["gen_duration"] == 30
    assert cell["ttft_count"] == 2
    assert cell["ttft_sum"] == 2.5
    assert cell["decode_count"] == 1
    assert cell["decode_output"] == 100
    assert cell["decode_duration"] == 8
    assert cell["context_count"] == 2
    assert cell["context_sum"] == 110
    assert cell["duration_count"] == sum(cell["duration_hist"]) == 7
    assert sum(cell["tps_hist"]) == 2
    assert cube["quality"]["missing_duration"] == 1
    assert cube["quality"]["missing_output"] == 1
    assert cube["quality"]["missing_context"] == 1


def test_zero_ttft_is_measured_but_missing_and_impossible_ttft_are_not():
    cube = perf.build_performance_cube([
        indexed(ttft=0), indexed(ttft=None), indexed(ttft=11),
        indexed(ttft=10), indexed(ttft=1, duration=None),
    ])
    cell = cube["cells"][0]
    assert cell["ttft_count"] == 2
    assert cell["ttft_sum"] == 10
    assert cell["decode_count"] == 1
    assert cell["decode_duration"] == 10
    assert cube["quality"]["invalid_ttft"] == 2


def test_unknown_status_and_zero_duration_never_generate_speed_samples():
    cube = perf.build_performance_cube([
        indexed(status_code=None), indexed(status_code=0), indexed(status_code=302),
        indexed(duration=0), indexed(output_tokens=0),
    ])
    cell = cube["cells"][0]
    assert cell["gen_count"] == 0
    assert cell["samples"] == []
    assert cell["unknown_status"] == 2


def test_dimensions_keep_models_dates_hours_and_context_boundaries():
    cube = perf.build_performance_cube([
        indexed(translated_model="translated", input_tokens=7999, upstream_provider="provider", endpoint="/a"),
        indexed(input_tokens=8000, upstream_profile="profile", upstream_provider="provider", endpoint="/b"),
        indexed(input_tokens=200000),
        indexed(timestamp=None, input_tokens=None),
        indexed(timestamp=10**30),
    ])
    assert cube["dates"] == ["2026-07-14"]
    assert cube["models"] == ["model-a", "translated"]
    assert {r["hour"] for r in cube["hourly"]} == {12}
    assert sum(r["count"] for r in cube["hourly"]) == 3
    assert {r["bucket"] for r in cube["contexts"]} == {0, 1, 5}
    assert {r["name"] for r in cube["upstream"]} == {"provider", "profile", "(unlogged)"}
    assert {r["name"] for r in cube["endpoints"]} == {"/a", "/b", "(unknown)"}
    assert cube["quality"]["date_fallback"] == 2


def test_histograms_report_clipped_values_without_clamping_totals():
    cube = perf.build_performance_cube([
        indexed(duration=.001, output_tokens=1000),
        indexed(duration=5000, output_tokens=1),
    ])
    assert cube["quality"]["duration_hist_clipped"] == 2
    assert cube["quality"]["tps_hist_clipped"] == 2
    assert cube["cells"][0]["duration"] == pytest.approx(5000.001)


def test_reservoir_is_bounded_and_repeatable_and_models_are_capped():
    def rows():
        for i in range(1000):
            yield indexed(duration=i + 1, output_tokens=10 + i)
        yield indexed(model="b")
        yield indexed(model="c")
    a = perf.build_performance_cube(rows(), max_models=2)
    b = perf.build_performance_cube(rows(), max_models=2)
    assert a == b
    cell = next(cell for cell in a["cells"] if cell["model"] == "model-a")
    assert len(cell["samples"]) == perf.SAMPLES_PER_CELL
    assert cell["gen_count"] == 1000
    assert set(a["models"]) == {"model-a", "b", "__other__"}
    assert a["model_overflow"] is True
    assert perf.build_performance_cube(iter([]))["cells"] == []
    json.dumps(a, allow_nan=False)


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "requests" / DAY
    path.parent.mkdir()
    records = [dict(model="a", duration=10, input_tokens=8000, output_tokens=100, status_code=200, ttft=2)] * 30
    path.write_text("".join(json.dumps(row) + "\n" for row in records) + "broken JSON\n", encoding="utf-8")
    return path


def wait(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] in manager.TERMINAL_STATUSES:
            return job
        time.sleep(.01)
    pytest.fail("job did not finish")


def test_performance_job_bypasses_dataset_limits_and_reuses_index(log_file, monkeypatch):
    monkeypatch.setattr(stats, "MAX_DATASET_ROWS", 1)
    monkeypatch.setattr(stats, "MAX_DATASET_ESTIMATED_BYTES", 1)
    monkeypatch.setattr(stats, "_load_dataset", mock.Mock(side_effect=AssertionError("must not create a dataset")))
    manager = stats.RequestStatsJobManager()
    first, _ = manager.start([log_file.name], kind="performance")
    completed = wait(manager, first["id"])
    assert completed["status"] == "completed", completed["error"]
    result = completed["result"]
    assert result["perf"]["total_rows"] == 30
    assert result["quality"]["invalid_json_lines"] == 1
    assert "dataset_id" not in result
    assert not manager.datasets._datasets
    assert completed["rebuild_scans"] == 1

    original_open = Path.open
    def no_source_open(path, *args, **kwargs):
        assert path != log_file, "cached performance analysis must only read sidecars"
        return original_open(path, *args, **kwargs)
    with mock.patch.object(Path, "open", new=no_source_open):
        second, _ = manager.start([log_file.name], kind="performance")
        cached = wait(manager, second["id"])
    assert cached["status"] == "completed", cached["error"]
    assert cached["cache_hits"] == 1
    assert cached["result"]["perf"] == result["perf"]


@pytest.mark.parametrize("old_version", [3, 4])
def test_schema_upgrade_rebuilds_old_sidecar(log_file, old_version):
    stats.build_or_load_file_index(log_file)
    path = stats._meta_path_for(log_file)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["schema_version"] = old_version
    path.write_text(json.dumps(meta), encoding="utf-8")
    _, mode, warnings = stats.build_or_load_file_index(log_file)
    assert mode == "rebuild"
    assert any(w["code"] == "index_rebuilt" for w in warnings)
    row = stats.load_indexed_rows([log_file.name])[0]
    assert row["ttft_ms"] == 2000
    assert row["context_tokens"] == 8000


def test_cancel_during_performance_aggregation(log_file):
    entered, release = threading.Event(), threading.Event()
    real_build = perf.build_performance_cube
    def slow_aggregate(rows):
        next(rows)
        entered.set()
        assert release.wait(5)
        return real_build(rows)
    manager = stats.RequestStatsJobManager()
    with mock.patch.object(stats, "build_performance_cube", side_effect=slow_aggregate):
        job, _ = manager.start([log_file.name], kind="performance")
        try:
            assert entered.wait(5)
            assert manager.get(job["id"])["current_mode"] == "aggregating"
            manager.cancel(job["id"])
        finally:
            release.set()
        completed = wait(manager, job["id"])
    assert completed["status"] == "cancelled"
    assert completed["result"] is None
    assert not manager.datasets._datasets
