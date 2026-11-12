"""The request log is the only full-fidelity store of request bodies.

Covers retention configuration (including the deprecated keys), expiry
cleanup, reading records back by locator, full-text search, and refilling the
in-memory request list after a restart.
"""

import json
import time
from datetime import date, datetime, timedelta
from unittest import mock

import pytest
from flask import Flask

from ghc_api.cache import RequestCache
from ghc_api.request_file_stats import _index_path_for, _meta_path_for, build_or_load_file_index
from ghc_api.request_log import (
    DEFAULT_RETENTION_DAYS,
    RequestLog,
    RequestLogError,
    resolve_retention_days,
)
from ghc_api.routes import dashboard as dashboard_routes
from ghc_api.state import state


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(state, "request_log_retention_days", DEFAULT_RETENTION_DAYS)
    return tmp_path / "requests"


def write_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as file:
        for record in records:
            file.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


# ------------------------------------------------------------------ retention


@pytest.mark.parametrize("config, expected, warns", [
    ({}, DEFAULT_RETENTION_DAYS, False),
    ({"request_log_retention_days": 30}, 30, False),
    ({"request_log_retention_days": 0}, 0, False),
    ({"save_request_to_file": True}, 0, True),
    ({"save_request_to_file": False}, DEFAULT_RETENTION_DAYS, True),
    # An explicit retention wins over the legacy key, in both directions.
    ({"save_request_to_file": True, "request_log_retention_days": 3}, 3, True),
    ({"save_request_to_file": False, "request_log_retention_days": 0}, 0, True),
])
def test_retention_resolution(config, expected, warns):
    days, warnings = resolve_retention_days(config)
    assert days == expected
    assert bool(warnings) == warns
    if warns:
        assert "save_request_to_file" in warnings[0]


def test_false_legacy_warning_says_logging_is_now_on(log_dir):
    _, warnings = resolve_retention_days({"save_request_to_file": False})
    assert "no longer turns request logging off" in warnings[0]
    assert str(log_dir) in warnings[0]


@pytest.mark.parametrize("value", [-1, 1.5, "7", True, None])
def test_invalid_retention_is_rejected(value):
    with pytest.raises(ValueError):
        resolve_retention_days({"request_log_retention_days": value})


# -------------------------------------------------------------------- cleanup


def test_cleanup_deletes_only_expired_daily_files_and_their_indexes(log_dir, monkeypatch):
    monkeypatch.setattr(state, "request_log_retention_days", 3)
    today = date(2026, 11, 12)
    names = [f"{(today - timedelta(days=n)).isoformat()}.jl" for n in range(5)]
    for name in names:
        write_lines(log_dir / name, [{"id": name, "timestamp": 1}])
    expired = log_dir / names[4]
    build_or_load_file_index(expired)
    (log_dir / "x.jl").write_text("{}\n", encoding="utf-8")
    (log_dir / "notes.txt").write_text("keep", encoding="utf-8")

    removed = RequestLog().cleanup(today)

    # Retention 3 keeps today and the two days before it.
    assert sorted(removed) == sorted(names[3:])
    assert sorted(p.name for p in log_dir.glob("*.jl")) == sorted(names[:3] + ["x.jl"])
    assert (log_dir / "notes.txt").exists()
    assert not _index_path_for(expired).exists()
    assert not _meta_path_for(expired).exists()


def test_zero_retention_keeps_everything(log_dir, monkeypatch):
    monkeypatch.setattr(state, "request_log_retention_days", 0)
    write_lines(log_dir / "2001-01-01.jl", [{"id": "old"}])
    assert RequestLog().cleanup(date(2026, 11, 12)) == []
    assert (log_dir / "2001-01-01.jl").exists()


def test_cleanup_runs_once_per_day_on_append(log_dir, monkeypatch):
    log = RequestLog()
    with mock.patch.object(log, "cleanup") as cleanup:
        log.append({"id": "a"})
        log.append({"id": "b"})
        assert cleanup.call_count == 1
        log.cleanup_if_new_day(date.today() + timedelta(days=1))
        assert cleanup.call_count == 2


# ----------------------------------------------------------------- read back


def test_append_returns_locator_that_reads_back_the_exact_record(log_dir):
    log = RequestLog()
    first = log.append({"id": "a", "body": "日志 " * 1000})
    second = log.append({"id": "b", "body": {"nested": [1, 2, 3]}})
    assert first["offset"] == 0
    assert second["offset"] == first["length"]
    assert log.read(first) == {"id": "a", "body": "日志 " * 1000}
    assert log.read(second)["body"] == {"nested": [1, 2, 3]}


@pytest.mark.parametrize("locator", [
    {"file": "../config.jl", "offset": 0, "length": 10},
    {"file": "..\\2026-11-12.jl", "offset": 0, "length": 10},
    {"file": "config.yaml", "offset": 0, "length": 10},
    {"file": "2026-11-12.jl", "offset": -1, "length": 10},
    {"file": "2026-11-12.jl", "offset": 0, "length": 0},
    {"file": "2026-11-12.jl", "offset": "0", "length": 10},
])
def test_invalid_locators_are_rejected(log_dir, locator):
    write_lines(log_dir / "2026-11-12.jl", [{"id": "a"}])
    with pytest.raises(RequestLogError):
        RequestLog().read(locator)


def test_locator_into_the_middle_of_a_line_is_rejected(log_dir):
    locator = RequestLog().append({"id": "a", "body": "x" * 100})
    with pytest.raises(RequestLogError):
        RequestLog().read({**locator, "offset": 5, "length": 20})


def test_expired_file_reports_a_clear_error(log_dir):
    locator = RequestLog().append({"id": "a"})
    (log_dir / locator["file"]).unlink()
    with pytest.raises(RequestLogError, match="no longer exists"):
        RequestLog().read(locator)


# --------------------------------------------------------------------- search


def test_search_matches_bodies_newest_first_within_time_range(log_dir):
    now = time.time()
    today = datetime.fromtimestamp(now).date()
    old_day = today - timedelta(days=3)
    write_lines(log_dir / f"{old_day.isoformat()}.jl", [
        {"id": "old", "timestamp": int(now - 3 * 86400), "request_body": {"q": "needle"}},
    ])
    write_lines(log_dir / f"{today.isoformat()}.jl", [
        {"id": "r1", "timestamp": int(now - 60), "request_body": {"q": "Needle in request"}},
        {"id": "meta-only", "timestamp": int(now - 50), "model": "needle-model"},
        {"id": "r2", "timestamp": int(now - 40), "raw_events": ["...needle..."], "user_id": "bob"},
        {"id": "r3", "timestamp": int(now - 30), "response_body": {"text": "中文needle"}},
    ])
    log = RequestLog()

    recent = log.search("NEEDLE", since=now - 86400)
    assert [item["id"] for item in recent] == ["r3", "r2", "r1"]
    assert all("request_body" not in item and item["log"]["file"] for item in recent)
    assert log.read(recent[0]["log"])["response_body"] == {"text": "中文needle"}

    everything = log.search("needle", since=None)
    assert [item["id"] for item in everything] == ["r3", "r2", "r1", "old"]

    assert [item["id"] for item in log.search("needle", since=None, user_id="bob")] == ["r2"]
    assert [item["id"] for item in log.search("中文", since=None)] == ["r3"]


def test_search_matches_literal_json_fragments(log_dir):
    now = int(time.time())
    event = json.dumps({"type": "delta", "content": "hello"}, separators=(",", ":"))
    write_lines(log_dir / f"{date.today().isoformat()}.jl", [
        {"id": "sse", "timestamp": now, "raw_events": [event]},
        {"id": "quoted", "timestamp": now, "request_body": {"text": 'he said "hi"'}},
    ])
    log = RequestLog()
    assert [item["id"] for item in log.search('"content":"hello"', since=None)] == ["sse"]
    assert [item["id"] for item in log.search('said "hi"', since=None)] == ["quoted"]


def test_search_handles_legacy_iso_and_string_timestamps(log_dir):
    write_lines(log_dir / "2026-03-10.jl", [
        {"id": "iso", "timestamp": "2026-03-10T10:20:30", "request_body": "needle"},
        {"id": "numeric-string", "timestamp": "1773000000", "request_body": "needle"},
        {"id": "missing", "request_body": "needle"},
    ])
    hits = RequestLog().search("needle", since=None)
    assert [item["id"] for item in hits] == ["missing", "numeric-string", "iso"]
    assert all(isinstance(item["timestamp"], int) for item in hits)
    # A string timestamp in a file inside the search window is compared, not crashed on.
    recent_iso = datetime.fromtimestamp(time.time() - 60).isoformat()
    write_lines(log_dir / f"{date.today().isoformat()}.jl", [
        {"id": "recent-iso", "timestamp": recent_iso, "request_body": "needle"},
        {"id": "old-iso", "timestamp": "2026-03-10T10:20:30", "request_body": "needle"},
    ])
    hits = RequestLog().search("needle", since=time.time() - 86400)
    assert [item["id"] for item in hits] == ["recent-iso"]


def test_search_ignores_a_partially_written_tail(log_dir):
    path = log_dir / f"{date.today().isoformat()}.jl"
    write_lines(path, [{"id": "done", "timestamp": int(time.time()), "request_body": "needle"}])
    with path.open("ab") as file:
        file.write(b'{"id": "partial", "request_body": "needle"')
    assert [item["id"] for item in RequestLog().search("needle", since=None)] == ["done"]


# -------------------------------------------------------------------- restore


def test_recent_entries_restore_newest_metadata_with_locators(log_dir):
    yesterday = date.today() - timedelta(days=1)
    write_lines(log_dir / f"{yesterday.isoformat()}.jl", [
        {"id": f"y{n}", "timestamp": 1000 + n, "model": "m", "endpoint": "/v1/messages",
         "status_code": 200, "duration": 1.5, "output_tokens": 30, "request_body": "big"}
        for n in range(3)
    ])
    write_lines(log_dir / f"{date.today().isoformat()}.jl", [
        {"id": "t0", "timestamp": 2000, "model": "m", "translated_model": "m2", "state": "error",
         "status_code": 499, "input_tokens": 5, "cache_read_input_tokens": 2, "ttft": 0.25,
         "user_id": "alice", "upstream_provider": "azure", "response_body": "big"},
    ])

    entries = RequestLog().recent_entries(3)

    assert [entry["id"] for entry in entries] == ["y1", "y2", "t0"]
    t0 = entries[-1]
    assert t0["translated_model"] == "m2"
    assert (t0["state"], t0["status_code"], t0["user_id"]) == ("error", 499, "alice")
    assert (t0["input_tokens"], t0["cache_read_input_tokens"], t0["ttft"]) == (5, 2, 0.25)
    assert t0["upstream_provider"] == "azure"
    assert "response_body" not in t0
    assert RequestLog().read(t0["log"])["response_body"] == "big"
    assert entries[0]["duration"] == 1.5


def test_restore_prepends_without_counting_or_duplicating(log_dir):
    write_lines(log_dir / f"{date.today().isoformat()}.jl", [
        {"id": f"old{n}", "timestamp": 1000 + n, "model": "m", "status_code": 200}
        for n in range(5)
    ])
    cache = RequestCache(max_entries=4)
    cache.start_request("live", {"model": "m"})
    cache.complete_request("live", {"status_code": 200})

    # "live" is in the log too; it must not be restored a second time.
    restored = cache.restore(RequestLog().recent_entries(cache.max_entries))

    assert restored == 3
    assert [item["id"] for item in cache.get_recent_requests(10)] == ["live", "old4", "old3", "old2"]
    assert cache.request_count == 1
    assert cache.get_record("old3")["id"] == "old3"


# ------------------------------------------------------------------ dashboard


@pytest.fixture
def client(log_dir):
    app = Flask(__name__)
    app.register_blueprint(dashboard_routes.dashboard_bp)
    cache = RequestCache()
    with mock.patch.object(dashboard_routes, "cache", cache):
        yield app.test_client(), cache


def test_fulltext_search_defaults_to_24_hours_and_details_open_by_locator(client, log_dir):
    test_client, _ = client
    now = time.time()
    write_lines(log_dir / f"{date.today().isoformat()}.jl", [
        {"id": "stale", "timestamp": int(now - 30 * 3600), "request_body": "needle"},
        {"id": "fresh", "timestamp": int(now - 60), "request_body": "needle"},
    ])

    default = test_client.get("/api/requests/search?q=needle").get_json()
    everything = test_client.get("/api/requests/search?q=needle&hours=0").get_json()
    bad = test_client.get("/api/requests/search?q=needle&hours=-1")

    assert default["hours"] == 24
    assert [item["id"] for item in default["items"]] == ["fresh"]
    assert [item["id"] for item in everything["items"]] == ["fresh", "stale"]
    assert bad.status_code == 400

    # "stale" is not in the in-memory list; its locator still opens it.
    hit = everything["items"][1]
    detail = test_client.get(f"/api/request/{hit['id']}", query_string=hit["log"])
    assert detail.status_code == 200
    assert detail.get_json()["request_body"] == "needle"


def test_detail_reports_missing_record_after_retention(client, log_dir):
    test_client, cache = client
    cache.start_request("gone", {"model": "m"})
    cache.complete_request("gone", {"status_code": 200})
    (log_dir / cache.get_request("gone")["log"]["file"]).unlink()

    response = test_client.get("/api/request/gone")

    assert response.status_code == 410
    assert "no longer exists" in response.get_json()["error"]


def test_runtime_config_exposes_and_validates_retention(client, monkeypatch):
    test_client, _ = client
    assert test_client.get("/api/runtime-config").get_json()["request_log_retention_days"] == DEFAULT_RETENTION_DAYS

    ok = test_client.post("/api/runtime-config", json={"request_log_retention_days": 30})
    bad = test_client.post("/api/runtime-config", json={"request_log_retention_days": -1})
    legacy = test_client.post("/api/runtime-config", json={"save_request_to_file": True})

    assert ok.status_code == 200 and state.request_log_retention_days == 30
    assert bad.status_code == 400
    assert legacy.status_code == 400


def test_import_and_export_routes_are_gone(client):
    test_client, _ = client
    assert test_client.get("/api/requests/export").status_code in (404, 405)
    assert test_client.post("/api/requests/import").status_code in (404, 405)
