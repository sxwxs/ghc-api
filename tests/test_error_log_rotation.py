import json
import os

import pytest

from ghc_api import utils
from ghc_api.state import state


@pytest.fixture
def log_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "get_config_dir", lambda: str(tmp_path))
    original = (state.error_log_max_bytes, state.error_log_backup_count)
    try:
        yield tmp_path
    finally:
        state.error_log_max_bytes, state.error_log_backup_count = original


def test_defaults_keep_fifty_megabytes(log_settings):
    assert state.error_log_max_bytes == 50 * 1024 * 1024
    assert state.error_log_backup_count == 0


def test_rotation_discards_old_file_when_no_backups(log_settings):
    state.error_log_max_bytes = 200
    state.error_log_backup_count = 0
    log_file = log_settings / "error.log"

    for i in range(50):
        utils.log_error_request("/v1/messages", {"i": i}, "boom", 500)

    assert log_file.stat().st_size <= 200
    assert not (log_settings / "error.log.1").exists()
    # The surviving lines are the most recent ones.
    last = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert last["request"] == {"i": 49}


def test_rotation_keeps_configured_backups(log_settings):
    state.error_log_max_bytes = 200
    state.error_log_backup_count = 2
    log_file = log_settings / "error.log"

    for i in range(50):
        utils.log_error_request("/v1/messages", {"i": i}, "boom", 500)

    assert log_file.exists()
    assert (log_settings / "error.log.1").exists()
    assert (log_settings / "error.log.2").exists()
    assert not (log_settings / "error.log.3").exists()


def test_rotation_disabled_when_max_bytes_is_zero(log_settings):
    state.error_log_max_bytes = 0
    log_file = log_settings / "error.log"

    for i in range(20):
        utils.log_error_request("/v1/messages", {"i": i}, "boom", 500)

    assert len(log_file.read_text().strip().splitlines()) == 20


def test_diagnostic_logs_live_in_the_config_dir(log_settings):
    state.error_log_max_bytes = 0
    utils.log_connection_retry("req-1", "/v1/messages", 0, 3, RuntimeError("nope"))
    utils.log_tool_result_cleanup({"request_id": "req-1"})

    assert (log_settings / "connection_retry.jl").exists()
    assert (log_settings / "tool_result_cleanup.jl").exists()
    package_dir = os.path.dirname(os.path.abspath(utils.__file__))
    assert not os.path.exists(os.path.join(package_dir, "connection_retry.jl"))
    assert not os.path.exists(os.path.join(package_dir, "tool_result_cleanup.jl"))
