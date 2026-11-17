"""All three daily logs must append complete records under concurrent writes."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest import mock

import pytest

from ghc_api import anysearch_log, file_logging, webiq_log
from ghc_api.request_log import request_log
from ghc_api.state import state


@pytest.fixture(params=["requests", "webiq", "anysearch"])
def log_writer(request, tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    write, flag = {
        # The request log has no on/off switch; it is always written.
        "requests": (request_log.append, None),
        "webiq": (webiq_log.record_search_to_file, "log_webiq_requests"),
        "anysearch": (anysearch_log.record_search_to_file, "log_anysearch_requests"),
    }[request.param]
    if flag:
        monkeypatch.setattr(state, flag, True)
    return write, flag, tmp_path / request.param


def read_records(directory):
    records = []
    for path in directory.glob("*.jl"):
        with path.open(encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file)
    return records


def test_concurrent_records_remain_whole(log_writer):
    write, _, directory = log_writer
    workers = 8
    ready = threading.Barrier(workers)
    body = '日志🙂"\\\n' * 65536

    @contextmanager
    def split_open(*args, **kwargs):
        # Force writes to span multiple flushes on every OS, including one
        # during close. Locking only write(), not the context exit, is unsafe.
        with open(*args, **kwargs) as file:
            pending = []

            def split_write(line):
                middle = len(line) // 2
                file.write(line[:middle])
                file.flush()
                time.sleep(0.002)
                pending.append(line[middle:])
                return len(line)

            yield mock.Mock(write=split_write)
            file.writelines(pending)

    def write_records(worker):
        ready.wait(timeout=10)
        for number in range(2):
            write({"id": f"{worker}-{number}", "body": body})

    with mock.patch("ghc_api.file_logging.open", side_effect=split_open):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(write_records, range(workers)))

    records = read_records(directory)
    assert len(records) == workers * 2
    assert {record["id"] for record in records} == {
        f"{worker}-{number}" for worker in range(workers) for number in range(2)
    }
    assert all(record["body"] == body for record in records)


def test_disabled_logging_does_not_open_a_file(log_writer, monkeypatch):
    write, flag, directory = log_writer
    if flag is None:
        pytest.skip("the request log cannot be disabled")
    monkeypatch.setattr(state, flag, False)
    with mock.patch("ghc_api.file_logging.open") as open_file:
        write({"id": "disabled"})
        open_file.assert_not_called()
    assert not directory.exists()


@pytest.mark.parametrize("failure_at", ["open", "write", "close"])
def test_disk_failure_is_reported_and_releases_lock(log_writer, failure_at, capsys):
    write, _, directory = log_writer
    error = OSError("private-path-or-key")
    with mock.patch("ghc_api.file_logging.open") as open_file:
        handle = open_file.return_value
        if failure_at == "open":
            open_file.side_effect = error
        elif failure_at == "write":
            handle.__enter__.return_value.write.side_effect = error
        else:
            handle.__exit__.side_effect = error
        write({"id": "failed"})  # logging failures must not break requests
        open_file.assert_called_once()

    output = capsys.readouterr().out
    assert "Failed to append" in output
    if directory.name == "anysearch":
        assert str(error) not in output

    # Fail immediately rather than hanging the next append if the lock leaked.
    assert file_logging._append_lock.acquire(blocking=False)
    file_logging._append_lock.release()
    write({"id": "after-failure"})
    assert read_records(directory) == [{"id": "after-failure"}]
