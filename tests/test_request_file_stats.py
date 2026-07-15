import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ghc_api import request_file_stats as stats


class RequestFileStatsTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temporary_directory.name)
        self.requests_dir = self.config_dir / "requests"
        self.requests_dir.mkdir()
        self.config_patch = mock.patch.object(stats, "get_config_dir", return_value=str(self.config_dir))
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.temporary_directory.cleanup()

    def write_request_file(self, filename, records, final_newline=True):
        path = self.requests_dir / filename
        with path.open("wb") as file:
            for index, record in enumerate(records):
                raw = record if isinstance(record, bytes) else json.dumps(record, ensure_ascii=False).encode("utf-8")
                file.write(raw)
                if final_newline or index < len(records) - 1:
                    file.write(b"\n")
        return path


class RequestFileListingTests(RequestFileStatsTestCase):
    def test_old_json_index_is_ignored(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "a"}])
        index_dir = self.requests_dir / stats.INDEX_DIR_NAME
        index_dir.mkdir()
        (index_dir / f"{path.name}.json").write_text("{}", encoding="utf-8")
        files = stats.list_request_files()
        self.assertEqual(files[0]["index_status"], "missing")


class RequestIndexExtractionTests(RequestFileStatsTestCase):
    def test_bucket_boundaries(self):
        self.assertEqual(stats._bucket_index("request_size", 511), 1)
        self.assertEqual(stats._bucket_index("request_size", 512), 2)
        self.assertEqual(stats._bucket_index("duration_ms", 600000), 11)
        self.assertEqual(stats._bucket_index("duration_ms", 600001), 12)

    def test_sidecar_records_offsets_hashes_fields_and_buckets(self):
        records = [
            {
                "id": "req-a",
                "timestamp": "2026-07-14T00:00:00Z",
                "model": "alias",
                "translated_model": "claude-sonnet",
                "endpoint": "/v1/messages",
                "status_code": 504,
                "duration": 601,
                "request_size": 1024,
                "response_size": 2048,
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 40,
                "user_id": "u",
                "client_ip": "127.0.0.2",
            },
            {"id": "req-b", "model": "gpt", "status_code": 200, "duration": 0.2, "request_size": 5},
        ]
        path = self.write_request_file("2026-07-14.jl", records)

        meta, mode, warnings = stats.build_or_load_file_index(path)
        rows = stats.load_indexed_rows([path.name])

        self.assertEqual(mode, "rebuild")
        self.assertEqual(warnings, [])
        self.assertEqual(meta["row_count"], 2)
        first_raw = path.read_bytes().splitlines(keepends=True)[0]
        first = next(row for row in rows if row["id"] == "req-a")
        self.assertEqual(first["offset"], 0)
        self.assertEqual(first["length"], len(first_raw))
        self.assertEqual(first["line_sha256"], hashlib.sha256(first_raw).hexdigest())
        self.assertEqual(first["effective_model"], "claude-sonnet")
        self.assertEqual(first["status_code"], 504)
        self.assertEqual(first["duration_ms"], 601000)
        self.assertEqual(first[stats.TOTAL_BILLED], 100)
        self.assertEqual(first["duration_ms_bucket"], stats._bucket_index("duration_ms", 601000))

    def test_missing_token_components_count_as_zero_when_present_fields_are_valid(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "x", "input_tokens": 3, "output_tokens": 2}])
        stats.build_or_load_file_index(path)
        row = stats.load_indexed_rows([path.name])[0]
        self.assertEqual(row[stats.TOTAL_BILLED], 5)
        self.assertIsNone(row[stats.CACHE_WRITE])

    def test_invalid_token_component_invalidates_total(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "x", "input_tokens": 3, "cache_read_input_tokens": -1}])
        stats.build_or_load_file_index(path)
        row = stats.load_indexed_rows([path.name])[0]
        self.assertIsNone(row[stats.TOTAL_BILLED])


class RequestIndexLifecycleTests(RequestFileStatsTestCase):
    def test_unchanged_file_reuses_sidecar_without_opening_source(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "a", "request_size": 1}])
        first, mode, _ = stats.build_or_load_file_index(path)
        self.assertEqual(mode, "rebuild")
        original_open = Path.open

        def guarded(candidate, *args, **kwargs):
            if Path(candidate) == path:
                raise AssertionError("cached source must not be opened")
            return original_open(candidate, *args, **kwargs)

        with mock.patch.object(Path, "open", new=guarded):
            second, second_mode, _ = stats.build_or_load_file_index(path)
        self.assertEqual(second_mode, "cached")
        self.assertEqual(second["row_count"], first["row_count"])

    def test_append_is_incremental_without_duplicates(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "a", "request_size": 1}])
        first, _, _ = stats.build_or_load_file_index(path)
        first_offset = stats.load_indexed_rows([path.name])[0]["offset"]
        with path.open("ab") as file:
            file.write(json.dumps({"id": "b", "request_size": 2}).encode() + b"\n")

        second, mode, _ = stats.build_or_load_file_index(path)
        rows = stats.load_indexed_rows([path.name])

        self.assertEqual(mode, "incremental")
        self.assertEqual(second["row_count"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(next(row for row in rows if row["id"] == "a")["offset"], first_offset)

    def test_partial_tail_waits_for_newline(self):
        raw = json.dumps({"id": "a", "request_size": 7}).encode()
        path = self.write_request_file("2026-07-14.jl", [raw], final_newline=False)
        first, _, _ = stats.build_or_load_file_index(path)
        self.assertEqual(first["row_count"], 0)
        self.assertEqual(first["processed_bytes"], 0)
        with path.open("ab") as file:
            file.write(b"\n")
        second, mode, _ = stats.build_or_load_file_index(path)
        self.assertEqual(mode, "incremental")
        self.assertEqual(second["row_count"], 1)

    def test_source_replacement_rebuilds_sidecar(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "a", "request_size": 1}])
        stats.build_or_load_file_index(path)
        original = path.read_bytes()
        replacement = original.replace(b'"request_size": 1', b'"request_size": 9')
        self.assertEqual(len(original), len(replacement))
        path.write_bytes(replacement)
        meta, mode, warnings = stats.build_or_load_file_index(path)
        self.assertEqual(mode, "rebuild")
        self.assertTrue(any(warning["code"] == "source_changed" for warning in warnings))
        self.assertEqual(meta["row_count"], 1)
        self.assertEqual(stats.load_indexed_rows([path.name])[0]["request_size"], 9)

    def test_cancel_does_not_replace_existing_sidecar(self):
        path = self.write_request_file("2026-07-14.jl", [{"id": "a"}])
        stats.build_or_load_file_index(path)
        before = stats._index_path_for(path).read_bytes()
        with path.open("ab") as file:
            file.write(json.dumps({"id": "b"}).encode() + b"\n")
        event = threading.Event(); event.set()
        with self.assertRaises(stats.RequestStatsCancelled):
            stats.build_or_load_file_index(path, cancel_event=event)
        self.assertEqual(stats._index_path_for(path).read_bytes(), before)


class RequestDatasetTests(RequestFileStatsTestCase):
    def wait(self, manager, job_id):
        for _ in range(300):
            job = manager.get(job_id)
            if job["status"] in manager.TERMINAL_STATUSES:
                return job
            threading.Event().wait(0.01)
        self.fail("job did not finish")

    def test_dataset_aggregates_model_status_and_bucket_requests(self):
        path = self.write_request_file("2026-07-14.jl", [
            {"id": "short", "timestamp": 1, "model": "m", "status_code": 200, "duration": 1, "request_size": 1, "input_tokens": 1},
            {"id": "long", "timestamp": 2, "model": "m", "status_code": 504, "duration": 700, "request_size": 2, "input_tokens": 2},
        ])
        manager = stats.RequestStatsJobManager()
        job, _ = manager.start([path.name])
        completed = self.wait(manager, job["id"])

        self.assertEqual(completed["status"], "completed")
        result = completed["result"]
        self.assertEqual(result["overall"]["request_count"], 2)
        self.assertEqual(result["by_model"]["m"]["request_count"], 2)
        self.assertEqual(result["by_status_code"]["504"]["request_count"], 1)
        bucket = stats._bucket_index("duration_ms", 700000)
        page = manager.query_dataset(result["dataset_id"], metric="duration_ms", bucket=bucket, view="overall", value=None, page=1, per_page=50)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], "long")
        self.assertIn("/request-file-detail?", page["items"][0]["detail_url"])

    def test_model_overflow_drilldown_matches_other_summary(self):
        rows = []
        with mock.patch.object(stats, "MAX_MODELS_PER_DATASET", 2):
            for index, model in enumerate(("a", "b", "c")):
                rows.append(stats._extract_index_row(
                    {"id": model, "timestamp": index, "model": model, "duration": 1},
                    "2026-07-14.jl", index * 10, 10, (model + "\n").encode(),
                ))
            dataset = stats.RequestStatsDataset("dataset", ["2026-07-14.jl"], rows, 1000)
            result = dataset.build_result([], stats._new_quality())
            bucket = stats._bucket_index("duration_ms", 1000)
            page = dataset.query_bucket("duration_ms", bucket, "model", stats.OTHER_MODEL_KEY, 1, 50)
        self.assertEqual(result["by_model"][stats.OTHER_MODEL_KEY]["request_count"], 1)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], "c")

    def test_read_request_detail_requires_indexed_locator_and_detects_changes(self):
        record = {"id": "detail", "request_body": {"hello": "world"}}
        path = self.write_request_file("2026-07-14.jl", [record])
        stats.build_or_load_file_index(path)
        row = stats.load_indexed_rows([path.name])[0]
        detail = stats.read_request_detail(path.name, row["offset"], row["length"], row["line_sha256"])
        self.assertEqual(detail["id"], "detail")
        with self.assertRaises(stats.RequestStatsValidationError):
            stats.read_request_detail(path.name, row["offset"] + 1, row["length"], row["line_sha256"])
        data = bytearray(path.read_bytes()); data[0] = ord("[")
        path.write_bytes(data)
        with self.assertRaises(stats.RequestFileChangedError):
            stats.read_request_detail(path.name, row["offset"], row["length"], row["line_sha256"])


if __name__ == "__main__":
    unittest.main()
