import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ghc_api.app import create_app
from ghc_api import request_file_stats as stats
from ghc_api.routes import dashboard as dashboard_routes


class RequestStatsRoutesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temporary_directory.name)
        self.requests_dir = self.config_dir / "requests"
        self.requests_dir.mkdir()
        self.config_patch = mock.patch.object(stats, "get_config_dir", return_value=str(self.config_dir))
        self.config_patch.start()
        self.manager = stats.RequestStatsJobManager()
        self.manager_patch = mock.patch.object(dashboard_routes, "request_stats_jobs", self.manager)
        self.manager_patch.start()
        self.app = create_app(); self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.manager_patch.stop(); self.config_patch.stop(); self.temporary_directory.cleanup()

    def write_file(self, filename="2026-07-14.jl", records=None):
        records = records or [{
            "id": "route-request", "timestamp": 1, "model": "claude-sonnet", "endpoint": "/v1/messages",
            "status_code": 504, "duration": 700, "request_size": 1024, "response_size": 2048,
            "input_tokens": 10, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30,
            "output_tokens": 40, "request_body": {"hello": "world"},
        }]
        path = self.requests_dir / filename
        with path.open("w", encoding="utf-8") as file:
            for record in records: file.write(json.dumps(record) + "\n")
        return path

    def wait_for_terminal(self, job_id):
        for _ in range(300):
            response = self.client.get(f"/api/request-stats/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            job = response.get_json()["job"]
            if job["status"] in self.manager.TERMINAL_STATUSES: return job
            threading.Event().wait(0.01)
        self.fail("job did not finish")

    def create_completed_job(self):
        response = self.client.post("/api/request-stats/jobs", json={"files": ["2026-07-14.jl"]})
        self.assertEqual(response.status_code, 202)
        return self.wait_for_terminal(response.get_json()["job"]["id"])

    def test_pages_are_served(self):
        self.assertIn(b"Request File Statistics", self.client.get("/request-stats").data)
        self.assertIn(b"Indexed request", self.client.get("/request-file-detail").data)

    def test_file_list_does_not_scan_source_contents(self):
        path = self.write_file()
        original_open = Path.open
        def guarded(candidate, *args, **kwargs):
            if Path(candidate) == path: raise AssertionError("listing must not open source")
            return original_open(candidate, *args, **kwargs)
        with mock.patch.object(Path, "open", new=guarded):
            response = self.client.get("/api/request-stats/files")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["files"][0]["index_status"], "missing")

    def test_job_result_bucket_list_and_detail_round_trip(self):
        self.write_file()
        job = self.create_completed_job()
        self.assertEqual(job["status"], "completed")
        result = job["result"]
        self.assertEqual(result["by_status_code"]["504"]["request_count"], 1)
        bucket = stats._bucket_index("duration_ms", 700000)
        response = self.client.get(
            f"/api/request-stats/datasets/{result['dataset_id']}/requests",
            query_string={"metric": "duration_ms", "bucket": bucket, "view": "status_code", "value": "504"},
        )
        self.assertEqual(response.status_code, 200)
        item = response.get_json()["items"][0]
        detail = self.client.get("/api/request-stats/request-detail", query_string={
            "file": item["source_file"], "offset": item["offset"], "length": item["length"], "sha256": item["line_sha256"],
        })
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["request_body"], {"hello": "world"})

    def test_dataset_query_validation_and_expiry(self):
        response = self.client.get("/api/request-stats/datasets/missing/requests", query_string={"metric": "duration_ms", "bucket": 1})
        self.assertEqual(response.status_code, 410)
        self.write_file(); job = self.create_completed_job(); dataset_id = job["result"]["dataset_id"]
        bad = self.client.get(f"/api/request-stats/datasets/{dataset_id}/requests", query_string={"metric": "bad", "bucket": 0})
        self.assertEqual(bad.status_code, 400)

    def test_unindexed_existing_file_detail_returns_conflict_not_500(self):
        self.write_file()
        response = self.client.get("/api/request-stats/request-detail", query_string={
            "file": "2026-07-14.jl", "offset": 0, "length": 1, "sha256": "0" * 64,
        })
        self.assertEqual(response.status_code, 409)

    def test_detail_rejects_arbitrary_locator_and_path_traversal(self):
        self.write_file(); self.create_completed_job()
        bad = self.client.get("/api/request-stats/request-detail", query_string={"file": "../secret.jl", "offset": 0, "length": 1, "sha256": "0" * 64})
        self.assertEqual(bad.status_code, 400)
        arbitrary = self.client.get("/api/request-stats/request-detail", query_string={"file": "2026-07-14.jl", "offset": 1, "length": 1, "sha256": "0" * 64})
        self.assertEqual(arbitrary.status_code, 400)

    def test_create_job_validates_selection(self):
        self.assertEqual(self.client.post("/api/request-stats/jobs", data="bad", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/api/request-stats/jobs", json={"files": []}).status_code, 400)

    def test_cancel_and_busy_routes(self):
        self.write_file(); self.write_file("2026-07-13.jl")
        release = threading.Event(); real_build = stats.build_or_load_file_index
        def slow_build(path, **kwargs):
            release.wait(2); return real_build(path, **kwargs)
        with mock.patch.object(stats, "build_or_load_file_index", side_effect=slow_build):
            first = self.client.post("/api/request-stats/jobs", json={"files": ["2026-07-14.jl"]})
            active_id = first.get_json()["job"]["id"]
            second = self.client.post("/api/request-stats/jobs", json={"files": ["2026-07-13.jl"]})
            self.assertEqual(second.status_code, 409)
            cancel = self.client.post(f"/api/request-stats/jobs/{active_id}/cancel")
            self.assertEqual(cancel.status_code, 200)
            release.set(); job = self.wait_for_terminal(active_id)
        self.assertEqual(job["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
