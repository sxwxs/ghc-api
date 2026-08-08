import json
import sqlite3
import tempfile
import threading
import unittest
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from ghc_api.reasoning_replay import (
    ReasoningReplayStore,
    ReplayConflictError,
    ReplayEncryptionConfigurationError,
    ReplayParentError,
    ReplayQuotaExceededError,
    ReplaySerializationError,
    ReplayStoreClosedError,
    assistant_visible_fingerprint,
    canonical_json,
    canonical_json_bytes,
)

try:
    from cryptography.fernet import Fernet

    HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover - validates optional-dependency behavior
    Fernet = None
    HAS_CRYPTOGRAPHY = False


class FakeClock:
    def __init__(self, value=1_800_000_000.0):
        self._value = float(value)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = float(value)

    def advance(self, seconds):
        with self._lock:
            self._value += float(seconds)


def sample_visible_blocks(suffix="one"):
    return [
        {"type": "text", "text": "准备工具调用 — %s" % suffix},
        {
            "type": "tool_use",
            "id": "toolu_%s" % suffix,
            "name": "Read",
            "input": {"file_path": "C:/src/你好/%s.py" % suffix},
        },
    ]


def sample_output_items(suffix="one", encrypted="opaque-secret-marker"):
    """Terminal Responses items with fields that must never be normalised away."""

    return [
        {
            "id": "rs_%s" % suffix,
            "type": "reasoning",
            "status": "completed",
            "summary": [
                {"type": "summary_text", "text": "检查路径 %s" % suffix}
            ],
            "content": None,
            "encrypted_content": encrypted,
        },
        {
            "id": "msg_%s" % suffix,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": [
                {
                    "type": "output_text",
                    "text": "正在读取 %s" % suffix,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {
            "id": "fc_%s" % suffix,
            "type": "function_call",
            "status": "completed",
            "phase": "commentary",
            "call_id": "call_%s" % suffix,
            "name": "Read",
            "arguments": json.dumps(
                {"file_path": "C:/src/你好/%s.py" % suffix},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def put_sample(store, suffix="one", **overrides):
    values = {
        "tenant_id": "tenant-a",
        "session_id": "session-a",
        "model": "gpt-5.6-sol",
        "output_items": sample_output_items(suffix),
        "assistant_visible_blocks": sample_visible_blocks(suffix),
        "profile": {
            "name": "copilot_responses_lite",
            "version": 1,
            "beta": ["context-management-2025-06-27"],
        },
    }
    values.update(overrides)
    return store.put(**values)


class CanonicalJsonTests(unittest.TestCase):
    def test_object_order_and_json_whitespace_have_one_canonical_form(self):
        first = {"z": [3, 2, 1], "a": "你好", "nested": {"b": True, "a": None}}
        second = json.loads(
            '{ "nested": { "a": null, "b": true }, "a": "\\u4f60\\u597d", '
            '"z": [3, 2, 1] }'
        )

        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(
            assistant_visible_fingerprint(first),
            assistant_visible_fingerprint(second),
        )

    def test_array_order_and_unicode_codepoints_are_not_normalised(self):
        self.assertNotEqual(
            assistant_visible_fingerprint([1, 2]),
            assistant_visible_fingerprint([2, 1]),
        )
        composed = "é"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        self.assertNotEqual(
            assistant_visible_fingerprint([composed]),
            assistant_visible_fingerprint([decomposed]),
        )
        self.assertNotEqual(canonical_json_bytes(True), canonical_json_bytes(1))
        self.assertNotEqual(canonical_json_bytes(1), canonical_json_bytes(1.0))

    def test_non_json_or_lossy_values_are_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        invalid_values = [
            float("nan"),
            float("inf"),
            float("-inf"),
            {1: "integer key would be coerced"},
            b"bytes are not JSON",
            "\ud800",
            cyclic,
        ]
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ReplaySerializationError):
                    canonical_json_bytes(value)


class ReasoningReplayStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "replay.sqlite3"
        self.clock = FakeClock()
        self.stores = []

    def tearDown(self):
        for store in reversed(self.stores):
            store.close()
        self.temporary_directory.cleanup()

    def make_store(self, **kwargs):
        kwargs.setdefault("clock", self.clock)
        store = ReasoningReplayStore(self.db_path, **kwargs)
        self.stores.append(store)
        return store

    def test_restart_preserves_every_output_item_phase_profile_and_time(self):
        output_items = sample_output_items(
            "restart", encrypted="opaque:\u0000-looking-but-json-safe"
        )
        visible = sample_visible_blocks("restart")
        profile = {
            "name": "copilot_responses_lite",
            "response_item_input_shape": {"drop_on_wire": ["id"], "revision": 7},
        }
        store = self.make_store(ttl_seconds=120)
        written = store.put(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            output_items=output_items,
            assistant_visible_blocks=visible,
            profile=profile,
        )
        store.close()

        reopened = self.make_store(ttl_seconds=999)
        result = reopened.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_blocks=visible,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.clean)
        self.assertFalse(result.ambiguous)
        self.assertEqual(len(result.records), 1)
        loaded = result.records[0]
        self.assertEqual(loaded.replay_id, written.replay_id)
        self.assertEqual(canonical_json_bytes(loaded.output_items), canonical_json_bytes(output_items))
        self.assertEqual(
            canonical_json_bytes(loaded.assistant_visible_blocks),
            canonical_json_bytes(visible),
        )
        self.assertEqual(canonical_json_bytes(loaded.profile), canonical_json_bytes(profile))
        self.assertEqual(loaded.output_items[1]["phase"], "commentary")
        self.assertEqual(loaded.output_items[2]["phase"], "commentary")
        self.assertEqual(loaded.created_at, self.clock())
        self.assertEqual(loaded.expires_at, self.clock() + 120)
        reopened.close()

    def test_plaintext_payload_is_exact_canonical_snapshot_without_truncation(self):
        store = self.make_store()
        record = put_sample(store, "raw")
        expected = canonical_json_bytes(
            {
                "version": 1,
                "output_items": sample_output_items("raw"),
                "assistant_visible_blocks": sample_visible_blocks("raw"),
                "profile": {
                    "name": "copilot_responses_lite",
                    "version": 1,
                    "beta": ["context-management-2025-06-27"],
                },
            }
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            payload, encrypted = connection.execute(
                "SELECT payload, encrypted FROM reasoning_replay_records WHERE replay_id = ?",
                (record.replay_id,),
            ).fetchone()
        self.assertEqual(bytes(payload), expected)
        self.assertEqual(encrypted, 0)
        store.close()

    def test_forks_and_same_visible_retry_are_not_overwritten(self):
        store = self.make_store(ttl_seconds=3600)
        root = put_sample(store, "root")
        branch_a = put_sample(
            store,
            "branch-a",
            parent_replay_id=root.replay_id,
        )
        branch_b = put_sample(
            store,
            "branch-b",
            parent_replay_id=root.replay_id,
        )
        retry_a = put_sample(
            store,
            "branch-a",
            parent_replay_id=root.replay_id,
            output_items=sample_output_items(
                "branch-a", encrypted="different-hidden-retry-state"
            ),
        )

        result_a = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_blocks=sample_visible_blocks("branch-a"),
        )
        self.assertTrue(result_a.ambiguous)
        self.assertEqual(
            {record.replay_id for record in result_a.records},
            {branch_a.replay_id, retry_a.replay_id},
        )
        self.assertEqual(
            {
                record.output_items[0]["encrypted_content"]
                for record in result_a.records
            },
            {"opaque-secret-marker", "different-hidden-retry-state"},
        )

        result_b = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=branch_b.assistant_visible_fingerprint,
        )
        self.assertEqual([record.replay_id for record in result_b.records], [branch_b.replay_id])
        self.assertEqual(result_b.records[0].parent_replay_id, root.replay_id)

        exact_retry = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_blocks=sample_visible_blocks("branch-a"),
            replay_id=retry_a.replay_id,
        )
        self.assertEqual([record.replay_id for record in exact_retry.records], [retry_a.replay_id])
        store.close()

    def test_parent_must_exist_and_stay_in_tenant_session_model_scope(self):
        store = self.make_store()
        root = put_sample(store, "parent")
        with self.assertRaises(ReplayParentError):
            put_sample(store, "missing", parent_replay_id="does-not-exist")
        for override in (
            {"tenant_id": "tenant-b"},
            {"session_id": "session-b"},
            {"model": "gpt-5.6-terra"},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ReplayParentError):
                    put_sample(
                        store,
                        "cross-scope",
                        parent_replay_id=root.replay_id,
                        **override,
                    )
        store.close()

    def test_tenant_session_and_model_are_isolated(self):
        store = self.make_store()
        visible = sample_visible_blocks("shared")
        dimensions = [
            ("tenant-a", "session-a", "gpt-5.6-sol", "a"),
            ("tenant-b", "session-a", "gpt-5.6-sol", "b"),
            ("tenant-a", "session-b", "gpt-5.6-sol", "c"),
            ("tenant-a", "session-a", "gpt-5.6-terra", "d"),
        ]
        ids = {}
        for tenant, session, model, marker in dimensions:
            record = put_sample(
                store,
                marker,
                tenant_id=tenant,
                session_id=session,
                model=model,
                assistant_visible_blocks=visible,
            )
            ids[(tenant, session, model)] = record.replay_id

        for tenant, session, model, _ in dimensions:
            with self.subTest(tenant=tenant, session=session, model=model):
                result = store.get(
                    tenant_id=tenant,
                    session_id=session,
                    model=model,
                    assistant_visible_blocks=visible,
                )
                self.assertEqual(
                    [record.replay_id for record in result.records],
                    [ids[(tenant, session, model)]],
                )
        store.close()

    def test_identity_validation_fingerprint_validation_and_conflict(self):
        store = self.make_store()
        for field in ("tenant_id", "session_id", "model"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    put_sample(store, "invalid-%s" % field, **{field: "  "})

        with self.assertRaises(ValueError):
            put_sample(
                store,
                "mismatch",
                assistant_visible_fingerprint=assistant_visible_fingerprint([]),
            )
        first = put_sample(store, "conflict", replay_id="fixed-id")
        self.assertEqual(first.replay_id, "fixed-id")
        with self.assertRaises(ReplayConflictError):
            put_sample(store, "conflict", replay_id="fixed-id")
        store.close()

    def test_ttl_boundary_reports_expiry_and_purge_is_scoped(self):
        store = self.make_store(ttl_seconds=10)
        short = put_sample(store, "short")
        long = put_sample(store, "long", ttl_seconds=30)
        self.clock.advance(9.999)
        before = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=short.assistant_visible_fingerprint,
        )
        self.assertTrue(before.found)

        self.clock.set(short.expires_at)
        at_boundary = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=short.assistant_visible_fingerprint,
        )
        self.assertFalse(at_boundary.found)
        self.assertEqual(at_boundary.expired_count, 1)
        self.assertEqual(store.purge(), 0)

        self.clock.set(long.expires_at)
        self.assertEqual(store.purge(tenant_id="tenant-a"), 1)
        self.assertEqual(store.purge(tenant_id="tenant-a"), 0)
        store.close()

    def test_plaintext_mode_is_explicit_and_required_encryption_fails(self):
        store = self.make_store()
        self.assertEqual(store.encryption_status.mode, "plaintext")
        self.assertFalse(store.encryption_status.key_configured)
        self.assertIn("plaintext", store.encryption_status.message)
        self.assertIn("No Fernet key", store.encryption_status.message)
        store.close()

        with self.assertRaises(ReplayEncryptionConfigurationError):
            self.make_store(require_encryption=True)

    def test_payload_and_expiry_tampering_fail_hmac_without_returning_state(self):
        store = self.make_store(ttl_seconds=100)
        payload_record = put_sample(store, "tamper-payload")
        expiry_record = put_sample(store, "tamper-expiry")

        with closing(sqlite3.connect(self.db_path)) as connection:
            payload = connection.execute(
                "SELECT payload FROM reasoning_replay_records WHERE replay_id = ?",
                (payload_record.replay_id,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE reasoning_replay_records SET payload = ? WHERE replay_id = ?",
                (bytes(payload) + b"x", payload_record.replay_id),
            )
            connection.execute(
                "UPDATE reasoning_replay_records SET expires_at = 0 WHERE replay_id = ?",
                (expiry_record.replay_id,),
            )
            connection.commit()

        payload_result = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=payload_record.assistant_visible_fingerprint,
        )
        self.assertFalse(payload_result.found)
        self.assertEqual(payload_result.expired_count, 0)
        self.assertEqual([issue.code for issue in payload_result.issues], ["integrity_check_failed"])

        expiry_result = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=expiry_record.assistant_visible_fingerprint,
        )
        self.assertFalse(expiry_result.found)
        self.assertEqual(expiry_result.expired_count, 0)
        self.assertEqual([issue.code for issue in expiry_result.issues], ["integrity_check_failed"])
        store.close()

    def test_concurrent_retries_on_one_store_are_all_preserved(self):
        store = self.make_store(ttl_seconds=300)
        visible = sample_visible_blocks("concurrent")
        worker_count = 32

        def write(index):
            return put_sample(
                store,
                "worker-%d" % index,
                assistant_visible_blocks=visible,
                output_items=sample_output_items(
                    "worker-%d" % index,
                    encrypted="encrypted-worker-%d" % index,
                ),
            ).replay_id

        with ThreadPoolExecutor(max_workers=12) as executor:
            ids = set(executor.map(write, range(worker_count)))
        self.assertEqual(len(ids), worker_count)
        result = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_blocks=visible,
        )
        self.assertTrue(result.ambiguous)
        self.assertEqual({record.replay_id for record in result.records}, ids)
        self.assertEqual(len(result.records), worker_count)
        self.assertTrue(result.clean)
        store.close()

    def test_two_store_instances_can_write_same_database_concurrently(self):
        first = self.make_store(ttl_seconds=300)
        second = self.make_store(ttl_seconds=300)
        visible = sample_visible_blocks("two-stores")

        def write(index):
            target = first if index % 2 == 0 else second
            return put_sample(
                target,
                "instance-%d" % index,
                assistant_visible_blocks=visible,
            ).replay_id

        with ThreadPoolExecutor(max_workers=10) as executor:
            ids = set(executor.map(write, range(20)))
        result = first.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_blocks=visible,
        )
        self.assertEqual({record.replay_id for record in result.records}, ids)
        first.close()
        second.close()

    def test_workers_can_initialise_new_database_concurrently(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)

        def initialise(_):
            barrier.wait(timeout=10)
            store = ReasoningReplayStore(
                self.db_path,
                ttl_seconds=300,
                clock=self.clock,
            )
            try:
                return store.encryption_status.mode
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            modes = list(executor.map(initialise, range(worker_count)))
        self.assertEqual(modes, ["plaintext"] * worker_count)

        # Every constructor must converge on one persistent HMAC key/schema.
        store = self.make_store()
        record = put_sample(store, "post-init")
        result = store.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=record.assistant_visible_fingerprint,
        )
        self.assertEqual([item.replay_id for item in result.records], [record.replay_id])

    def test_close_is_idempotent_and_operations_fail_explicitly(self):
        store = self.make_store()
        store.close()
        store.close()
        self.assertTrue(store.closed)
        with self.assertRaises(ReplayStoreClosedError):
            store.get(
                tenant_id="tenant-a",
                session_id="session-a",
                model="gpt-5.6-sol",
                assistant_visible_blocks=[],
            )

    def test_logical_quotas_reject_before_insert_and_deleted_rows_free_capacity(self):
        store = self.make_store()
        first = put_sample(store, "one")
        used = store.logical_size_bytes()
        self.assertGreater(used, 0)
        self.assertEqual(store.logical_size_bytes(tenant_id="tenant-a"), used)

        with self.assertRaises(ReplayQuotaExceededError):
            put_sample(store, "two", max_total_bytes=used + 1)
        self.assertEqual(store.logical_size_bytes(), used)

        with self.assertRaises(ReplayQuotaExceededError):
            put_sample(store, "two", max_tenant_bytes=used + 1)
        other = put_sample(
            store,
            "two",
            tenant_id="tenant-b",
            max_tenant_bytes=used + 1,
        )
        self.assertGreater(store.logical_size_bytes(), used)

        store.purge(replay_id=first.replay_id, expired_only=False)
        store.purge(replay_id=other.replay_id, expired_only=False)
        self.assertEqual(store.logical_size_bytes(), 0)
        put_sample(store, "two", max_total_bytes=used + 1)

    def test_per_record_quota_rejects_without_writing(self):
        store = self.make_store()
        with self.assertRaises(ReplayQuotaExceededError):
            put_sample(store, "oversized", max_record_bytes=128)
        self.assertEqual(store.logical_size_bytes(), 0)
        for field in ("max_record_bytes", "max_tenant_bytes", "max_total_bytes"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    put_sample(store, "invalid", **{field: True})
        self.assertEqual(store.logical_size_bytes(), 0)


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "optional cryptography package is not installed")
class EncryptedReasoningReplayStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "encrypted-replay.sqlite3"
        self.clock = FakeClock()
        self.key = Fernet.generate_key()
        self.stores = []

    def tearDown(self):
        for store in reversed(self.stores):
            store.close()
        self.temporary_directory.cleanup()

    def make_store(self, key=None):
        store = ReasoningReplayStore(
            self.db_path,
            ttl_seconds=300,
            encryption_key=self.key if key is None else key,
            clock=self.clock,
        )
        self.stores.append(store)
        return store

    def test_encrypted_round_trip_restart_and_ciphertext_has_no_plaintext(self):
        store = self.make_store()
        record = put_sample(
            store,
            "encrypted",
            output_items=sample_output_items(
                "encrypted", encrypted="highly-sensitive-encrypted-reasoning"
            ),
        )
        self.assertTrue(record.encrypted)
        self.assertEqual(store.encryption_status.mode, "fernet")
        store.close()

        with closing(sqlite3.connect(self.db_path)) as connection:
            payload, encrypted = connection.execute(
                "SELECT payload, encrypted FROM reasoning_replay_records WHERE replay_id = ?",
                (record.replay_id,),
            ).fetchone()
        self.assertEqual(encrypted, 1)
        self.assertNotIn(b"highly-sensitive-encrypted-reasoning", bytes(payload))
        self.assertNotIn(b'"phase":"commentary"', bytes(payload))

        reopened = self.make_store()
        result = reopened.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=record.assistant_visible_fingerprint,
        )
        self.assertTrue(result.clean)
        self.assertEqual(
            result.records[0].output_items[0]["encrypted_content"],
            "highly-sensitive-encrypted-reasoning",
        )
        self.assertEqual(result.records[0].output_items[1]["phase"], "commentary")
        reopened.close()

    def test_missing_and_wrong_keys_report_failure_and_return_no_state(self):
        store = self.make_store()
        record = put_sample(store, "key-check")
        store.close()

        without_key = ReasoningReplayStore(
            self.db_path,
            ttl_seconds=300,
            clock=self.clock,
        )
        missing = without_key.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=record.assistant_visible_fingerprint,
        )
        self.assertFalse(missing.found)
        self.assertEqual([issue.code for issue in missing.issues], ["encryption_key_missing"])
        self.assertEqual(without_key.encryption_status.mode, "plaintext")
        without_key.close()

        wrong_key = self.make_store(key=Fernet.generate_key())
        failed = wrong_key.get(
            tenant_id="tenant-a",
            session_id="session-a",
            model="gpt-5.6-sol",
            assistant_visible_fingerprint=record.assistant_visible_fingerprint,
        )
        self.assertFalse(failed.found)
        self.assertEqual([issue.code for issue in failed.issues], ["decryption_failed"])
        wrong_key.close()

    def test_ciphertext_or_hmac_tampering_never_returns_replay_state(self):
        store = self.make_store()
        ciphertext_record = put_sample(store, "ciphertext-tamper")
        hmac_record = put_sample(store, "hmac-tamper")
        with closing(sqlite3.connect(self.db_path)) as connection:
            payload = connection.execute(
                "SELECT payload FROM reasoning_replay_records WHERE replay_id = ?",
                (ciphertext_record.replay_id,),
            ).fetchone()[0]
            changed = bytearray(payload)
            changed[len(changed) // 2] ^= 1
            connection.execute(
                "UPDATE reasoning_replay_records SET payload = ? WHERE replay_id = ?",
                (bytes(changed), ciphertext_record.replay_id),
            )
            connection.execute(
                "UPDATE reasoning_replay_records SET payload_hmac = zeroblob(32) "
                "WHERE replay_id = ?",
                (hmac_record.replay_id,),
            )
            connection.commit()

        for record in (ciphertext_record, hmac_record):
            with self.subTest(replay_id=record.replay_id):
                result = store.get(
                    tenant_id="tenant-a",
                    session_id="session-a",
                    model="gpt-5.6-sol",
                    assistant_visible_fingerprint=record.assistant_visible_fingerprint,
                )
                self.assertFalse(result.found)
                self.assertEqual(
                    [issue.code for issue in result.issues],
                    ["integrity_check_failed"],
                )
        store.close()


if __name__ == "__main__":
    unittest.main()
