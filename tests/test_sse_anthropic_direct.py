"""Tests for the direct-Anthropic SSE handler and its recovery subclass.

The user's main concern: when ``state.enable_tool_call_recovery`` is False the
recovery code must NOT be in the hot path. The route factory picks the base
class in that case; the base class must not transitively reference
``tool_call_recovery``. This file pins both invariants.
"""

import json
import unittest
from unittest import mock

from ghc_api.cache import RequestCache
from ghc_api.sse import (
    AnthropicDirectStreamHandler,
    AnthropicDirectStreamHandlerWithRecovery,
)
from ghc_api.sse import anthropic_direct as anthropic_direct_module
from ghc_api.sse import base as base_module


class _FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""

    def iter_lines(self):
        for line in self._lines:
            yield line

    def close(self):
        pass


class RecoveryClassIsolationTest(unittest.TestCase):
    """When the route picks the base class (recovery toggle off), the
    transformer must not be instantiated or called."""

    def test_base_class_does_not_import_or_reference_transformer(self):
        # AnthropicDirectStreamHandler must NOT carry a transformer attribute.
        cache = RequestCache()
        with mock.patch.object(base_module, "cache", cache):
            handler = AnthropicDirectStreamHandler(
                response=_FakeResponse([b"data: [DONE]"]),
                request_id="req-1",
                request_size=0,
                start_time=0.0,
                original_model="claude-opus-4",
                translated_model="claude-opus-4",
                request_body_for_cache={},
            )
        self.assertFalse(hasattr(handler, "_transformer"))

    def test_base_class_extra_cache_fields_does_not_include_recovered_content(self):
        handler = AnthropicDirectStreamHandler(
            response=_FakeResponse([]),
            request_id="x",
            request_size=0,
            start_time=0.0,
            original_model="m",
            translated_model="m",
            request_body_for_cache={},
        )
        self.assertEqual(handler.extra_cache_fields(), {})


class RecoverySubclassRoutingTest(unittest.TestCase):
    """The subclass plumbs every event through the transformer's ``process`` and
    emits a sidecar ``recovered_content`` field in the cache record."""

    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()

    def test_subclass_attaches_transformer_and_calls_process(self):
        message_start = json.dumps({"type": "message_start", "message": {"usage": {}}})
        delta = json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            b"data: [DONE]",
        ]
        handler = AnthropicDirectStreamHandlerWithRecovery(
            response=_FakeResponse(lines),
            request_id="req-r",
            request_size=0,
            start_time=0.0,
            original_model="claude-opus-4",
            translated_model="claude-opus-4",
            request_body_for_cache={},
        )
        self.assertTrue(hasattr(handler, "_transformer"))

        process_calls = []
        real_process = handler._transformer.process
        def spy(event_type, event, raw_data):
            process_calls.append((event_type, event.get("type")))
            return real_process(event_type, event, raw_data)
        handler._transformer.process = spy

        out = "".join(handler._generate())
        # Non-leaked stream: passthrough, every event surfaces.
        self.assertIn(f"data: {message_start}\n", out)
        self.assertIn(f"data: {delta}\n", out)
        # Each upstream event reached the transformer.
        self.assertEqual(len(process_calls), 2)

        entry = self.cache.get_request("req-r")
        # Sidecar field present; raw_events still complete.
        self.assertIn("recovered_content", entry or {})
        self.assertEqual(entry["raw_events"], [message_start, delta])


class UsageAccountingTest(unittest.TestCase):
    """Usage counters must survive Anthropic-compatible gateways that report
    ``input_tokens`` only in the trailing ``message_delta`` event."""

    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(base_module, "cache", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()

    def _entry_after(self, message_start_usage, message_delta_usage):
        message_start = json.dumps(
            {"type": "message_start", "message": {"model": "m", "usage": message_start_usage}}
        )
        delta = json.dumps(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
        )
        message_delta = json.dumps(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": message_delta_usage}
        )
        lines = [
            b"event: message_start",
            f"data: {message_start}".encode(),
            b"event: content_block_delta",
            f"data: {delta}".encode(),
            b"event: message_delta",
            f"data: {message_delta}".encode(),
            b"event: message_stop",
            b'data: {"type": "message_stop"}',
        ]
        handler = AnthropicDirectStreamHandler(
            response=_FakeResponse(lines),
            request_id="req-usage",
            request_size=0,
            start_time=0.0,
            original_model="m",
            translated_model="m",
            request_body_for_cache={},
        )
        "".join(handler._generate())
        entry = self.cache.get_request("req-usage")
        self.assertIsNotNone(entry)
        return entry

    def test_input_tokens_reported_only_in_message_delta_are_recorded(self):
        entry = self._entry_after(
            {"input_tokens": 0, "output_tokens": 0},
            {"input_tokens": 128207, "output_tokens": 136, "cache_read_input_tokens": 0},
        )
        self.assertEqual(entry["input_tokens"], 128207)
        self.assertEqual(entry["output_tokens"], 136)
        self.assertEqual(entry["cache_read_input_tokens"], 0)

    def test_message_delta_does_not_erase_counters_it_omits(self):
        entry = self._entry_after(
            {"input_tokens": 11, "output_tokens": 0, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3},
            {"output_tokens": 9},
        )
        self.assertEqual(entry["input_tokens"], 11)
        self.assertEqual(entry["output_tokens"], 9)
        self.assertEqual(entry["cache_creation_input_tokens"], 2)
        self.assertEqual(entry["cache_read_input_tokens"], 3)

    def test_message_delta_overrides_stale_cache_counters(self):
        entry = self._entry_after(
            {"input_tokens": 5, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 2},
            {"input_tokens": 40, "output_tokens": 7, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 20},
        )
        self.assertEqual(entry["input_tokens"], 40)
        self.assertEqual(entry["output_tokens"], 7)
        self.assertEqual(entry["cache_creation_input_tokens"], 10)
        self.assertEqual(entry["cache_read_input_tokens"], 20)

    def test_non_dict_usage_is_ignored(self):
        entry = self._entry_after({"input_tokens": 5}, None)
        # A null usage object must not raise nor reset what message_start reported.
        self.assertEqual(entry["input_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
