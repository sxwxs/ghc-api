import copy
import json
import unittest

from ghc_api.anthropic_responses import (
    MODE_LOSSLESS_REQUIRED,
    AnthropicResponsesConversionError,
    IdentifierCodec,
    StrictJSONError,
    anthropic_error_from_responses,
    convert_anthropic_to_responses,
    convert_responses_to_anthropic,
    prepare_replay_items_for_wire,
    parse_strict_json_bytes,
)


class StrictJsonTests(unittest.TestCase):
    def test_rejects_ambiguous_or_non_unicode_json(self):
        invalid = (
            b'{"a":1,"a":2}',
            b'{"value":NaN}',
            b'{"value":1e999}',
            b'{"value":1} trailing',
            b'{"value":"\xff"}',
            b'{"value":"\\ud800"}',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(StrictJSONError):
                    parse_strict_json_bytes(raw)

    def test_accepts_valid_surrogate_pair_as_unicode_scalar(self):
        self.assertEqual(
            parse_strict_json_bytes(b'{"value":"\\ud83d\\ude00"}'),
            {"value": "\U0001f600"},
        )


class AnthropicResponsesRequestTranslationTests(unittest.TestCase):
    def full_payload(self):
        return {
            "model": "gpt-5.6-sol",
            "system": [
                {"type": "text", "text": "system one"},
                {"type": "text", "text": "system two", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "README.md"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok", "is_error": False},
                        {"type": "text", "text": "continue", "cache_control": {"type": "ephemeral"}},
                    ],
                },
            ],
            "tools": [{
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "strict": True,
                "cache_control": {"type": "ephemeral"},
            }],
            "metadata": {"user_id": '{"session_id":"session-test"}'},
            "max_tokens": 32000,
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
            "service_tier": "auto",
            "stream": True,
        }

    def test_full_observed_request_is_accounted_and_ordered(self):
        result = convert_anthropic_to_responses(
            self.full_payload(),
            wire_profile="copilot_responses_lite",
            session_id="session-test",
            tenant_id="tenant-test",
        )
        self.assertEqual(result.report.unaccounted_paths, [])
        self.assertEqual(result.payload["input"][0]["type"], "additional_tools")
        types = [item.get("type") for item in result.payload["input"]]
        self.assertEqual(types, [
            "additional_tools", "message", "message", "message",
            "function_call", "function_call_output", "message",
        ])
        self.assertEqual(result.payload["reasoning"], {"effort": "max", "context": "all_turns"})
        self.assertEqual(result.payload["max_output_tokens"], 32000)
        self.assertTrue(result.payload["parallel_tool_calls"])
        schema = result.payload["input"][0]["tools"][0]["parameters"]
        self.assertIn("$schema", schema)
        self.assertTrue(result.payload["input"][0]["tools"][0]["strict"])
        self.assertIn("prompt_cache_key", result.payload)

    def test_billing_system_block_is_omitted_without_other_prompt_rewrites(self):
        payload = {
            "model": "gpt-test",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=test; cc_entrypoint=cli;",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "keep this system text",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "prefix x-anthropic-billing-header: is ordinary text"},
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = convert_anthropic_to_responses(
            payload,
            wire_profile="public_responses",
            sidecar_available=True,
        )
        system_item = result.payload["input"][0]
        self.assertEqual(
            [part["text"] for part in system_item["content"]],
            [
                "keep this system text",
                "prefix x-anthropic-billing-header: is ordinary text",
            ],
        )
        self.assertEqual(
            system_item["content"][0]["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )
        self.assertEqual(result.report.unaccounted_paths, [])
        string_only = convert_anthropic_to_responses({
            "model": "gpt-test",
            "system": "x-anthropic-billing-header: cc_version=test;",
            "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(
            [item["role"] for item in string_only.payload["input"]],
            ["user"],
        )
        billing_record = next(
            record for record in result.report.records
            if record.source_path == "/system/0"
        )
        self.assertTrue(billing_record.subtree)
        self.assertEqual(billing_record.disposition, "semantic_encoding")

    def test_anthropic_web_search_maps_to_native_responses_tool(self):
        payload = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
                "allowed_domains": ["python.org"],
                "user_location": {"type": "approximate", "country": "US"},
            }],
            "tool_choice": {"type": "tool", "name": "web_search"},
        }
        for profile in ("public_responses", "copilot_responses_lite"):
            with self.subTest(profile=profile):
                result = convert_anthropic_to_responses(
                    payload,
                    wire_profile=profile,
                    sidecar_available=True,
                )
                self.assertEqual(result.payload["tools"], [{
                    "type": "web_search",
                    "filters": {"allowed_domains": ["python.org"]},
                    "user_location": {"type": "approximate", "country": "US"},
                }])
                self.assertEqual(result.payload["tool_choice"], {"type": "web_search"})
                self.assertFalse(any(
                    item.get("type") == "additional_tools"
                    for item in result.payload["input"]
                ))
                self.assertEqual(result.report.unaccounted_paths, [])
                self.assertTrue(any(
                    record.source_path == "/tools/0/max_uses"
                    and record.disposition == "approximation"
                    for record in result.report.records
                ))
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(
                payload,
                wire_profile="copilot_responses_lite",
                mode=MODE_LOSSLESS_REQUIRED,
                sidecar_available=True,
            )
        malformed = copy.deepcopy(payload)
        malformed["tools"][0]["input_schema"] = {"type": "object"}
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)
        malformed = copy.deepcopy(payload)
        malformed["tools"][0]["type"] = "web_search_20990101"
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)
        malformed = copy.deepcopy(payload)
        malformed["tools"][0]["max_uses"] = -1
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)
        malformed = copy.deepcopy(payload)
        malformed["tools"][0]["allowed_domains"] = "python.org"
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)
        malformed = copy.deepcopy(payload)
        malformed["tools"][0]["user_location"]["future"] = "private"
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)

    def test_json_schema_format_gets_deterministic_required_name(self):
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        }
        payload = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        first = convert_anthropic_to_responses(payload)
        second = convert_anthropic_to_responses(copy.deepcopy(payload))
        self.assertEqual(first.payload["text"]["format"]["schema"], schema)
        self.assertRegex(
            first.payload["text"]["format"]["name"],
            r"^ghc_schema_[0-9a-f]{16}$",
        )
        self.assertEqual(
            first.payload["text"]["format"]["name"],
            second.payload["text"]["format"]["name"],
        )
        reordered = copy.deepcopy(payload)
        reordered["output_config"]["format"]["schema"] = {
            "additionalProperties": False,
            "required": ["title"],
            "properties": {"title": {"type": "string"}},
            "type": "object",
        }
        self.assertEqual(
            first.payload["text"]["format"]["name"],
            convert_anthropic_to_responses(reordered).payload["text"]["format"]["name"],
        )
        self.assertEqual(first.report.unaccounted_paths, [])

    def test_json_schema_format_preserves_explicit_fields_and_rejects_bad_schema(self):
        payload = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"format": {
                "type": "json_schema",
                "name": "response_shape",
                "description": "Return a response shape",
                "schema": {"type": "object", "properties": {}},
                "strict": False,
            }},
        }
        result = convert_anthropic_to_responses(payload)
        self.assertEqual(result.payload["text"]["format"], payload["output_config"]["format"])
        malformed = copy.deepcopy(payload)
        malformed["output_config"]["format"]["schema"] = "not-an-object"
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(malformed)
        extended = copy.deepcopy(payload)
        extended["output_config"]["format"]["future"] = "unsupported"
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(extended)

    def test_missing_or_non_array_messages_is_always_rejected(self):
        invalid_payloads = (
            {"model": "gpt-test"},
            {"model": "gpt-test", "messages": None},
            {"model": "gpt-test", "messages": {}},
            {"model": "gpt-test", "messages": "not-an-array"},
        )
        for payload in invalid_payloads:
            with self.subTest(messages=payload.get("messages", "<missing>")):
                with self.assertRaises(AnthropicResponsesConversionError) as raised:
                    convert_anthropic_to_responses(payload)
                self.assertTrue(any(
                    record.source_path == "/messages"
                    and record.disposition == "unsupported"
                    for record in raised.exception.report.records
                ))

    def test_string_and_multimodal_blocks(self):
        payload = {
            "model": "gpt-test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.invalid/image.png"}},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA=="}},
                    {"type": "document", "source": {"type": "text", "data": "document text"}},
                ],
            }],
            "stream": False,
        }
        result = convert_anthropic_to_responses(payload, wire_profile="public_responses")
        content = result.payload["input"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_image", "input_image", "input_file", "input_text"])
        self.assertEqual(result.report.unaccounted_paths, [])

    def test_web_search_call_is_sidecar_only_and_tool_usage_is_accounted(self):
        result = convert_responses_to_anthropic(
            {
                "id": "resp_search",
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "search_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "private query",
                            "queries": ["private query"],
                        },
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "search answer",
                            "annotations": [],
                        }],
                    },
                ],
                "usage": {},
                "tool_usage": {"web_search": {"num_requests": 1}},
            },
            original_model="gpt-5.6-sol",
            sidecar_available=True,
        )
        self.assertEqual(result.response["content"], [{"type": "text", "text": "search answer"}])
        self.assertEqual(result.response["stop_reason"], "end_turn")
        self.assertEqual(
            result.response["usage"]["server_tool_use"],
            {"web_search_requests": 1},
        )
        self.assertEqual(result.replay_items[0]["type"], "web_search_call")
        self.assertEqual(result.report.unaccounted_paths, [])

        other_usage = convert_responses_to_anthropic(
            {
                "id": "resp_other_usage",
                "model": "gpt-test",
                "status": "completed",
                "output": [],
                "usage": {},
                "tool_usage": {"image_gen": {"total_tokens": 1}},
            },
            original_model="gpt-test",
            sidecar_available=True,
        )
        self.assertEqual(other_usage.report.unaccounted_paths, [])
        self.assertNotIn("server_tool_use", other_usage.response["usage"])

    def test_terminal_response_requires_id_output_and_supported_items(self):
        base = {
            "id": "resp_valid",
            "model": "gpt-test",
            "status": "completed",
            "output": [],
            "usage": {},
        }
        for mutation in (
            lambda value: value.pop("id"),
            lambda value: value.pop("output"),
            lambda value: value.update({"output": [{"type": "agent_message"}]}),
        ):
            payload = copy.deepcopy(base)
            mutation(payload)
            with self.assertRaises(AnthropicResponsesConversionError):
                convert_responses_to_anthropic(payload, original_model="gpt-test")

    def test_tool_result_error_is_explicit_approximation(self):
        payload = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "call_1", "content": "failed", "is_error": True,
            }]}],
        }
        result = convert_anthropic_to_responses(payload)
        self.assertTrue(any(w["path"].endswith("/is_error") for w in result.report.warnings))
        envelope = json.loads(result.payload["input"][0]["output"])
        self.assertEqual(envelope, {
            "ghc_anthropic_tool_result": {
                "is_error": True,
                "content": "failed",
            }
        })
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_anthropic_to_responses(payload, mode=MODE_LOSSLESS_REQUIRED)

    def test_lossless_mode_rejects_unknown_and_top_k(self):
        payload = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 12,
            "future_field": {"new": True},
        }
        with self.assertRaises(AnthropicResponsesConversionError) as raised:
            convert_anthropic_to_responses(payload, mode=MODE_LOSSLESS_REQUIRED)
        paths = {record.source_path for record in raised.exception.report.records}
        self.assertIn("/top_k", paths)
        self.assertIn("/future_field/new", paths)

    def test_replay_resolver_replaces_approximate_assistant_history(self):
        replay = [
            {"type": "reasoning", "summary": [], "encrypted_content": "encrypted"},
            {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "working"}]},
            {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"},
        ]
        payload = {
            "model": "gpt-test",
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "working"},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}},
            ]}],
        }
        result = convert_anthropic_to_responses(
            payload,
            replay_resolver=lambda index, message: copy.deepcopy(replay),
            mode=MODE_LOSSLESS_REQUIRED,
            sidecar_available=True,
        )
        self.assertEqual(result.payload["input"], replay)
        self.assertEqual(result.replay_misses, [])

    def test_identifier_codec_is_reversible_and_bounded(self):
        codec = IdentifierCodec(max_length=32)
        original = "tool name with spaces/and/a/very/long/suffix"
        encoded = codec.encode(original, "name")
        self.assertLessEqual(len(encoded), 32)
        self.assertNotEqual(encoded, original)
        self.assertEqual(codec.decode(encoded), original)

    def test_copilot_replay_wire_projection_keeps_reasoning_and_phase(self):
        terminal_items = [
            {
                "id": "rs_1", "type": "reasoning", "content": [],
                "summary": [], "encrypted_content": "opaque",
            },
            {
                "id": "msg_1", "type": "message", "role": "assistant",
                "status": "completed", "phase": "commentary",
                "content": [{
                    "type": "output_text", "text": "working",
                    "annotations": [], "logprobs": [],
                }],
            },
            {
                "id": "fc_1", "type": "function_call",
                "status": "completed", "call_id": "call_1",
                "name": "Read", "namespace": "tools", "arguments": "{}",
            },
        ]
        projected = prepare_replay_items_for_wire(
            terminal_items, "copilot_responses_lite"
        )
        self.assertEqual(projected, [
            {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
            {
                "type": "message", "role": "assistant", "phase": "commentary",
                "content": [{"type": "output_text", "text": "working"}],
            },
            {
                "type": "function_call", "call_id": "call_1", "name": "Read",
                "namespace": "tools", "arguments": "{}",
            },
        ])
        self.assertEqual(terminal_items[0]["id"], "rs_1")

    def test_lossless_sidecar_fields_require_a_durable_store_signal(self):
        payload = {
            "model": "gpt-test",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "Zml4dHVyZQ==",
                    },
                    "title": "PRIVATE-TITLE-SENTINEL",
                }],
            }],
        }
        with self.assertRaises(AnthropicResponsesConversionError) as raised:
            convert_anthropic_to_responses(
                payload,
                wire_profile="public_responses",
                mode=MODE_LOSSLESS_REQUIRED,
            )
        self.assertIn(
            "conversion.sidecar_unavailable",
            {warning["code"] for warning in raised.exception.report.warnings},
        )
        self.assertNotIn(
            "PRIVATE-TITLE-SENTINEL",
            json.dumps(raised.exception.report.to_dict()),
        )
        converted = convert_anthropic_to_responses(
            payload,
            wire_profile="public_responses",
            mode=MODE_LOSSLESS_REQUIRED,
            sidecar_available=True,
        )
        self.assertTrue(converted.report.sidecar_available)

    def test_metadata_json_types_are_accounted_in_sidecar(self):
        base = {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        string_result = convert_anthropic_to_responses({
            **base, "metadata": {"value": "1"},
        })
        number_result = convert_anthropic_to_responses({
            **base, "metadata": {"value": 1},
        })
        self.assertEqual(
            string_result.payload["metadata"], number_result.payload["metadata"]
        )
        records = []
        for result in (string_result, number_result):
            records.append(next(
                record for record in result.report.records
                if record.source_path == "/metadata/value"
            ))
        self.assertEqual(records[0].disposition, "exact")
        self.assertEqual(records[1].disposition, "sidecar")


class AnthropicResponsesResponseTranslationTests(unittest.TestCase):
    def terminal_response(self):
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 123,
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": [], "content": [], "encrypted_content": "encrypted"},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "working", "annotations": [], "logprobs": []}],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"path":"README.md"}',
                },
            ],
            "parallel_tool_calls": True,
            "reasoning": {"effort": "max", "context": "all_turns"},
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 12},
                "total_tokens": 120,
            },
        }

    def test_terminal_response_preserves_replay_items_and_phase(self):
        result = convert_responses_to_anthropic(
            self.terminal_response(),
            original_model="claude-opus-4.8",
            mode=MODE_LOSSLESS_REQUIRED,
            sidecar_available=True,
        )
        self.assertEqual(result.report.unaccounted_paths, [])
        self.assertEqual(result.replay_items, self.terminal_response()["output"])
        self.assertEqual(result.replay_items[1]["phase"], "commentary")
        self.assertEqual(result.response["stop_reason"], "tool_use")
        self.assertEqual(result.response["usage"]["input_tokens"], 50)
        self.assertEqual(result.response["usage"]["cache_read_input_tokens"], 40)
        self.assertEqual(result.response["usage"]["cache_creation_input_tokens"], 10)
        self.assertEqual(result.response["usage"]["output_tokens_details"]["thinking_tokens"], 12)
        self.assertEqual([b["type"] for b in result.response["content"]], ["text", "tool_use"])
        total_record = next(
            record for record in result.report.records
            if record.source_path == "/usage/total_tokens"
        )
        self.assertEqual(total_record.disposition, "sidecar")

    def test_lossless_usage_extension_requires_sidecar(self):
        response = {
            "id": "resp_usage",
            "model": "gpt-test",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
        }
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_responses_to_anthropic(
                response,
                original_model="claude",
                mode=MODE_LOSSLESS_REQUIRED,
            )
        converted = convert_responses_to_anthropic(
            response,
            original_model="claude",
            mode=MODE_LOSSLESS_REQUIRED,
            sidecar_available=True,
        )
        self.assertEqual(converted.response["usage"]["output_tokens"], 2)

    def test_nonstream_stop_sequence(self):
        response = self.terminal_response()
        response["output"] = [
            {
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "pre-stop-reasoning",
            },
            {
                "type": "message", "role": "assistant", "phase": "final_answer",
                "content": [{
                    "type": "output_text",
                    "text": "before<STOP>after",
                    "annotations": [],
                }],
            },
            {
                "type": "function_call",
                "call_id": "hidden-call",
                "name": "HiddenTool",
                "arguments": "{}",
            },
            {
                "type": "message", "role": "assistant", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "also hidden"}],
            },
        ]
        result = convert_responses_to_anthropic(
            response,
            original_model="claude",
            stop_sequences=["<STOP>"],
        )
        self.assertEqual(result.response["content"][0]["text"], "before")
        self.assertEqual(len(result.response["content"]), 1)
        self.assertEqual(result.response["stop_reason"], "stop_sequence")
        self.assertEqual(result.response["stop_sequence"], "<STOP>")
        self.assertEqual([item["type"] for item in result.replay_items], [
            "reasoning", "message",
        ])
        self.assertEqual(
            result.replay_items[1]["content"][0]["text"], "before"
        )
        self.assertEqual(
            result.replay_items[1]["content"][0]["annotations"], []
        )
        # Conversion must not mutate the authoritative upstream object retained
        # by the encrypted audit snapshot.
        self.assertEqual(
            response["output"][1]["content"][0]["text"],
            "before<STOP>after",
        )

    def test_stop_sequence_never_matches_across_output_item_or_tool_boundary(self):
        response = {
            "id": "resp_stop_boundary",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "leftX"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_boundary",
                    "name": "Read",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Yright"}],
                },
            ],
            "usage": {},
        }
        result = convert_responses_to_anthropic(
            response,
            original_model="claude",
            stop_sequences=["XY"],
        )
        self.assertIsNone(result.matched_stop_sequence)
        self.assertEqual(
            [block["type"] for block in result.response["content"]],
            ["text", "tool_use", "text"],
        )

        same_item = copy.deepcopy(response)
        same_item["output"] = [{
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "leftX"},
                {"type": "output_text", "text": "Yright"},
            ],
        }]
        matched = convert_responses_to_anthropic(
            same_item,
            original_model="claude",
            stop_sequences=["XY"],
        )
        self.assertEqual(matched.matched_stop_sequence, "XY")
        self.assertEqual(matched.response["content"], [
            {"type": "text", "text": "left"}
        ])

    def test_refusal_is_visible_and_has_refusal_stop_reason(self):
        response = self.terminal_response()
        response["output"] = [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "cannot comply"}],
        }]
        result = convert_responses_to_anthropic(
            response, original_model="claude"
        )
        self.assertEqual(
            result.response["content"],
            [{"type": "text", "text": "cannot comply"}],
        )
        self.assertEqual(result.response["stop_reason"], "refusal")

    def test_content_filter_incomplete_maps_to_refusal(self):
        response = self.terminal_response()
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "content_filter"}
        response["output"] = []
        result = convert_responses_to_anthropic(
            response, original_model="claude"
        )
        self.assertEqual(result.response["stop_reason"], "refusal")

    def test_max_output_incomplete_maps_to_max_tokens(self):
        response = self.terminal_response()
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        response["output"] = []
        result = convert_responses_to_anthropic(
            response, original_model="claude"
        )
        self.assertEqual(result.response["stop_reason"], "max_tokens")

    def test_unknown_or_missing_incomplete_reason_fails_closed(self):
        for details in ({"reason": "future-private-reason"}, {}, None):
            response = self.terminal_response()
            response["status"] = "incomplete"
            response["incomplete_details"] = details
            response["output"] = []
            with self.subTest(details=details):
                with self.assertRaises(AnthropicResponsesConversionError) as raised:
                    convert_responses_to_anthropic(
                        response, original_model="claude"
                    )
                self.assertTrue(any(
                    record.disposition == "unsupported"
                    and record.source_path.startswith("/incomplete_details")
                    for record in raised.exception.report.records
                ))
                self.assertNotIn(
                    "future-private-reason",
                    json.dumps(raised.exception.report.to_dict()),
                )

    def test_function_arguments_must_be_one_strict_json_object(self):
        invalid_arguments = (
            "[]",
            "1",
            "null",
            "not-json",
            '{"duplicate":1,"duplicate":2}',
        )
        for arguments in invalid_arguments:
            response = self.terminal_response()
            response["output"] = [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "Read",
                "arguments": arguments,
            }]
            with self.subTest(arguments=arguments):
                with self.assertRaises(AnthropicResponsesConversionError) as raised:
                    convert_responses_to_anthropic(
                        response, original_model="claude"
                    )
                self.assertEqual(
                    raised.exception.report.warnings[0]["path"],
                    "/output/0/arguments",
                )

        response = self.terminal_response()
        response["output"] = [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "Read",
            "arguments": '{"path":"README.md"}',
        }]
        result = convert_responses_to_anthropic(
            response,
            original_model="claude",
            mode=MODE_LOSSLESS_REQUIRED,
            sidecar_available=True,
        )
        self.assertEqual(
            result.response["content"][0]["input"],
            {"path": "README.md"},
        )

    def test_custom_tool_input_uses_reversible_wrapper(self):
        response = self.terminal_response()
        response["output"] = [{
            "type": "custom_tool_call",
            "call_id": "call_custom",
            "name": "shell",
            "input": '{"looks":"like json"}',
        }]
        result = convert_responses_to_anthropic(
            response, original_model="claude"
        )
        self.assertEqual(
            result.response["content"][0]["input"],
            {"input": '{"looks":"like json"}'},
        )
        self.assertTrue(any(
            warning["path"] == "/output/0/input"
            for warning in result.report.warnings
        ))
        with self.assertRaises(AnthropicResponsesConversionError):
            convert_responses_to_anthropic(
                response,
                original_model="claude",
                mode=MODE_LOSSLESS_REQUIRED,
            )

    def test_error_mapping(self):
        mapped = anthropic_error_from_responses({"error": {"message": "bad", "code": "x"}}, 429)
        self.assertEqual(mapped["error"]["type"], "rate_limit_error")
        self.assertEqual(mapped["error"]["message"], "bad")


if __name__ == "__main__":
    unittest.main()
