import json
import unittest
from pathlib import Path

from flask import Flask, request

from ghc_api.compat_profiles import (
    KNOWN_ANTHROPIC_BETAS,
    MODE_COMPATIBILITY,
    MODE_LOSSLESS_REQUIRED,
    audit_anthropic_request,
    audit_responses_event,
    audit_responses_item,
    canonical_tool_contract_hash,
    normalize_anthropic_betas,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "anthropic_responses"


def known_headers(version="2.1.207"):
    return {
        "User-Agent": "claude-cli/%s (external, cli)" % version,
        "Anthropic-Version": "2023-06-01",
        "Anthropic-Beta": ",".join(sorted(KNOWN_ANTHROPIC_BETAS)),
    }


def known_payload():
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "metadata": {"user_id": "private-user"},
        "stream": True,
        "system": [{"type": "text", "text": "private system"}],
        "temperature": 1,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
        "tools": [],
    }


def known_responses_items():
    private = "PRIVATE-REQUIRED-VALUE"
    return {
        "additional_tools": {
            "type": "additional_tools", "role": "developer", "tools": [],
        },
        "agent_message": {
            "type": "agent_message", "author": private, "content": [],
            "recipient": private,
        },
        "custom_tool_call": {
            "type": "custom_tool_call", "call_id": private, "name": private,
            "input": private,
        },
        "custom_tool_call_output": {
            "type": "custom_tool_call_output", "call_id": private,
            "output": private,
        },
        "function_call": {
            "type": "function_call", "call_id": private, "name": private,
            "arguments": "{}",
        },
        "function_call_output": {
            "type": "function_call_output", "call_id": private,
            "output": private,
        },
        "message": {
            "type": "message", "role": "assistant", "content": [],
        },
        "reasoning": {
            "type": "reasoning", "summary": [], "encrypted_content": private,
        },
    }


def known_responses_events():
    private = "PRIVATE-REQUIRED-VALUE"
    sequence = 1
    output_text_part = {"type": "output_text", "text": private}
    summary_part = {"type": "summary_text", "text": private}
    function_item = known_responses_items()["function_call"]
    common_delta = {
        "sequence_number": sequence,
        "output_index": 0,
        "item_id": private,
    }
    common_content = {**common_delta, "content_index": 0}
    common_summary = {**common_delta, "summary_index": 0}
    return {
        "keepalive": {"type": "keepalive", "sequence_number": sequence},
        "response.created": {
            "type": "response.created", "sequence_number": sequence,
            "response": {},
        },
        "response.queued": {
            "type": "response.queued", "sequence_number": sequence,
            "response": {},
        },
        "response.in_progress": {
            "type": "response.in_progress", "sequence_number": sequence,
            "response": {},
        },
        "response.output_item.added": {
            "type": "response.output_item.added", "sequence_number": sequence,
            "output_index": 0, "item": dict(function_item),
        },
        "response.output_item.done": {
            "type": "response.output_item.done", "sequence_number": sequence,
            "output_index": 0, "item": dict(function_item),
        },
        "response.content_part.added": {
            "type": "response.content_part.added", **common_content,
            "part": dict(output_text_part),
        },
        "response.content_part.done": {
            "type": "response.content_part.done", **common_content,
            "part": dict(output_text_part),
        },
        "response.output_text.delta": {
            "type": "response.output_text.delta", **common_content,
            "delta": private, "logprobs": [],
        },
        "response.output_text.done": {
            "type": "response.output_text.done", **common_content,
            "text": private, "logprobs": [],
        },
        "response.function_call_arguments.delta": {
            "type": "response.function_call_arguments.delta", **common_delta,
            "delta": private,
        },
        "response.function_call_arguments.done": {
            "type": "response.function_call_arguments.done", **common_delta,
            "arguments": "{}",
        },
        "response.custom_tool_call_input.delta": {
            "type": "response.custom_tool_call_input.delta", **common_delta,
            "delta": private,
        },
        "response.custom_tool_call_input.done": {
            "type": "response.custom_tool_call_input.done", **common_delta,
            "input": private,
        },
        "response.refusal.delta": {
            "type": "response.refusal.delta", **common_content,
            "delta": private,
        },
        "response.refusal.done": {
            "type": "response.refusal.done", **common_content,
            "refusal": private,
        },
        "response.reasoning_summary_part.added": {
            "type": "response.reasoning_summary_part.added", **common_summary,
            "part": dict(summary_part),
        },
        "response.reasoning_summary_part.done": {
            "type": "response.reasoning_summary_part.done", **common_summary,
            "part": dict(summary_part),
        },
        "response.reasoning_summary_text.delta": {
            "type": "response.reasoning_summary_text.delta", **common_summary,
            "delta": private,
        },
        "response.reasoning_summary_text.done": {
            "type": "response.reasoning_summary_text.done", **common_summary,
            "text": private,
        },
        "response.reasoning_text.delta": {
            "type": "response.reasoning_text.delta", **common_content,
            "delta": private,
        },
        "response.reasoning_text.done": {
            "type": "response.reasoning_text.done", **common_content,
            "text": private,
        },
        "response.completed": {
            "type": "response.completed", "sequence_number": sequence,
            "response": {},
        },
        "response.incomplete": {
            "type": "response.incomplete", "sequence_number": sequence,
            "response": {},
        },
        "response.failed": {
            "type": "response.failed", "sequence_number": sequence,
            "response": {},
        },
        "error": {
            "type": "error", "sequence_number": sequence, "code": private,
            "message": private, "param": None,
        },
    }


class ClaudeCompatibilityProfileTests(unittest.TestCase):
    def test_known_capture_versions_and_normalized_beta_set(self):
        for version in ("2.1.197", "2.1.207"):
            headers = known_headers(version)
            headers["anthropic-beta"] = " , ".join(
                reversed([token.upper() for token in KNOWN_ANTHROPIC_BETAS])
            ) + ",CLAUDE-CODE-20250219"
            # Header lookup is case-insensitive; remove the original spelling
            # so this is not a synthetic duplicate-header case.
            del headers["Anthropic-Beta"]
            audit = audit_anthropic_request(headers, known_payload())
            with self.subTest(version=version):
                self.assertEqual(audit.profile.name, "claude_cli_" + version.replace(".", "_"))
                self.assertEqual(audit.profile.cli_version, version)
                self.assertTrue(audit.profile.known_cli_version)
                self.assertEqual(audit.profile.anthropic_betas, tuple(sorted(KNOWN_ANTHROPIC_BETAS)))
                self.assertEqual(audit.warnings, [])
                self.assertTrue(audit.allowed)
                self.assertFalse(audit.should_fail)

    def test_flask_headers_are_accepted(self):
        app = Flask(__name__)
        with app.test_request_context("/", headers=known_headers()):
            audit = audit_anthropic_request(request.headers, known_payload())
        self.assertEqual(audit.warnings, [])

    def test_unknown_cli_version_always_warns_but_never_fails_by_itself(self):
        for mode in (MODE_COMPATIBILITY, MODE_LOSSLESS_REQUIRED):
            audit = audit_anthropic_request(known_headers("2.2.0"), known_payload(), mode=mode)
            warning = next(item for item in audit.warnings if item["code"] == "claude_cli.unknown_version")
            with self.subTest(mode=mode):
                self.assertEqual(warning["version"], "2.2.0")
                self.assertEqual(warning["action"], "warn")
                self.assertFalse(audit.should_fail)

    def test_unknown_beta_and_anthropic_version_are_mode_aware(self):
        headers = known_headers()
        headers["Anthropic-Version"] = "2099-01-01"
        headers["Anthropic-Beta"] += ",future-private-beta"
        compatibility = audit_anthropic_request(headers, known_payload(), mode=MODE_COMPATIBILITY)
        lossless = audit_anthropic_request(headers, known_payload(), mode=MODE_LOSSLESS_REQUIRED)

        expected_codes = {"anthropic.version_unknown", "anthropic.beta_unknown"}
        self.assertTrue(expected_codes.issubset({item["code"] for item in compatibility.warnings}))
        self.assertFalse(compatibility.should_fail)
        self.assertTrue(lossless.should_fail)
        self.assertTrue(
            all(
                item["action"] == "reject"
                for item in lossless.warnings
                if item["code"] in expected_codes
            )
        )
        # Header protocol values influence only the fingerprint/profile.  The
        # warning itself must not disclose a newly introduced beta name.
        warning_json = json.dumps(lossless.warnings, sort_keys=True)
        self.assertNotIn("future-private-beta", warning_json)
        self.assertNotIn("2099-01-01", warning_json)

    def test_conditional_context_beta_is_known_and_other_subset_is_visible(self):
        headers = known_headers()
        headers["Anthropic-Beta"] = ",".join(
            sorted(KNOWN_ANTHROPIC_BETAS - {"context-1m-2025-08-07"})
        )
        self.assertEqual(audit_anthropic_request(headers, known_payload()).warnings, [])

        headers["Anthropic-Beta"] = "claude-code-20250219"
        audit = audit_anthropic_request(headers, known_payload(), mode=MODE_LOSSLESS_REQUIRED)
        warning = next(
            item for item in audit.warnings if item["code"] == "anthropic.beta_set_unknown"
        )
        self.assertEqual(warning["action"], "warn")
        self.assertFalse(audit.should_fail)

    def test_unknown_fields_types_and_enums_are_structured_and_mode_aware(self):
        payload = known_payload()
        payload["future_option"] = "DO-NOT-LOG-this-body-value"
        payload["max_tokens"] = "32000"
        payload["thinking"] = {"type": "future-thinking-kind"}

        compatibility = audit_anthropic_request(known_headers(), payload)
        lossless = audit_anthropic_request(
            known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
        )
        codes = {item["code"] for item in compatibility.warnings}
        self.assertIn("request.unknown_field", codes)
        self.assertIn("request.invalid_type", codes)
        self.assertIn("request.unknown_enum", codes)
        self.assertFalse(compatibility.should_fail)
        self.assertTrue(lossless.should_fail)

        required_warning_fields = {
            "code", "path", "types", "observed_type", "expected_types",
            "version", "fingerprint", "action",
        }
        for warning in compatibility.warnings:
            self.assertTrue(required_warning_fields.issubset(warning))
            self.assertRegex(warning["fingerprint"], r"^sha256:[0-9a-f]{64}$")
        serialized = json.dumps(compatibility.warnings, sort_keys=True)
        self.assertNotIn("DO-NOT-LOG", serialized)
        self.assertNotIn("future-thinking-kind", serialized)

    def test_missing_top_level_required_fields_warn_or_reject_without_values(self):
        for field in ("model", "messages", "max_tokens"):
            payload = known_payload()
            removed = payload.pop(field)
            compatibility = audit_anthropic_request(known_headers(), payload)
            warning = next(
                item for item in compatibility.warnings
                if item["code"] == "request.missing_required_field"
                and item["path"] == "/" + field
            )
            with self.subTest(field=field, mode=MODE_COMPATIBILITY):
                self.assertEqual(warning["observed_type"], "missing")
                self.assertEqual(warning["action"], "warn")
                self.assertFalse(compatibility.should_fail)
                self.assertNotIn(str(removed), json.dumps(compatibility.warnings))

            lossless = audit_anthropic_request(
                known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
            )
            with self.subTest(field=field, mode=MODE_LOSSLESS_REQUIRED):
                self.assertTrue(lossless.should_fail)
                self.assertTrue(any(
                    item["code"] == "request.missing_required_field"
                    and item["path"] == "/" + field
                    and item["action"] == "reject"
                    for item in lossless.warnings
                ))

    def test_missing_tool_and_content_fields_are_discriminator_aware(self):
        private = "PRIVATE-REQUIRED-VALUE"
        cases = [
            ({"type": "text"}, ("text",)),
            ({"type": "image"}, ("source",)),
            ({"type": "document"}, ("source",)),
            ({"type": "tool_use"}, ("id", "name", "input")),
            ({"type": "tool_result"}, ("tool_use_id",)),
            ({"type": "thinking", "thinking": private}, ("signature",)),
            ({"type": "redacted_thinking"}, ("data",)),
        ]
        for block, missing_fields in cases:
            payload = known_payload()
            payload["messages"] = [{"role": "user", "content": [block]}]
            compatibility = audit_anthropic_request(known_headers(), payload)
            expected_paths = {
                "/messages/0/content/0/" + field for field in missing_fields
            }
            warnings = [
                item for item in compatibility.warnings
                if item["code"] == "content_block.missing_required_field"
            ]
            with self.subTest(block=block["type"], mode=MODE_COMPATIBILITY):
                self.assertEqual({item["path"] for item in warnings}, expected_paths)
                self.assertTrue(all(item["action"] == "warn" for item in warnings))
                self.assertNotIn(private, json.dumps(compatibility.warnings))
            lossless = audit_anthropic_request(
                known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
            )
            self.assertTrue(lossless.should_fail)

        payload = known_payload()
        payload["tools"] = [{"description": private}]
        compatibility = audit_anthropic_request(known_headers(), payload)
        tool_warnings = [
            item for item in compatibility.warnings
            if item["code"] == "tool.missing_required_field"
        ]
        self.assertEqual(
            {item["path"] for item in tool_warnings},
            {"/tools/0/name", "/tools/0/input_schema"},
        )
        self.assertFalse(compatibility.should_fail)
        self.assertNotIn(private, json.dumps(compatibility.warnings))
        self.assertTrue(audit_anthropic_request(
            known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
        ).should_fail)

    def test_missing_source_fields_are_reported_without_source_values(self):
        private = "PRIVATE-REQUIRED-VALUE"
        sources = [
            ({"type": "base64", "data": private}, ("media_type",)),
            ({"type": "url"}, ("url",)),
            ({"type": "text"}, ("data",)),
        ]
        for source, missing_fields in sources:
            payload = known_payload()
            payload["messages"] = [{
                "role": "user",
                "content": [{"type": "document", "source": source}],
            }]
            compatibility = audit_anthropic_request(known_headers(), payload)
            warnings = [
                item for item in compatibility.warnings
                if item["code"] == "content_source.missing_required_field"
            ]
            expected = {
                "/messages/0/content/0/source/" + field
                for field in missing_fields
            }
            with self.subTest(source=source["type"]):
                self.assertEqual({item["path"] for item in warnings}, expected)
                self.assertFalse(compatibility.should_fail)
                self.assertNotIn(private, json.dumps(compatibility.warnings))
                self.assertTrue(audit_anthropic_request(
                    known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
                ).should_fail)

    def test_suspicious_unknown_field_key_is_redacted_from_warning_path(self):
        identity = "user@example.invalid"
        payload = known_payload()
        payload[identity] = "private body"
        audit = audit_anthropic_request(known_headers(), payload)
        serialized = json.dumps(audit.warnings, sort_keys=True)
        self.assertNotIn(identity, serialized)
        self.assertEqual(audit.warnings[0]["path"], "/<redacted-key>")

    def test_unknown_content_block_and_context_edit_fail_lossless(self):
        payload = known_payload()
        payload["messages"][0]["content"].append(
            {"type": "future_private_block", "private_body": "never log me"}
        )
        payload["context_management"] = {
            "edits": [{"type": "future_private_edit", "keep": "all"}]
        }
        compatibility = audit_anthropic_request(known_headers(), payload)
        lossless = audit_anthropic_request(
            known_headers(), payload, mode=MODE_LOSSLESS_REQUIRED
        )
        codes = {item["code"] for item in compatibility.warnings}
        self.assertIn("content_block.unknown_type", codes)
        self.assertIn("context_edit.unknown_type", codes)
        self.assertFalse(compatibility.should_fail)
        self.assertTrue(lossless.should_fail)
        serialized = json.dumps(lossless.warnings, sort_keys=True)
        self.assertNotIn("future_private_block", serialized)
        self.assertNotIn("future_private_edit", serialized)
        self.assertNotIn("never log me", serialized)

    def test_all_converter_facing_request_fields_have_audited_shapes(self):
        payload = known_payload()
        payload.update(
            {
                "stop_sequences": ["private-stop"],
                "top_p": 0.9,
                "top_k": 4,
                "tool_choice": {
                    "type": "tool",
                    "name": "Read",
                    "disable_parallel_tool_use": False,
                },
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
                },
                "output_config": {
                    "effort": "high",
                    "format": {"type": "json_schema", "private": "opaque"},
                },
                "service_tier": "auto",
            }
        )
        audit = audit_anthropic_request(known_headers(), payload)
        self.assertEqual(audit.warnings, [])

    def test_content_block_variants_and_nested_tool_result_are_known(self):
        payload = known_payload()
        payload["messages"] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "private"},
                    {"type": "redacted_thinking", "data": "private"},
                    {
                        "type": "tool_use",
                        "id": "toolu_private",
                        "name": "Read",
                        "input": {"private": "opaque"},
                        "metadata": {"identity": "opaque"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_private",
                        "is_error": False,
                        "content": [
                            {"type": "text", "text": "private"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "private",
                                },
                            },
                            {
                                "type": "document",
                                "source": {"type": "text", "data": "private"},
                                "title": "private",
                                "citations": {"private": "opaque"},
                            },
                        ],
                    }
                ],
            },
        ]
        audit = audit_anthropic_request(known_headers(), payload)
        self.assertEqual(audit.warnings, [])

    def test_dynamic_tool_subtrees_are_opaque(self):
        secret = "user@example.invalid SUPER-SECRET-PROMPT"
        payload = known_payload()
        payload["metadata"] = {"user_id": secret, "arbitrary": {"nested": secret}}
        payload["messages"] = [
            {
                "role": "assistant",
                "metadata": {"identity": secret},
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "PrivateTool",
                        "input": {"arbitrary_property": secret},
                        "metadata": {"identity": secret},
                    }
                ],
            }
        ]
        payload["tools"] = [
            {
                "name": "PrivateTool",
                "description": secret,
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "arbitrary_property": {"type": "string", "description": secret}
                    },
                },
                "metadata": {"identity": secret},
            }
        ]
        audit = audit_anthropic_request(known_headers(), payload)
        self.assertEqual(audit.warnings, [])
        self.assertNotIn(secret, json.dumps(audit.to_dict(), sort_keys=True))

    def test_canonical_builtin_tool_contract_hash_and_baseline_comparison(self):
        contract = {
            "name": "Read",
            "description": "read a file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        }
        reordered = {
            "input_schema": {
                "required": ["file_path"],
                "properties": {"file_path": {"type": "string"}},
                "type": "object",
            },
            "description": "read a file",
            "name": "Read",
        }
        self.assertEqual(
            canonical_tool_contract_hash(contract), canonical_tool_contract_hash(reordered)
        )
        # Cache placement is deliberately not part of the callable contract.
        cached = dict(contract, cache_control={"type": "ephemeral"})
        self.assertEqual(canonical_tool_contract_hash(contract), canonical_tool_contract_hash(cached))

        payload = known_payload()
        payload["tools"] = [dict(contract)]
        baseline = {"tools": {"Read": canonical_tool_contract_hash(contract)}}
        self.assertEqual(
            audit_anthropic_request(
                known_headers(), payload, baseline_manifest=baseline
            ).warnings,
            [],
        )

        payload["tools"][0]["description"] = "PRIVATE CONTRACT CHANGE"
        compatibility = audit_anthropic_request(
            known_headers(), payload, baseline_manifest=baseline
        )
        warning = next(item for item in compatibility.warnings if item["code"] == "tool.contract_mismatch")
        self.assertEqual(warning["path"], "/tools/0")
        self.assertEqual(warning["action"], "warn")
        self.assertNotIn("PRIVATE CONTRACT CHANGE", json.dumps(compatibility.warnings))
        lossless = audit_anthropic_request(
            known_headers(),
            payload,
            mode=MODE_LOSSLESS_REQUIRED,
            baseline_manifest=baseline,
        )
        self.assertTrue(lossless.should_fail)

    def test_warning_order_and_deduplication_are_stable(self):
        headers_a = known_headers()
        headers_a["Anthropic-Beta"] = "future-z, future-a, future-z"
        headers_b = {
            "anthropic-beta": " future-a,future-z ",
            "anthropic-version": "2023-06-01",
            "user-agent": "claude-cli/2.1.207",
        }
        payload_a = dict(known_payload(), z_unknown=1, a_unknown=2)
        payload_b = {"a_unknown": 99, **known_payload(), "z_unknown": 88}
        first = audit_anthropic_request(headers_a, payload_a).warnings
        second = audit_anthropic_request(headers_b, payload_b).warnings
        self.assertEqual(first, second)
        self.assertEqual(sum(item["code"] == "anthropic.beta_unknown" for item in first), 1)
        unknown_paths = [item["path"] for item in first if item["code"] == "request.unknown_field"]
        self.assertEqual(unknown_paths, ["/a_unknown", "/z_unknown"])

    def test_audit_to_dict_is_directly_json_serializable(self):
        result = audit_anthropic_request(known_headers(), known_payload()).to_dict()
        encoded = json.dumps(result, sort_keys=True)
        self.assertIn('"allowed": true', encoded)
        self.assertIn('"profile"', encoded)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_anthropic_request(known_headers(), known_payload(), mode="strict")


class ResponsesCompatibilityAuditTests(unittest.TestCase):
    def test_known_event_and_item_have_no_warnings(self):
        for event_type, event in known_responses_events().items():
            with self.subTest(event_type=event_type):
                self.assertEqual(audit_responses_event(event).warnings, [])
        for item_type, item in known_responses_items().items():
            with self.subTest(item_type=item_type):
                self.assertEqual(audit_responses_item(item).warnings, [])

    def test_every_event_type_reports_each_missing_required_field(self):
        private = "PRIVATE-REQUIRED-VALUE"
        for event_type, complete in known_responses_events().items():
            for field in sorted(set(complete) - {"type"}):
                event = dict(complete)
                event.pop(field)
                compatibility = audit_responses_event(event)
                expected_path = "/events/" + field
                with self.subTest(
                    event_type=event_type, field=field,
                    mode=MODE_COMPATIBILITY,
                ):
                    warning = next(
                        item for item in compatibility.warnings
                        if item["code"] == "responses.missing_event_field"
                        and item["path"] == expected_path
                    )
                    self.assertEqual(warning["observed_type"], "missing")
                    self.assertEqual(warning["action"], "warn")
                    self.assertFalse(compatibility.should_fail)
                    self.assertNotIn(private, json.dumps(compatibility.warnings))

                lossless = audit_responses_event(
                    event, mode=MODE_LOSSLESS_REQUIRED
                )
                with self.subTest(
                    event_type=event_type, field=field,
                    mode=MODE_LOSSLESS_REQUIRED,
                ):
                    self.assertTrue(lossless.should_fail)
                    self.assertTrue(any(
                        item["code"] == "responses.missing_event_field"
                        and item["path"] == expected_path
                        and item["action"] == "reject"
                        for item in lossless.warnings
                    ))
                    self.assertNotIn(private, json.dumps(lossless.warnings))

    def test_every_item_type_reports_each_missing_required_field(self):
        private = "PRIVATE-REQUIRED-VALUE"
        for item_type, complete in known_responses_items().items():
            for field in sorted(set(complete) - {"type"}):
                item = dict(complete)
                item.pop(field)
                compatibility = audit_responses_item(item)
                expected_path = "/output/0/" + field
                with self.subTest(
                    item_type=item_type, field=field,
                    mode=MODE_COMPATIBILITY,
                ):
                    warning = next(
                        value for value in compatibility.warnings
                        if value["code"] == "responses.missing_item_field"
                        and value["path"] == expected_path
                    )
                    self.assertEqual(warning["observed_type"], "missing")
                    self.assertEqual(warning["action"], "warn")
                    self.assertFalse(compatibility.should_fail)
                    self.assertNotIn(private, json.dumps(compatibility.warnings))

                lossless = audit_responses_item(
                    item, mode=MODE_LOSSLESS_REQUIRED
                )
                with self.subTest(
                    item_type=item_type, field=field,
                    mode=MODE_LOSSLESS_REQUIRED,
                ):
                    self.assertTrue(lossless.should_fail)
                    self.assertTrue(any(
                        value["code"] == "responses.missing_item_field"
                        and value["path"] == expected_path
                        and value["action"] == "reject"
                        for value in lossless.warnings
                    ))
                    self.assertNotIn(private, json.dumps(lossless.warnings))

    def test_response_content_parts_require_their_payload_field(self):
        parts = {
            "input_text": "text",
            "output_text": "text",
            "refusal": "refusal",
            "summary_text": "text",
            "encrypted_content": "encrypted_content",
        }
        for part_type, required_field in parts.items():
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": part_type}],
            }
            compatibility = audit_responses_item(item)
            expected_path = "/output/0/content/0/" + required_field
            with self.subTest(part_type=part_type):
                warning = next(
                    value for value in compatibility.warnings
                    if value["code"] == "responses.missing_content_field"
                )
                self.assertEqual(warning["path"], expected_path)
                self.assertFalse(compatibility.should_fail)
                self.assertTrue(audit_responses_item(
                    item, mode=MODE_LOSSLESS_REQUIRED
                ).should_fail)

    def test_unknown_event_fails_closed_in_both_modes_without_value_leak(self):
        for mode in (MODE_COMPATIBILITY, MODE_LOSSLESS_REQUIRED):
            audit = audit_responses_event(
                {"type": "response.private_future.delta", "delta": "SECRET"}, mode=mode
            )
            with self.subTest(mode=mode):
                self.assertTrue(audit.should_fail)
                self.assertEqual(audit.warnings[0]["code"], "responses.unknown_event")
                self.assertEqual(audit.warnings[0]["action"], "reject")
                serialized = json.dumps(audit.warnings, sort_keys=True)
                self.assertNotIn("response.private_future.delta", serialized)
                self.assertNotIn("SECRET", serialized)

    def test_unknown_item_fails_closed_directly_and_inside_terminal_event(self):
        direct = audit_responses_item({"type": "private_future_item", "body": "SECRET"})
        terminal = audit_responses_event(
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "private_future_item", "body": "SECRET"}]
                },
            }
        )
        for audit in (direct, terminal):
            self.assertTrue(audit.should_fail)
            self.assertIn("responses.unknown_item", {item["code"] for item in audit.warnings})
            serialized = json.dumps(audit.warnings, sort_keys=True)
            self.assertNotIn("private_future_item", serialized)
            self.assertNotIn("SECRET", serialized)

    def test_non_object_event_and_item_fail_closed(self):
        self.assertTrue(audit_responses_event("private event").should_fail)
        self.assertTrue(audit_responses_item(["private item"]).should_fail)

    def test_new_fields_on_known_event_and_item_warn_or_reject_without_leak(self):
        secret = "PRIVATE-FUTURE-FIELD-VALUE"
        event = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "fixture",
            "delta": "fixture",
            "future_field": secret,
        }
        item = {
            "type": "message",
            "role": "assistant",
            "content": [],
            "future_field": secret,
        }
        for value, audit_function, code in (
            (event, audit_responses_event, "responses.unknown_event_field"),
            (item, audit_responses_item, "responses.unknown_item_field"),
        ):
            compatibility = audit_function(value)
            self.assertFalse(compatibility.should_fail)
            self.assertIn(code, {warning["code"] for warning in compatibility.warnings})
            lossless = audit_function(value, mode=MODE_LOSSLESS_REQUIRED)
            self.assertTrue(lossless.should_fail)
            self.assertNotIn(secret, json.dumps(lossless.warnings))


class NormalizationTests(unittest.TestCase):
    def test_beta_normalization_strips_sorts_casefolds_and_deduplicates(self):
        self.assertEqual(
            normalize_anthropic_betas(" Zeta,alpha, ALPHA ,, beta "),
            ("alpha", "beta", "zeta"),
        )


class SanitizedDumpProfileRegressionTests(unittest.TestCase):
    def test_all_sanitized_claude_request_shapes_match_their_profiles(self):
        # The generated fixtures are a structural cover of every field/type/
        # discriminator observed in the user-supplied captures.  Samples are
        # minimised independently per root, so exercise every known header-set
        # variant against every retained request shape for that CLI version.
        for filename in ("claude_cli_2_1_197.json", "claude_cli_2_1_207.json"):
            document = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
            headers = [sample["value"] for sample in document["samples"] if sample["root"] == "headers"]
            payloads = [
                sample["value"]
                for sample in document["samples"]
                if sample["root"] == "original_request"
            ]
            self.assertTrue(headers)
            self.assertTrue(payloads)
            for header in headers:
                for payload in payloads:
                    with self.subTest(filename=filename, model=payload.get("model")):
                        self.assertEqual(audit_anthropic_request(header, payload).warnings, [])

    def test_all_sanitized_gpt_event_and_input_item_discriminators_are_known(self):
        document = json.loads(
            (FIXTURE_DIR / "gpt_5_6_responses.json").read_text(encoding="utf-8")
        )
        for sample in document["samples"]:
            if sample["root"] == "event":
                self.assertEqual(audit_responses_event(sample["value"]).warnings, [])
            elif sample["root"] == "request":
                for index, item in enumerate(sample["value"].get("input", [])):
                    self.assertEqual(
                        audit_responses_item(item, path="/input/%d" % index).warnings,
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
