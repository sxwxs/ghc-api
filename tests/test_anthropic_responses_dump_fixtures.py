import importlib.util
import json
import sys
import unittest
from pathlib import Path

from ghc_api.anthropic_responses import convert_anthropic_to_responses
from ghc_api.compat_profiles import audit_responses_event
from ghc_api.sse.anthropic_responses import ResponsesAnthropicEventTranslator


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "anthropic_responses"
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_anthropic_responses_fixtures.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_anthropic_responses_fixtures", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


class AnthropicResponsesDumpFixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
        cls.documents = {}
        for profile, profile_manifest in cls.manifest["profiles"].items():
            path = FIXTURE_DIR / profile_manifest["fixture"]
            cls.documents[profile] = json.loads(path.read_text(encoding="utf-8"))

    def test_fixture_inventory_and_schema(self):
        self.assertEqual(
            set(self.documents),
            {"claude_cli_2_1_197", "claude_cli_2_1_207", "gpt_5_6_responses"},
        )
        self.assertEqual(self.manifest["manifest_schema"], generator.MANIFEST_SCHEMA)
        for profile, document in self.documents.items():
            self.assertEqual(document["fixture_schema"], generator.FIXTURE_SCHEMA)
            self.assertEqual(document["profile"], profile)
            self.assertTrue(document["samples"])

    def test_fixtures_are_structural_and_sanitized(self):
        for profile, document in self.documents.items():
            with self.subTest(profile=profile):
                violations = generator.audit_sanitized_fixture(document)
                self.assertEqual(violations, [], "\n".join(violations))
                serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
                self.assertNotIn('"Authorization"', serialized)
                self.assertNotIn('"authorization"', serialized)

                # Every observed free-form payload class must be represented by
                # an explicit synthetic marker, never by source prose.
                self.assertIn("<text>", serialized)
                if profile == "gpt_5_6_responses":
                    self.assertIn("<opaque>", serialized)
                    self.assertIn("<input>", serialized)

    def test_manifest_declares_no_coverage_gaps(self):
        for profile, profile_manifest in self.manifest["profiles"].items():
            with self.subTest(profile=profile):
                self.assertTrue(profile_manifest["all_covered"])
                for root, coverage in profile_manifest["roots"].items():
                    self.assertEqual(coverage["summary"]["missing_tokens"], 0, root)
                    for entry in coverage["paths"]:
                        self.assertEqual(entry["missing_types"], [], entry["path"])
                    for entry in coverage["object_shapes"]:
                        self.assertTrue(entry["covered"], (entry["path"], entry["keys"]))
                    for entry in coverage["discriminators"]:
                        self.assertEqual(entry["missing_values"], [], entry["path"])

    def test_manifest_is_independently_covered_by_fixture_values(self):
        """Recompute coverage rather than trusting generated covered_* fields."""

        for profile, document in self.documents.items():
            actual_roots = generator.observe_fixture_document(document)
            expected_roots = self.manifest["profiles"][profile]["roots"]
            for root, expected in expected_roots.items():
                actual = actual_roots[root]
                with self.subTest(profile=profile, root=root):
                    for entry in expected["paths"]:
                        self.assertTrue(
                            set(entry["types"]).issubset(actual.paths.get(entry["path"], set())),
                            entry["path"],
                        )
                    for entry in expected["object_shapes"]:
                        self.assertIn(tuple(entry["keys"]), actual.shapes.get(entry["path"], set()))
                    for entry in expected["discriminators"]:
                        actual_values = {
                            json.loads(value) for value in actual.discriminators.get(entry["path"], set())
                        }
                        # JSON discriminator values observed here are scalar and
                        # therefore hashable.
                        self.assertTrue(set(entry["values"]).issubset(actual_values), entry["path"])

    def test_source_counts_capture_known_dump_limitations(self):
        p197 = self.manifest["profiles"]["claude_cli_2_1_197"]
        p207 = self.manifest["profiles"]["claude_cli_2_1_207"]
        gpt = self.manifest["profiles"]["gpt_5_6_responses"]
        self.assertEqual(p197["source_stats"]["records"], 5)
        self.assertEqual(p207["source_stats"]["records"], 8)
        self.assertEqual(gpt["source_stats"]["records"], 314)
        self.assertEqual(gpt["source_stats"]["complete_requests"], 274)
        self.assertEqual(gpt["source_stats"]["truncated_requests"], 40)
        self.assertEqual(gpt["source_stats"]["truncated_events"], 2)
        self.assertGreater(gpt["source_stats"]["events"], 40_000)

    def test_claude_versions_keep_the_observed_contract_differences(self):
        def path_types(profile, root, path):
            entries = self.manifest["profiles"][profile]["roots"][root]["paths"]
            return set(next(entry for entry in entries if entry["path"] == path)["types"])

        content_path = "original_request/messages/[]/content"
        self.assertEqual(
            path_types("claude_cli_2_1_197", "original_request", content_path),
            {"array"},
        )
        self.assertEqual(
            path_types("claude_cli_2_1_207", "original_request", content_path),
            {"array", "string"},
        )

        def find_tool(profile, name):
            for sample in self.documents[profile]["samples"]:
                if sample["root"] != "original_request":
                    continue
                for tool in sample["value"].get("tools", []):
                    if tool.get("name") == name:
                        return tool
            self.fail(f"missing {name} in {profile}")

        old_findings = find_tool("claude_cli_2_1_197", "ReportFindings")
        new_findings = find_tool("claude_cli_2_1_207", "ReportFindings")
        old_finding_properties = old_findings["input_schema"]["properties"]["findings"]["items"]["properties"]
        new_finding_properties = new_findings["input_schema"]["properties"]["findings"]["items"]["properties"]
        self.assertNotIn("category", old_finding_properties)
        self.assertEqual(new_finding_properties["category"]["type"], "string")
        self.assertEqual(new_finding_properties["category"]["maxLength"], 40)

        old_wakeup = find_tool("claude_cli_2_1_197", "ScheduleWakeup")["input_schema"]
        new_wakeup = find_tool("claude_cli_2_1_207", "ScheduleWakeup")["input_schema"]
        self.assertIn("required", old_wakeup)
        self.assertNotIn("required", new_wakeup)
        self.assertEqual(new_wakeup["properties"]["stop"]["type"], "boolean")

        for profile, version in (
            ("claude_cli_2_1_197", "2.1.197"),
            ("claude_cli_2_1_207", "2.1.207"),
        ):
            user_agents = {
                sample["value"]["User-Agent"]
                for sample in self.documents[profile]["samples"]
                if sample["root"] == "headers"
            }
            self.assertEqual(user_agents, {f"claude-cli/{version} (fixture)"})

    def test_protocol_discriminator_catalogs_cover_observed_variants(self):
        def values(profile, root, suffix):
            entries = self.manifest["profiles"][profile]["roots"][root]["discriminators"]
            matches = [entry for entry in entries if entry["path"] == root + suffix]
            self.assertEqual(len(matches), 1, (profile, root, suffix))
            return set(matches[0]["values"])

        for profile in ("claude_cli_2_1_197", "claude_cli_2_1_207"):
            self.assertEqual(
                values(profile, "event", "/type"),
                {
                    "message_start",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                    "message_delta",
                    "message_stop",
                },
            )
            block_types = values(profile, "original_request", "/messages/[]/content/[]/type")
            self.assertTrue({"text", "tool_use", "tool_result"}.issubset(block_types))
            self.assertEqual(
                values(profile, "event", "/content_block/type"),
                {"text", "thinking", "tool_use"},
            )
            self.assertEqual(
                values(profile, "event", "/delta/type"),
                {"input_json_delta", "signature_delta", "text_delta"},
            )

        self.assertEqual(
            values("gpt_5_6_responses", "request", "/input/[]/type"),
            {
                "additional_tools",
                "agent_message",
                "custom_tool_call",
                "custom_tool_call_output",
                "function_call",
                "function_call_output",
                "message",
                "reasoning",
            },
        )
        gpt_events = values("gpt_5_6_responses", "event", "/type")
        self.assertTrue(
            {
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.output_item.done",
                "response.output_text.delta",
                "response.output_text.done",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.completed",
                "keepalive",
            }.issubset(gpt_events)
        )
        self.assertEqual(
            values("gpt_5_6_responses", "event", "/response/output/[]/type"),
            {"custom_tool_call", "function_call", "message", "reasoning"},
        )

    def test_every_sanitized_claude_request_field_reaches_the_converter(self):
        for profile in ("claude_cli_2_1_197", "claude_cli_2_1_207"):
            for sample_index, sample in enumerate(self.documents[profile]["samples"]):
                if sample["root"] != "original_request":
                    continue
                with self.subTest(profile=profile, sample=sample_index):
                    converted = convert_anthropic_to_responses(
                        sample["value"],
                        wire_profile="copilot_responses_lite",
                        session_id="fixture-session",
                        tenant_id="fixture-tenant",
                    )
                    self.assertEqual(converted.report.unaccounted_paths, [])
                    self.assertTrue(converted.payload["input"])

    def test_every_sanitized_gpt_event_shape_reaches_the_sse_state_machine(self):
        for sample_index, sample in enumerate(
            self.documents["gpt_5_6_responses"]["samples"]
        ):
            if sample["root"] != "event":
                continue
            event = sample["value"]
            with self.subTest(sample=sample_index, event=event.get("type")):
                translator = ResponsesAnthropicEventTranslator(
                    original_model="fixture-model"
                )
                # Samples are field-covering minimal events rather than one
                # coherent stream. Processing each from a clean state proves
                # every observed event/field shape is accepted without using
                # any source prompt, identity, or encrypted value.
                translated = translator.process(event.get("type", ""), event)
                self.assertIsInstance(translated, list)

    def test_sanitized_coherent_stream_matches_output_oracle(self):
        document = json.loads(
            (FIXTURE_DIR / "coherent_stream.json").read_text(encoding="utf-8")
        )
        serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
        self.assertEqual(generator.audit_sanitized_fixture(document), [])
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("Bearer ", serialized)
        self.assertNotRegex(serialized, r"[A-Za-z]:\\")

        translator = ResponsesAnthropicEventTranslator(
            original_model="claude-fixture",
            sidecar_available=True,
        )
        output = []
        for event in document["events"]:
            audit = audit_responses_event(event, mode="compatibility")
            self.assertEqual(audit.warnings, [], event["type"])
            output.extend(translator.process(event["type"], event))

        self.assertFalse(translator.protocol_failed)
        self.assertIsNotNone(translator.terminal_result)
        expected = document["expected"]
        names = [name for name, _ in output]
        self.assertEqual(names[-2:], expected["event_suffix"])
        self.assertEqual(
            [
                event["content_block"]["type"]
                for name, event in output
                if name == "content_block_start"
            ],
            expected["content_block_types"],
        )
        terminal = translator.terminal_result
        self.assertEqual(terminal.response["content"][0]["text"], expected["text"])
        self.assertEqual(terminal.response["content"][1]["name"], expected["tool_name"])
        self.assertEqual(terminal.response["content"][1]["input"], expected["tool_input"])
        self.assertEqual(terminal.response["stop_reason"], expected["stop_reason"])
        self.assertEqual(
            [item["type"] for item in terminal.replay_items],
            expected["replay_item_types"],
        )
        for key, value in expected["usage"].items():
            self.assertEqual(terminal.response["usage"][key], value)


if __name__ == "__main__":
    unittest.main()
