#!/usr/bin/env python3
"""Generate structurally complete, content-free Anthropic/Responses fixtures.

The source dumps contain real prompts, tool output, credentials, host metadata,
and opaque model state.  This script never copies those values.  It first
sanitizes each protocol object, then builds a small greedy set cover over JSON
path/type pairs, object key shapes, and discriminator values.

The resulting fixtures are structural catalogs, not replay transcripts.  Each
sample records the protocol root it covers (headers, request, event, ...), so
the accompanying manifest can prove that every observed structural token is
represented without retaining the original conversation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


FIXTURE_SCHEMA = "ghc-anthropic-responses-structural-fixture-v1"
MANIFEST_SCHEMA = "ghc-anthropic-responses-coverage-manifest-v1"

CLAUDE_DUMP = "2026-07-12.jl"
GPT_DUMP = "2026-07-10.jl"
TARGET_GPT_MODEL = "gpt-5.6-sol"
# Hand-authored, fully synthetic output-oracle fixture. It lives beside the
# generated structural catalogs but must survive a normal regeneration.
PRESERVED_FIXTURE_NAMES = {"coherent_stream.json"}

SENSITIVE_HEADER_NAMES = {"authorization", "proxy-authorization", "cookie", "set-cookie"}

DISCRIMINATOR_KEYS = {
    "type",
    "role",
    "status",
    "stop_reason",
    "phase",
    "object",
    "model",
    "tool_choice",
    "effort",
    "context",
    "mode",
    "verbosity",
    "service_tier",
    "truncation",
    "prompt_cache_retention",
    "namespace",
    "code",
    "token_type",
    "format",
    "keep",
}

CONFIG_VALUE_KEYS = {
    "stream",
    "store",
    "parallel_tool_calls",
    "is_error",
    "temperature",
    "top_p",
    "max_tokens",
    "max_output_tokens",
    "budget_tokens",
    "strict",
}

SCHEMA_NUMERIC_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
}

# These names are public protocol/built-in names and useful for detecting the
# concrete 2.1.197 -> 2.1.207 tool-contract drift.  Environment-specific MCP
# names are deterministically replaced below.
SAFE_TOOL_NAMES = {
    "Agent",
    "AskUserQuestion",
    "Bash",
    "CronCreate",
    "CronDelete",
    "CronList",
    "Edit",
    "EnterPlanMode",
    "EnterWorktree",
    "ExitPlanMode",
    "ExitWorktree",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Skill",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "WebFetch",
    "WebSearch",
    "Workflow",
    "Write",
    "exec",
    "wait",
    "request_user_input",
    "collaboration",
    "followup_task",
    "interrupt_agent",
    "list_agents",
    "send_message",
    "spawn_agent",
    "wait_agent",
}

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IPV4_RE = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
WINDOWS_PATH_RE = re.compile(r"(?i)(?:[a-z]:\\|\\\\[a-z0-9_.-]+\\)")
AUTH_VALUE_RE = re.compile(r"(?i)\b(?:bearer|basic|token)\s+[a-z0-9._~+/=-]{8,}")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _escape_pointer(component: str) -> str:
    return component.replace("~", "~0").replace("/", "~1")


def _pointer(root: str, components: Sequence[str]) -> str:
    if not components:
        return root
    return root + "/" + "/".join(_escape_pointer(part) for part in components)


def _canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_document(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class Observation:
    paths: MutableMapping[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    shapes: MutableMapping[str, Set[Tuple[str, ...]]] = field(default_factory=lambda: defaultdict(set))
    discriminators: MutableMapping[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def merge(self, other: "Observation") -> None:
        for path, values in other.paths.items():
            self.paths[path].update(values)
        for path, values in other.shapes.items():
            self.shapes[path].update(values)
        for path, values in other.discriminators.items():
            self.discriminators[path].update(values)

    def tokens(self) -> Set[Tuple[Any, ...]]:
        result: Set[Tuple[Any, ...]] = set()
        for path, types in self.paths.items():
            result.update(("path", path, value_type) for value_type in types)
        for path, shapes in self.shapes.items():
            result.update(("shape", path, shape) for shape in shapes)
        for path, values in self.discriminators.items():
            result.update(("discriminator", path, value) for value in values)
        return result


def _is_discriminator(components: Sequence[str], key: Optional[str], root: str) -> bool:
    if root == "headers":
        return True
    if key in DISCRIMINATOR_KEYS or key in CONFIG_VALUE_KEYS:
        return True
    if key == "name" and "properties" not in components:
        return True
    if components and components[-1] in {"enum", "required", "include"}:
        return True
    if key == "const":
        return True
    return False


def observe_value(value: Any, root: str) -> Observation:
    """Return normalized JSON paths, object shapes, and discriminator values."""

    observed = Observation()

    def walk(current: Any, components: Tuple[str, ...], key: Optional[str] = None) -> None:
        path = _pointer(root, components)
        observed.paths[path].add(_json_type(current))

        if _is_discriminator(components[:-1], key, root) and not isinstance(current, (dict, list)):
            observed.discriminators[path].add(_canonical_value(current))

        if isinstance(current, dict):
            observed.shapes[path].add(tuple(sorted(str(k) for k in current)))
            for child_key, child_value in current.items():
                walk(child_value, components + (str(child_key),), str(child_key))
        elif isinstance(current, list):
            for child in current:
                walk(child, components + ("[]",), key)

    walk(value, ())
    return observed


class Sanitizer:
    """Value sanitizer that preserves protocol contracts but no source text."""

    def __init__(self) -> None:
        self._tool_names: Dict[str, str] = {}

    def tool_name(self, value: str) -> str:
        if value in SAFE_TOOL_NAMES:
            return value
        if value not in self._tool_names:
            self._tool_names[value] = f"fixture_tool_{len(self._tool_names) + 1:03d}"
        return self._tool_names[value]

    def headers(self, headers: Mapping[str, Any], profile: str) -> Tuple[Dict[str, Any], int]:
        result: Dict[str, Any] = {}
        removed = 0
        for key, value in headers.items():
            lower = key.lower()
            if lower in SENSITIVE_HEADER_NAMES:
                removed += 1
                continue
            if not isinstance(value, str):
                result[key] = self.value(value, ("headers", key))
                continue

            if lower == "user-agent":
                if profile.startswith("claude-cli/"):
                    result[key] = f"{profile} (fixture)"
                else:
                    result[key] = "codex-client/fixture"
            elif lower in {"anthropic-beta", "anthropic-version", "content-type", "accept", "connection", "accept-encoding", "x-app"}:
                result[key] = value
            elif lower == "x-stainless-package-version":
                result[key] = value
            elif lower in {
                "x-stainless-arch",
                "x-stainless-os",
                "x-stainless-runtime",
                "x-stainless-runtime-version",
                "x-stainless-lang",
            }:
                result[key] = "<device>"
            elif lower == "content-length":
                result[key] = "0"
            elif lower == "host":
                result[key] = "<host>"
            elif "session" in lower or "thread" in lower or "window" in lower or lower.endswith("-id"):
                result[key] = "<id>"
            elif lower == "x-codex-turn-metadata":
                result[key] = "<metadata>"
            else:
                result[key] = "<header>"
        return result, removed

    def value(self, value: Any, path: Tuple[str, ...] = (), parent: Optional[Mapping[str, Any]] = None) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            key = path[-1] if path else ""
            if self._in_schema(path) and key in SCHEMA_NUMERIC_KEYS:
                return value
            if key in {"max_tokens", "max_output_tokens", "budget_tokens", "temperature", "top_p"}:
                return value
            return 0 if value == 0 else 1
        if isinstance(value, str):
            return self._string(value, path, parent)
        if isinstance(value, list):
            return [self.value(item, path + ("[]",), parent=None) for item in value]
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, child in value.items():
                if path == ("headers",) and key.lower() in SENSITIVE_HEADER_NAMES:
                    continue
                # Tool input keys are part of the observed protocol shape and
                # therefore remain visible; their values still pass through the
                # same content sanitizer as every other subtree.
                result[key] = self.value(child, path + (str(key),), parent=value)
            return result
        raise TypeError(f"unsupported value at {path}: {type(value)!r}")

    @staticmethod
    def _in_schema(path: Sequence[str]) -> bool:
        return any(part in {"input_schema", "parameters", "json_schema"} for part in path)

    def _string(self, value: str, path: Tuple[str, ...], parent: Optional[Mapping[str, Any]]) -> str:
        key = path[-1] if path else ""
        parent_type = parent.get("type") if isinstance(parent, Mapping) else None
        in_schema = self._in_schema(path)

        if key == "description":
            return "<description>"
        if key in {"text", "thinking"}:
            return "<text>"
        if key == "content":
            return "<content>"
        if key in {"message", "prompt"}:
            return "<message>"
        if key in {"signature", "encrypted_content", "data"}:
            return "<opaque>"
        if key in {"arguments", "partial_json"}:
            return "{}"
        if key in {"delta", "input", "output"}:
            return f"<{key}>"
        if key in {"syntax", "definition"}:
            return "<grammar>"

        if key in {"call_id", "tool_use_id"}:
            return "<call_id>"
        if key == "item_id":
            return "<item_id>"
        if key == "id":
            if parent_type in {"tool_use", "function_call", "custom_tool_call"}:
                return "<call_id>"
            return "<id>"
        if key in {"previous_response_id", "safety_identifier", "prompt_cache_key"}:
            return "<id>"

        lower_key = key.lower()
        if any(token in lower_key for token in ("user", "session", "thread", "turn", "window", "installation")):
            return "<identity>"
        if key in {"author", "recipient"}:
            return "<agent>"

        if key == "name" and not in_schema:
            return self.tool_name(value)
        if key == "model":
            return value
        if key in DISCRIMINATOR_KEYS:
            return value
        if key in {"include", "$schema", "format", "pattern"}:
            return value

        if in_schema:
            if len(path) >= 2 and path[-2] == "required":
                return value
            if len(path) >= 2 and path[-2] == "enum":
                return value if re.fullmatch(r"[A-Za-z0-9_.:/-]{1,80}", value) else "<enum>"
            if key == "const":
                return value if re.fullmatch(r"[A-Za-z0-9_.:/-]{1,80}", value) else "<const>"
            if key == "default":
                return "<default>"

        return "<string>"


@dataclass(frozen=True)
class Candidate:
    root: str
    value: Any
    record: int
    event_index: Optional[int] = None

    def label(self) -> Tuple[Any, ...]:
        return (self.root, self.record, -1 if self.event_index is None else self.event_index)


@dataclass
class ProfileData:
    key: str
    source_file: str
    records: Set[int] = field(default_factory=set)
    candidates: MutableMapping[str, List[Candidate]] = field(default_factory=lambda: defaultdict(list))
    observed: MutableMapping[str, Observation] = field(default_factory=lambda: defaultdict(Observation))
    stats: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))
    redactions: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, candidate: Candidate) -> None:
        self.candidates[candidate.root].append(candidate)
        self.observed[candidate.root].merge(observe_value(candidate.value, candidate.root))


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, value


def _parse_raw_event(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    payload = raw.strip()
    if payload.startswith("data:"):
        payload = payload[5:].strip()
    if payload == "[DONE]":
        return None
    value = json.loads(payload)
    return value if isinstance(value, dict) else None


def collect_profiles(source_root: Path) -> Dict[str, ProfileData]:
    sanitizer = Sanitizer()
    profiles: Dict[str, ProfileData] = {
        "claude_cli_2_1_197": ProfileData("claude_cli_2_1_197", CLAUDE_DUMP),
        "claude_cli_2_1_207": ProfileData("claude_cli_2_1_207", CLAUDE_DUMP),
        "gpt_5_6_responses": ProfileData("gpt_5_6_responses", GPT_DUMP),
    }

    claude_path = source_root / CLAUDE_DUMP
    gpt_path = source_root / GPT_DUMP
    if not claude_path.is_file() or not gpt_path.is_file():
        missing = [str(path) for path in (claude_path, gpt_path) if not path.is_file()]
        raise FileNotFoundError("missing source dump(s): " + ", ".join(missing))

    for record_number, record in _iter_jsonl(claude_path):
        headers = record.get("request_headers") or {}
        user_agent = str(headers.get("User-Agent", ""))
        match = re.search(r"claude-cli/(\d+\.\d+\.\d+)", user_agent)
        if not match:
            continue
        version = match.group(1)
        key = "claude_cli_" + version.replace(".", "_")
        if key not in profiles:
            continue
        profile = profiles[key]
        profile.records.add(record_number)
        profile.stats["records"] += 1

        clean_headers, removed = sanitizer.headers(headers, f"claude-cli/{version}")
        profile.redactions["authorization_headers_removed"] += removed
        profile.add(Candidate("headers", clean_headers, record_number))

        original = record.get("original_request_body")
        if isinstance(original, dict):
            profile.add(Candidate("original_request", sanitizer.value(original), record_number))
        forwarded = record.get("request_body")
        if isinstance(forwarded, dict):
            profile.add(Candidate("forwarded_request", sanitizer.value(forwarded), record_number))

        response_body = record.get("response_body")
        if isinstance(response_body, dict):
            profile.add(Candidate("error_response", sanitizer.value(response_body), record_number))

        for event_index, raw_event in enumerate(record.get("raw_events") or []):
            event = _parse_raw_event(raw_event)
            if event is None:
                profile.stats["unparsed_events"] += 1
                continue
            profile.stats["events"] += 1
            profile.add(Candidate("event", sanitizer.value(event), record_number, event_index))

    gpt_profile = profiles["gpt_5_6_responses"]
    for record_number, record in _iter_jsonl(gpt_path):
        if record.get("model") != TARGET_GPT_MODEL:
            continue
        gpt_profile.records.add(record_number)
        gpt_profile.stats["records"] += 1

        headers = record.get("request_headers") or {}
        clean_headers, removed = sanitizer.headers(headers, "gpt-5.6-codex")
        gpt_profile.redactions["authorization_headers_removed"] += removed
        gpt_profile.add(Candidate("headers", clean_headers, record_number))

        original = record.get("original_request_body")
        if isinstance(original, dict) and original.get("_truncated"):
            gpt_profile.stats["truncated_requests"] += 1
        elif isinstance(original, dict):
            gpt_profile.stats["complete_requests"] += 1
            gpt_profile.add(Candidate("request", sanitizer.value(original), record_number))

        response_body = record.get("response_body")
        if isinstance(response_body, dict):
            gpt_profile.stats["error_responses"] += 1
            gpt_profile.add(Candidate("error_response", sanitizer.value(response_body), record_number))

        for event_index, raw_event in enumerate(record.get("raw_events") or []):
            event = _parse_raw_event(raw_event)
            if event is None:
                gpt_profile.stats["unparsed_events"] += 1
                continue
            if event.get("_truncated"):
                gpt_profile.stats["truncated_events"] += 1
                continue
            gpt_profile.stats["events"] += 1
            gpt_profile.add(Candidate("event", sanitizer.value(event), record_number, event_index))

    return profiles


def _deduplicate_candidates(candidates: Iterable[Candidate]) -> List[Candidate]:
    unique: Dict[str, Candidate] = {}
    for candidate in candidates:
        canonical = _canonical_document(candidate.value)
        unique.setdefault(canonical, candidate)
    return sorted(unique.values(), key=lambda item: item.label())


def greedy_cover(candidates: Iterable[Candidate], universe: Set[Tuple[Any, ...]]) -> List[Candidate]:
    """Pick a deterministic small set of samples covering every token."""

    remaining = set(universe)
    pool = _deduplicate_candidates(candidates)
    selected: List[Candidate] = []
    token_cache = {id(item): observe_value(item.value, item.root).tokens() for item in pool}

    while remaining:
        ranked = []
        for candidate in pool:
            if candidate in selected:
                continue
            gain = len(token_cache[id(candidate)] & remaining)
            if gain:
                ranked.append((gain, -len(_canonical_document(candidate.value)), candidate))
        if not ranked:
            preview = sorted(repr(token) for token in remaining)[:10]
            raise RuntimeError(f"fixture set-cover stalled; missing {preview}")
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2].label()))
        best = ranked[0][2]
        selected.append(best)
        remaining.difference_update(token_cache[id(best)])

    return sorted(selected, key=lambda item: item.label())


def observe_fixture_document(document: Mapping[str, Any]) -> Dict[str, Observation]:
    result: Dict[str, Observation] = defaultdict(Observation)
    for sample in document.get("samples") or []:
        root = sample["root"]
        result[root].merge(observe_value(sample["value"], root))
    return dict(result)


def _manifest_for_root(observed: Observation, covered: Observation) -> Dict[str, Any]:
    path_entries = []
    for path in sorted(observed.paths):
        expected = sorted(observed.paths[path])
        actual = sorted(covered.paths.get(path, set()))
        missing = sorted(set(expected) - set(actual))
        path_entries.append(
            {"path": path, "types": expected, "covered_types": actual, "missing_types": missing}
        )

    shape_entries = []
    for path in sorted(observed.shapes):
        for shape in sorted(observed.shapes[path]):
            shape_entries.append(
                {
                    "path": path,
                    "keys": list(shape),
                    "covered": shape in covered.shapes.get(path, set()),
                }
            )

    discriminator_entries = []
    for path in sorted(observed.discriminators):
        expected_raw = sorted(observed.discriminators[path])
        covered_raw = covered.discriminators.get(path, set())
        discriminator_entries.append(
            {
                "path": path,
                "values": [json.loads(item) for item in expected_raw],
                "covered_values": [json.loads(item) for item in expected_raw if item in covered_raw],
                "missing_values": [json.loads(item) for item in expected_raw if item not in covered_raw],
            }
        )

    missing_count = sum(len(item["missing_types"]) for item in path_entries)
    missing_count += sum(not item["covered"] for item in shape_entries)
    missing_count += sum(len(item["missing_values"]) for item in discriminator_entries)
    return {
        "summary": {
            "observed_tokens": len(observed.tokens()),
            "covered_tokens": len(observed.tokens() & covered.tokens()),
            "missing_tokens": missing_count,
        },
        "paths": path_entries,
        "object_shapes": shape_entries,
        "discriminators": discriminator_entries,
    }


def build_outputs(source_root: Path) -> Dict[str, str]:
    profiles = collect_profiles(source_root)
    output_documents: Dict[str, Dict[str, Any]] = {}
    manifest_profiles: Dict[str, Any] = {}

    fixture_names = {
        "claude_cli_2_1_197": "claude_cli_2_1_197.json",
        "claude_cli_2_1_207": "claude_cli_2_1_207.json",
        "gpt_5_6_responses": "gpt_5_6_responses.json",
    }

    for profile_key, profile in profiles.items():
        selected: List[Candidate] = []
        for root in sorted(profile.observed):
            selected.extend(greedy_cover(profile.candidates[root], profile.observed[root].tokens()))
        selected.sort(key=lambda item: item.label())

        document = {
            "fixture_schema": FIXTURE_SCHEMA,
            "profile": profile_key,
            "source": profile.source_file,
            "source_records": sorted(profile.records),
            "samples": [
                {
                    "root": item.root,
                    "source_record": item.record,
                    **({"source_event_index": item.event_index} if item.event_index is not None else {}),
                    "value": item.value,
                }
                for item in selected
            ],
        }
        output_documents[profile_key] = document
        covered = observe_fixture_document(document)

        roots_manifest = {
            root: _manifest_for_root(profile.observed[root], covered.get(root, Observation()))
            for root in sorted(profile.observed)
        }
        manifest_profiles[profile_key] = {
            "fixture": fixture_names[profile_key],
            "source": profile.source_file,
            "source_records": sorted(profile.records),
            "source_stats": dict(sorted(profile.stats.items())),
            "redactions": dict(sorted(profile.redactions.items())),
            "selected_samples": len(selected),
            "all_covered": all(root["summary"]["missing_tokens"] == 0 for root in roots_manifest.values()),
            "roots": roots_manifest,
        }

    manifest = {
        "manifest_schema": MANIFEST_SCHEMA,
        "fixture_schema": FIXTURE_SCHEMA,
        "profiles": manifest_profiles,
        "scope": {
            "included": [
                "sanitized request headers",
                "original and forwarded Claude request bodies",
                "complete GPT-5.6 request bodies",
                "parsed Claude and GPT stream event payloads",
                "non-stream error response bodies",
            ],
            "excluded": [
                "dump envelope identifiers, timestamps, client IPs, and accounting fields",
                "authorization headers",
                "dump truncation sentinels (counts retained in source_stats)",
            ],
        },
    }

    result = {
        fixture_names[key]: json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        for key, document in output_documents.items()
    }
    result["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return result


def audit_sanitized_fixture(document: Mapping[str, Any]) -> List[str]:
    """Return human-readable violations if a fixture contains source-like data."""

    violations: List[str] = []

    def walk(value: Any, path: Tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in SENSITIVE_HEADER_NAMES:
                    violations.append(f"{_pointer('fixture', path + (key,))}: sensitive header retained")
                if key == "description" and isinstance(child, str) and child != "<description>":
                    violations.append(f"{_pointer('fixture', path + (key,))}: description not redacted")
                if key in {"text", "thinking"} and isinstance(child, str) and child != "<text>":
                    violations.append(f"{_pointer('fixture', path + (key,))}: text not redacted")
                if key in {"signature", "encrypted_content", "data"} and isinstance(child, str) and child != "<opaque>":
                    violations.append(f"{_pointer('fixture', path + (key,))}: opaque content not redacted")
                walk(child, path + (str(key),))
        elif isinstance(value, list):
            for child in value:
                walk(child, path + ("[]",))
        elif isinstance(value, str):
            location = _pointer("fixture", path)
            if UUID_RE.search(value):
                violations.append(f"{location}: UUID-like value retained")
            if IPV4_RE.search(value):
                violations.append(f"{location}: IP-like value retained")
            if WINDOWS_PATH_RE.search(value):
                violations.append(f"{location}: device path retained")
            if AUTH_VALUE_RE.search(value):
                violations.append(f"{location}: credential-like value retained")

    walk(document, ())
    return violations


def _write_or_check(outputs: Mapping[str, str], output_dir: Path, check: bool) -> int:
    if check:
        mismatches = []
        for name, expected in outputs.items():
            path = output_dir / name
            actual = path.read_text(encoding="utf-8") if path.is_file() else None
            if actual != expected:
                mismatches.append(name)
        if mismatches:
            print("fixture files are stale or missing: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = set(outputs)
    for existing in output_dir.glob("*.json"):
        if (
            existing.name not in expected_names
            and existing.name not in PRESERVED_FIXTURE_NAMES
        ):
            existing.unlink()
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8", newline="\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "tests" / "fixtures" / "anthropic_responses",
    )
    parser.add_argument("--check", action="store_true", help="fail if checked-in fixtures differ")
    args = parser.parse_args(argv)

    outputs = build_outputs(args.source_root.resolve())
    return _write_or_check(outputs, args.output_dir.resolve(), args.check)


if __name__ == "__main__":
    raise SystemExit(main())
