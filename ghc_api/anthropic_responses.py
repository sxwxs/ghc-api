"""Loss-aware Anthropic Messages <-> OpenAI Responses translation.

This module intentionally does not depend on Flask or the transport layer.  It
turns an Anthropic request into ordered Responses input items and converts a
terminal Responses object back into an Anthropic message.  Every source leaf is
accounted for in :class:`ConversionReport`; unsupported data is never silently
dropped.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .reasoning_carrier import (
    build_reasoning_carrier,
    is_reasoning_carrier,
    parse_reasoning_carrier,
)


PRESERVATION_EXACT = "exact"
PRESERVATION_SEMANTIC = "semantic_encoding"
PRESERVATION_SIDECAR = "sidecar"
PRESERVATION_APPROXIMATION = "approximation"
PRESERVATION_UNSUPPORTED = "unsupported"

MODE_COMPATIBILITY = "compatibility"
MODE_LOSSLESS_REQUIRED = "lossless_required"
VALID_MODES = {MODE_COMPATIBILITY, MODE_LOSSLESS_REQUIRED}


class AnthropicResponsesConversionError(ValueError):
    """Raised when lossless mode encounters an unrepresentable source field."""

    def __init__(self, message: str, report: "ConversionReport") -> None:
        super().__init__(message)
        self.report = report


class StrictJSONError(ValueError):
    """Raised when a lossless request is not strict, unambiguous JSON."""


def parse_strict_json_bytes(raw: bytes) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictJSONError(f"Request body is not valid UTF-8 at byte {exc.start}") from exc

    def reject_constant(value: str) -> None:
        raise StrictJSONError(f"Non-finite JSON number is not allowed: {value}")

    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJSONError(f"Duplicate JSON object key: {key}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    try:
        value, end = decoder.raw_decode(text)
    except StrictJSONError:
        raise
    except RecursionError as exc:
        raise StrictJSONError("JSON nesting is too deep") from exc
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"Invalid JSON at character {exc.pos}: {exc.msg}") from exc
    if text[end:].strip():
        raise StrictJSONError(f"Trailing data after JSON value at character {end}")

    # Walk iteratively so the validation boundary cannot itself overflow the
    # Python stack after the decoder accepted a deeply nested document.
    stack: List[Tuple[Any, str]] = [(value, "$")]
    while stack:
        node, path = stack.pop()
        if isinstance(node, float) and not math.isfinite(node):
            raise StrictJSONError(f"Non-finite JSON number is not allowed at {path}")
        if isinstance(node, str):
            try:
                node.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise StrictJSONError(
                    f"JSON string contains an unpaired surrogate at {path}"
                ) from exc
        elif isinstance(node, dict):
            for key, child in node.items():
                stack.append((child, path + ".<value>"))
                stack.append((key, path + ".<key>"))
        elif isinstance(node, list):
            for index in range(len(node) - 1, -1, -1):
                stack.append((node[index], f"{path}[{index}]"))
    return value


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _join_pointer(base: str, component: Any) -> str:
    escaped = _pointer_escape(str(component))
    return f"{base}/{escaped}" if base else f"/{escaped}"


def iter_json_leaf_paths(value: Any, path: str = "") -> Iterable[str]:
    """Yield RFC-6901-ish paths for leaves and empty containers."""
    if isinstance(value, dict):
        if not value:
            yield path or "/"
            return
        for key, child in value.items():
            yield from iter_json_leaf_paths(child, _join_pointer(path, key))
        return
    if isinstance(value, list):
        if not value:
            yield path or "/"
            return
        for index, child in enumerate(value):
            yield from iter_json_leaf_paths(child, _join_pointer(path, index))
        return
    yield path or "/"


@dataclass
class PreservationRecord:
    source_path: str
    disposition: str
    target_path: Optional[str] = None
    detail: Optional[str] = None
    subtree: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "source_path": self.source_path,
            "disposition": self.disposition,
        }
        if self.target_path is not None:
            result["target_path"] = self.target_path
        if self.detail:
            result["detail"] = self.detail
        if self.subtree:
            result["subtree"] = True
        return result


@dataclass
class ConversionReport:
    """Audit trail for one direction of a protocol conversion.

    ``sidecar`` is a historical disposition name meaning that a source field
    is not represented on the client wire. Compatibility mode records it
    silently; ``lossless_required`` rejects it.
    """

    direction: str
    records: List[PreservationRecord] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    unaccounted_paths: List[str] = field(default_factory=list)
    _marked_paths: Set[str] = field(default_factory=set, repr=False)
    _marked_subtrees: Set[str] = field(default_factory=set, repr=False)
    _seen_records: Set[Tuple[Any, ...]] = field(default_factory=set, repr=False)

    def mark(
        self,
        source_path: str,
        disposition: str,
        target_path: Optional[str] = None,
        detail: Optional[str] = None,
        subtree: bool = False,
    ) -> None:
        if not source_path:
            source_path = "/"
        key = (source_path, disposition, target_path, detail, subtree)
        if key in self._seen_records:
            return
        self._seen_records.add(key)
        self.records.append(PreservationRecord(
            source_path=source_path,
            disposition=disposition,
            target_path=target_path,
            detail=detail,
            subtree=subtree,
        ))
        self._marked_paths.add(source_path)
        if subtree:
            self._marked_subtrees.add(source_path.rstrip("/"))
        if disposition in (PRESERVATION_APPROXIMATION, PRESERVATION_UNSUPPORTED):
            warning = {
                "code": "conversion.approximation" if disposition == PRESERVATION_APPROXIMATION else "conversion.unsupported",
                "path": source_path,
                "action": disposition,
            }
            if detail:
                warning["detail"] = detail
            if warning not in self.warnings:
                self.warnings.append(warning)

    def _is_accounted(self, path: str) -> bool:
        if path in self._marked_paths:
            return True
        if "" in self._marked_subtrees:
            return True
        candidate = path.rstrip("/")
        while candidate:
            if candidate in self._marked_subtrees:
                return True
            separator = candidate.rfind("/")
            if separator <= 0:
                break
            candidate = candidate[:separator]
        return False

    def account_unknown_paths(self, source: Any) -> None:
        self.unaccounted_paths = []
        for path in iter_json_leaf_paths(source):
            if not self._is_accounted(path):
                self.unaccounted_paths.append(path)
                self.mark(
                    path,
                    PRESERVATION_UNSUPPORTED,
                    detail="No registered conversion rule for this source path",
                )

    def require_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown Anthropic/Responses compatibility mode: {mode}")
        if mode != MODE_LOSSLESS_REQUIRED:
            return
        lossy = [
            record for record in self.records
            if record.disposition in (
                PRESERVATION_APPROXIMATION,
                PRESERVATION_UNSUPPORTED,
                PRESERVATION_SIDECAR,
            )
        ]
        if lossy:
            paths = ", ".join(record.source_path for record in lossy[:5])
            if len(lossy) > 5:
                paths += f", ... ({len(lossy)} total)"
            raise AnthropicResponsesConversionError(
                f"Request cannot be represented losslessly: {paths}", self
            )

    def finalize(self, source: Any, mode: str) -> None:
        self.account_unknown_paths(source)
        self.require_mode(mode)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "records": [record.to_dict() for record in self.records],
            "warnings": copy.deepcopy(self.warnings),
            "unaccounted_paths": list(self.unaccounted_paths),
        }


@dataclass(frozen=True)
class ResponsesWireProfile:
    name: str
    tools_in_input: bool
    supports_native_web_search: bool
    native_server_tools_in_input: bool
    supports_prompt_cache_breakpoint: bool
    supports_temperature: bool
    supports_top_p: bool
    supports_max_output_tokens: bool
    supports_message_phase: bool
    supports_reasoning_context: bool
    reasoning_efforts: Tuple[str, ...]
    default_text_verbosity: Optional[str] = None
    # Some backends re-encrypt every identifier per SSE event, so the same
    # logical response/item carries a different opaque id in each frame.
    stable_ids: bool = True


WIRE_PROFILES: Dict[str, ResponsesWireProfile] = {
    "public_responses": ResponsesWireProfile(
        name="public_responses",
        tools_in_input=False,
        supports_native_web_search=True,
        native_server_tools_in_input=False,
        supports_prompt_cache_breakpoint=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_output_tokens=True,
        supports_message_phase=False,
        supports_reasoning_context=False,
        reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh"),
    ),
    "copilot_responses_lite": ResponsesWireProfile(
        name="copilot_responses_lite",
        tools_in_input=True,
        supports_native_web_search=True,
        # The live backend accepts native server tools only in top-level tools;
        # placing web_search inside additional_tools silently removes it.
        native_server_tools_in_input=False,
        # The supplied dump proves prompt_cache_key but not explicit breakpoints.
        supports_prompt_cache_breakpoint=False,
        supports_temperature=False,
        supports_top_p=False,
        supports_max_output_tokens=True,
        supports_message_phase=True,
        supports_reasoning_context=True,
        reasoning_efforts=("none", "low", "medium", "high", "xhigh", "max"),
        default_text_verbosity="low",
        # Observed live: response.created / response.in_progress /
        # response.completed each carry a different encrypted response id, and
        # every item event carries a freshly encrypted item id.
        stable_ids=False,
    ),
}


def get_wire_profile(name: str) -> ResponsesWireProfile:
    try:
        return WIRE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Responses wire profile: {name}") from exc


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class IdentifierCodec:
    """Request-scoped reversible codec for function names and call IDs."""

    def __init__(self, max_length: int = 64) -> None:
        self.max_length = max_length
        self._encoded_to_original: Dict[str, str] = {}
        self._original_to_encoded: Dict[str, str] = {}

    def _hashed_candidate(self, value: str, kind: str, attempt: int) -> str:
        digest_source = value if attempt == 0 else f"{value}\x00{attempt}"
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_")
        prefix = "ghc_call_" if kind == "call" else "ghc_tool_"
        room = max(0, self.max_length - len(prefix) - len(digest) - 1)
        candidate = (
            f"{prefix}{safe[:room]}_{digest}"
            if room
            else f"{prefix}{digest}"
        )
        return candidate[:self.max_length]

    def encode(self, value: str, kind: str = "id") -> str:
        value = str(value or "")
        if value in self._original_to_encoded:
            return self._original_to_encoded[value]

        direct = (
            value
            if value
            and len(value) <= self.max_length
            and _IDENTIFIER_RE.fullmatch(value)
            else None
        )
        if direct is not None and direct not in self._encoded_to_original:
            encoded = direct
        else:
            encoded = ""
            # A client can deliberately choose a valid identifier equal to the
            # hashed representation of another value. Keep the codec injective
            # instead of silently aliasing two tools or call IDs.
            for attempt in range(1024):
                candidate = self._hashed_candidate(value, kind, attempt)
                owner = self._encoded_to_original.get(candidate)
                if owner is None or owner == value:
                    encoded = candidate
                    break
            if not encoded:
                raise ValueError("Unable to allocate a unique encoded identifier")

        self._encoded_to_original[encoded] = value
        self._original_to_encoded[value] = encoded
        return encoded

    def decode(self, value: str) -> str:
        return self._encoded_to_original.get(value, value)

@dataclass
class AnthropicToResponsesResult:
    payload: Dict[str, Any]
    report: ConversionReport
    name_codec: IdentifierCodec
    call_id_codec: IdentifierCodec
    stop_sequences: List[str]
    wire_profile: str


@dataclass
class ResponsesToAnthropicResult:
    response: Dict[str, Any]
    report: ConversionReport
    matched_stop_sequence: Optional[str] = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _data_url(media_type: str, data: str) -> str:
    return f"data:{media_type or 'application/octet-stream'};base64,{data}"


def _cache_control_to_part(
    cache_control: Any,
    target_part: Dict[str, Any],
    report: ConversionReport,
    source_path: str,
    target_path: str,
    profile: ResponsesWireProfile,
) -> None:
    if not isinstance(cache_control, dict):
        report.mark(source_path, PRESERVATION_UNSUPPORTED, detail="cache_control must be an object", subtree=True)
        return
    if profile.supports_prompt_cache_breakpoint and cache_control.get("type") == "ephemeral":
        target_part["prompt_cache_breakpoint"] = {"mode": "explicit"}
        disposition = PRESERVATION_SEMANTIC
        detail = "Cache boundary preserved; provider TTL/scope semantics may differ"
        if cache_control.get("ttl") is not None or cache_control.get("scope") is not None:
            disposition = PRESERVATION_APPROXIMATION
        report.mark(source_path, disposition, target_path + "/prompt_cache_breakpoint", detail, subtree=True)
    else:
        report.mark(
            source_path,
            PRESERVATION_SIDECAR,
            detail=f"{profile.name} has no verified explicit cache breakpoint support",
            subtree=True,
        )


def _convert_image_block(
    block: Dict[str, Any],
    report: ConversionReport,
    path: str,
    target_path: str,
    profile: ResponsesWireProfile,
) -> Optional[Dict[str, Any]]:
    source = block.get("source")
    if not isinstance(source, dict):
        report.mark(_join_pointer(path, "source"), PRESERVATION_UNSUPPORTED, detail="image source must be an object", subtree=True)
        return None
    source_type = source.get("type")
    if source_type == "base64":
        part = {"type": "input_image", "image_url": _data_url(str(source.get("media_type") or "image/png"), str(source.get("data") or ""))}
    elif source_type == "url":
        part = {"type": "input_image", "image_url": str(source.get("url") or "")}
    else:
        report.mark(_join_pointer(path, "source"), PRESERVATION_UNSUPPORTED, detail=f"Unsupported image source type: {source_type}", subtree=True)
        return None
    report.mark(_join_pointer(path, "type"), PRESERVATION_EXACT, target_path + "/type")
    report.mark(_join_pointer(path, "source"), PRESERVATION_EXACT, target_path, subtree=True)
    if "cache_control" in block:
        _cache_control_to_part(block["cache_control"], part, report, _join_pointer(path, "cache_control"), target_path, profile)
    return part


def _convert_document_block(
    block: Dict[str, Any],
    report: ConversionReport,
    path: str,
    target_path: str,
    profile: ResponsesWireProfile,
) -> Optional[Dict[str, Any]]:
    source = block.get("source")
    if not isinstance(source, dict):
        report.mark(_join_pointer(path, "source"), PRESERVATION_UNSUPPORTED, detail="document source must be an object", subtree=True)
        return None
    source_type = source.get("type")
    part: Dict[str, Any] = {"type": "input_file"}
    if source_type == "base64":
        part["file_data"] = _data_url(str(source.get("media_type") or "application/pdf"), str(source.get("data") or ""))
    elif source_type == "url":
        part["file_url"] = str(source.get("url") or "")
    elif source_type == "text":
        # Public Responses has no byte-identical document-text part. Keep it as
        # input_text and record the semantic encoding.
        part = {"type": "input_text", "text": str(source.get("data") or source.get("text") or "")}
    else:
        report.mark(_join_pointer(path, "source"), PRESERVATION_UNSUPPORTED, detail=f"Unsupported document source type: {source_type}", subtree=True)
        return None
    report.mark(_join_pointer(path, "type"), PRESERVATION_SEMANTIC, target_path + "/type")
    report.mark(_join_pointer(path, "source"), PRESERVATION_EXACT, target_path, subtree=True)
    for key in ("title", "context", "citations"):
        if key in block:
            report.mark(_join_pointer(path, key), PRESERVATION_SIDECAR, detail=f"Anthropic document {key} has no verified Responses equivalent", subtree=True)
    if "cache_control" in block:
        _cache_control_to_part(block["cache_control"], part, report, _join_pointer(path, "cache_control"), target_path, profile)
    return part


def _convert_tool_result_output(
    block: Dict[str, Any],
    report: ConversionReport,
    path: str,
    profile: ResponsesWireProfile,
) -> Any:
    content = block.get("content", "")
    content_path = _join_pointer(path, "content")
    if block.get("is_error") is True:
        # Responses has no native tool-result error bit. Use a namespaced,
        # deterministic JSON envelope so the target model receives both the
        # error semantic and the complete original JSON value. This remains an
        # approximation (the caller marks is_error accordingly), but it avoids
        # silently presenting a failure as a successful tool result.
        report.mark(
            content_path,
            PRESERVATION_SEMANTIC,
            detail="Tool error content encoded in a reversible compatibility envelope",
            subtree=True,
        )
        return _canonical_json({
            "ghc_anthropic_tool_result": {
                "is_error": True,
                "content": copy.deepcopy(content),
            }
        })
    if isinstance(content, str):
        report.mark(content_path, PRESERVATION_EXACT, subtree=True)
        return content
    if not isinstance(content, list):
        report.mark(content_path, PRESERVATION_UNSUPPORTED, detail="tool_result content must be string or block array", subtree=True)
        return _canonical_json(content)
    output: List[Dict[str, Any]] = []
    for index, child in enumerate(content):
        child_path = _join_pointer(content_path, index)
        if not isinstance(child, dict):
            report.mark(child_path, PRESERVATION_UNSUPPORTED, detail="tool_result content entry must be an object", subtree=True)
            continue
        target_path = f"/output/{len(output)}"
        child_type = child.get("type")
        if child_type == "text":
            output.append({"type": "input_text", "text": str(child.get("text") or "")})
            report.mark(_join_pointer(child_path, "type"), PRESERVATION_SEMANTIC, target_path + "/type")
            report.mark(_join_pointer(child_path, "text"), PRESERVATION_EXACT, target_path + "/text")
            if "cache_control" in child:
                _cache_control_to_part(child["cache_control"], output[-1], report, _join_pointer(child_path, "cache_control"), target_path, profile)
        elif child_type == "image":
            part = _convert_image_block(child, report, child_path, target_path, profile)
            if part is not None:
                output.append(part)
        elif child_type == "document":
            part = _convert_document_block(child, report, child_path, target_path, profile)
            if part is not None:
                output.append(part)
        else:
            report.mark(child_path, PRESERVATION_UNSUPPORTED, detail=f"Unsupported tool_result block type: {child_type}", subtree=True)
    return output


def _append_message_items(
    input_items: List[Dict[str, Any]],
    message: Dict[str, Any],
    message_index: int,
    report: ConversionReport,
    profile: ResponsesWireProfile,
    name_codec: IdentifierCodec,
    call_id_codec: IdentifierCodec,
    target_model: str,
) -> None:
    base = f"/messages/{message_index}"
    role = message.get("role")
    report.mark(base + "/role", PRESERVATION_EXACT)
    content = message.get("content", "")
    blocks: List[Any]
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
        report.mark(base + "/content", PRESERVATION_EXACT)
    elif isinstance(content, list):
        blocks = content
    else:
        report.mark(base + "/content", PRESERVATION_UNSUPPORTED, detail="message content must be string or array", subtree=True)
        return

    has_tool_use = any(isinstance(block, dict) and block.get("type") == "tool_use" for block in blocks)
    current_parts: List[Dict[str, Any]] = []
    segment_number = 0

    def flush_message() -> None:
        nonlocal current_parts, segment_number
        if not current_parts:
            return
        item: Dict[str, Any] = {"type": "message", "role": role, "content": current_parts}
        if role == "assistant" and profile.supports_message_phase:
            item["phase"] = "commentary" if has_tool_use else "final_answer"
            report.mark(base, PRESERVATION_SEMANTIC, detail="Assistant phase inferred from visible Anthropic content")
        input_items.append(item)
        segment_number += 1
        current_parts = []

    for block_index, block in enumerate(blocks):
        path = f"{base}/content/{block_index}" if isinstance(content, list) else base + "/content"
        if not isinstance(block, dict):
            report.mark(path, PRESERVATION_UNSUPPORTED, detail="content block must be an object", subtree=True)
            continue
        block_type = block.get("type")
        target_path = f"/input/{len(input_items)}/content/{len(current_parts)}"
        if block_type == "text":
            part_type = "output_text" if role == "assistant" else "input_text"
            part = {"type": part_type, "text": str(block.get("text") or "")}
            current_parts.append(part)
            report.mark(_join_pointer(path, "type"), PRESERVATION_SEMANTIC, target_path + "/type")
            report.mark(_join_pointer(path, "text"), PRESERVATION_EXACT, target_path + "/text")
            if "cache_control" in block:
                _cache_control_to_part(block["cache_control"], part, report, _join_pointer(path, "cache_control"), target_path, profile)
        elif block_type == "image":
            part = _convert_image_block(block, report, path, target_path, profile)
            if part is not None:
                current_parts.append(part)
        elif block_type == "document":
            part = _convert_document_block(block, report, path, target_path, profile)
            if part is not None:
                current_parts.append(part)
        elif block_type == "tool_use":
            flush_message()
            original_name = str(block.get("name") or "")
            original_id = str(block.get("id") or "")
            encoded_name = name_codec.encode(original_name, "name")
            encoded_id = call_id_codec.encode(original_id, "call")
            arguments = block.get("input", {})
            try:
                argument_text = _canonical_json(arguments)
            except (TypeError, ValueError):
                argument_text = "{}"
                report.mark(_join_pointer(path, "input"), PRESERVATION_UNSUPPORTED, detail="tool input is not valid JSON", subtree=True)
            input_items.append({
                "type": "function_call",
                "call_id": encoded_id,
                "name": encoded_name,
                "arguments": argument_text,
            })
            report.mark(_join_pointer(path, "type"), PRESERVATION_SEMANTIC)
            report.mark(_join_pointer(path, "id"), PRESERVATION_EXACT if encoded_id == original_id else PRESERVATION_SEMANTIC)
            report.mark(_join_pointer(path, "name"), PRESERVATION_EXACT if encoded_name == original_name else PRESERVATION_SEMANTIC)
            if not any(record.source_path == _join_pointer(path, "input") for record in report.records):
                report.mark(_join_pointer(path, "input"), PRESERVATION_EXACT, subtree=True)
            if "cache_control" in block:
                report.mark(_join_pointer(path, "cache_control"), PRESERVATION_SIDECAR, detail="tool_use cache marker has no verified target", subtree=True)
        elif block_type == "tool_result":
            flush_message()
            original_id = str(block.get("tool_use_id") or "")
            encoded_id = call_id_codec.encode(original_id, "call")
            output = _convert_tool_result_output(block, report, path, profile)
            input_items.append({"type": "function_call_output", "call_id": encoded_id, "output": output})
            report.mark(_join_pointer(path, "type"), PRESERVATION_SEMANTIC)
            report.mark(_join_pointer(path, "tool_use_id"), PRESERVATION_EXACT if encoded_id == original_id else PRESERVATION_SEMANTIC)
            if "is_error" in block:
                if block.get("is_error") is False:
                    report.mark(_join_pointer(path, "is_error"), PRESERVATION_SEMANTIC, detail="False equals the Responses default success semantics")
                else:
                    report.mark(_join_pointer(path, "is_error"), PRESERVATION_APPROXIMATION, detail="Responses function output has no equivalent error flag")
            if "cache_control" in block:
                report.mark(_join_pointer(path, "cache_control"), PRESERVATION_SIDECAR, detail="outer tool_result cache marker is preserved in the report", subtree=True)
        elif block_type == "thinking" and role == "assistant":
            flush_message()
            signature = block.get("signature")
            if is_reasoning_carrier(signature):
                summary_text = str(block.get("thinking") or "")
                try:
                    carrier = parse_reasoning_carrier(signature)
                except ValueError:
                    carrier = None
                    report.mark(
                        path,
                        PRESERVATION_APPROXIMATION,
                        detail="Malformed Responses reasoning carrier was dropped; visible summary retained",
                        subtree=True,
                    )
                encrypted_content: Optional[str] = None
                carrier_item_id: Optional[str] = None
                if carrier is not None:
                    if carrier.model != target_model or carrier.wire_profile != profile.name:
                        report.mark(
                            path,
                            PRESERVATION_APPROXIMATION,
                            detail="Responses reasoning carrier belongs to a different model or wire profile; visible summary retained",
                            subtree=True,
                        )
                    else:
                        encrypted_content = carrier.encrypted_content
                        carrier_item_id = carrier.item_id
                        report.mark(
                            path,
                            PRESERVATION_SEMANTIC,
                            detail="Synthetic thinking block restored as a Responses reasoning item",
                            subtree=True,
                        )
                reasoning_item: Dict[str, Any] = {
                    "type": "reasoning",
                    "summary": (
                        [{"type": "summary_text", "text": summary_text}]
                        if summary_text else []
                    ),
                }
                if profile.name == "public_responses" and carrier_item_id:
                    reasoning_item["id"] = carrier_item_id
                if encrypted_content is not None:
                    reasoning_item["encrypted_content"] = encrypted_content
                input_items.append(reasoning_item)
            else:
                report.mark(
                    path,
                    PRESERVATION_APPROXIMATION,
                    detail="Provider-signed thinking is not portable to a Responses model",
                    subtree=True,
                )
        elif block_type == "redacted_thinking":
            report.mark(
                path,
                PRESERVATION_APPROXIMATION,
                detail="Provider-specific redacted thinking is not portable to a Responses model",
                subtree=True,
            )
        else:
            report.mark(path, PRESERVATION_UNSUPPORTED, detail=f"Unsupported content block type: {block_type}", subtree=True)
    flush_message()


def _convert_system(
    system: Any,
    input_items: List[Dict[str, Any]],
    report: ConversionReport,
    profile: ResponsesWireProfile,
) -> None:
    if isinstance(system, str):
        if system.startswith("x-anthropic-billing-header:"):
            report.mark(
                "/system",
                PRESERVATION_SEMANTIC,
                detail="Synthetic Anthropic billing metadata omitted from model input",
                subtree=True,
            )
            return
        input_items.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": system}]})
        report.mark("/system", PRESERVATION_EXACT, f"/input/{len(input_items)-1}/content/0/text")
        return
    if not isinstance(system, list):
        report.mark("/system", PRESERVATION_UNSUPPORTED, detail="system must be string or block array", subtree=True)
        return
    parts: List[Dict[str, Any]] = []
    item_index = len(input_items)
    for index, block in enumerate(system):
        path = f"/system/{index}"
        if not isinstance(block, dict) or block.get("type") != "text":
            report.mark(path, PRESERVATION_UNSUPPORTED, detail="Only system text blocks are supported", subtree=True)
            continue
        text = str(block.get("text") or "")
        if text.startswith("x-anthropic-billing-header:"):
            report.mark(
                path,
                PRESERVATION_SEMANTIC,
                detail="Synthetic Anthropic billing metadata omitted from model input",
                subtree=True,
            )
            continue
        part = {"type": "input_text", "text": text}
        target = f"/input/{item_index}/content/{len(parts)}"
        parts.append(part)
        report.mark(path + "/type", PRESERVATION_SEMANTIC, target + "/type")
        report.mark(path + "/text", PRESERVATION_EXACT, target + "/text")
        if "cache_control" in block:
            _cache_control_to_part(block["cache_control"], part, report, path + "/cache_control", target, profile)
    if parts:
        input_items.append({"type": "message", "role": "developer", "content": parts})


def _map_reasoning_effort(payload: Dict[str, Any], profile: ResponsesWireProfile, report: ConversionReport) -> Optional[str]:
    thinking = payload.get("thinking")
    output_config = payload.get("output_config")
    explicit = output_config.get("effort") if isinstance(output_config, dict) else None
    if explicit is not None:
        effort = str(explicit).lower()
        if effort in profile.reasoning_efforts:
            report.mark("/output_config/effort", PRESERVATION_EXACT, "/reasoning/effort")
            return effort
        report.mark("/output_config/effort", PRESERVATION_UNSUPPORTED, detail=f"Effort {effort!r} is not supported by {profile.name}")
    if not isinstance(thinking, dict):
        return None
    thinking_type = thinking.get("type")
    report.mark("/thinking/type", PRESERVATION_SEMANTIC, "/reasoning/effort")
    if thinking_type == "disabled":
        return "none" if "none" in profile.reasoning_efforts else "low"
    if thinking_type in ("adaptive", "auto"):
        return "high" if "high" in profile.reasoning_efforts else profile.reasoning_efforts[-1]
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens")
        if budget is not None:
            try:
                numeric = int(budget)
            except (TypeError, ValueError):
                report.mark("/thinking/budget_tokens", PRESERVATION_UNSUPPORTED, detail="budget_tokens must be an integer")
                return None
            if numeric >= 30000 and "max" in profile.reasoning_efforts:
                effort = "max"
            elif numeric >= 16000 and "xhigh" in profile.reasoning_efforts:
                effort = "xhigh"
            elif numeric >= 8000:
                effort = "high"
            elif numeric >= 3000:
                effort = "medium"
            else:
                effort = "low"
            report.mark("/thinking/budget_tokens", PRESERVATION_APPROXIMATION, "/reasoning/effort", "Numeric thinking budget mapped to a discrete effort")
            return effort
    return None


def _convert_tool_choice(
    value: Any,
    name_codec: IdentifierCodec,
    report: ConversionReport,
    *,
    has_native_web_search: bool = False,
) -> Any:
    if isinstance(value, str):
        mapped = "required" if value == "any" else value
        report.mark(
            "/tool_choice",
            PRESERVATION_EXACT if mapped == value else PRESERVATION_SEMANTIC,
            "/tool_choice",
        )
        return mapped
    if not isinstance(value, dict):
        report.mark("/tool_choice", PRESERVATION_UNSUPPORTED, detail="tool_choice must be string or object", subtree=True)
        return "auto"
    choice_type = value.get("type", "auto")
    report.mark("/tool_choice/type", PRESERVATION_SEMANTIC, "/tool_choice")
    if choice_type == "auto":
        result: Any = "auto"
    elif choice_type == "any":
        result = "required"
    elif choice_type == "none":
        result = "none"
    elif choice_type == "tool":
        original = str(value.get("name") or "")
        if original == "web_search" and has_native_web_search:
            result = {"type": "web_search"}
            report.mark("/tool_choice/name", PRESERVATION_SEMANTIC, "/tool_choice/type")
        else:
            encoded = name_codec.encode(original, "name")
            result = {"type": "function", "name": encoded}
            report.mark("/tool_choice/name", PRESERVATION_EXACT if encoded == original else PRESERVATION_SEMANTIC, "/tool_choice/name")
    else:
        result = "auto"
        report.mark("/tool_choice/type", PRESERVATION_UNSUPPORTED, detail=f"Unknown tool_choice type: {choice_type}")
    if "disable_parallel_tool_use" in value:
        if not isinstance(value["disable_parallel_tool_use"], bool):
            report.mark(
                "/tool_choice/disable_parallel_tool_use",
                PRESERVATION_UNSUPPORTED,
                detail="disable_parallel_tool_use must be a boolean",
            )
            raise AnthropicResponsesConversionError(
                "Anthropic tool_choice.disable_parallel_tool_use must be a boolean",
                report,
            )
        report.mark("/tool_choice/disable_parallel_tool_use", PRESERVATION_SEMANTIC, "/parallel_tool_calls")
    return result


def _convert_web_search_tool(
    tool: Dict[str, Any],
    path: str,
    target_base: str,
    profile: ResponsesWireProfile,
    report: ConversionReport,
) -> Optional[Dict[str, Any]]:
    if not profile.supports_native_web_search:
        report.mark(path, PRESERVATION_UNSUPPORTED, detail=f"{profile.name} has no native web search mapping", subtree=True)
        return None

    tool_type = str(tool.get("type") or "")
    name = str(tool.get("name") or "")
    if tool_type != "web_search_20250305" or name != "web_search":
        report.mark(path, PRESERVATION_UNSUPPORTED, detail="Unsupported Anthropic web search tool variant", subtree=True)
        return None

    target: Dict[str, Any] = {"type": "web_search"}
    report.mark(path + "/type", PRESERVATION_SEMANTIC, target_base + "/type")
    report.mark(path + "/name", PRESERVATION_SEMANTIC, target_base + "/type")

    allowed_domains = tool.get("allowed_domains")
    blocked_domains = tool.get("blocked_domains")
    if allowed_domains is not None and blocked_domains is not None:
        report.mark(path + "/allowed_domains", PRESERVATION_UNSUPPORTED, detail="web search cannot combine allowed_domains and blocked_domains", subtree=True)
        report.mark(path + "/blocked_domains", PRESERVATION_UNSUPPORTED, detail="web search cannot combine allowed_domains and blocked_domains", subtree=True)
        raise AnthropicResponsesConversionError("Anthropic web search cannot combine allowed_domains and blocked_domains", report)
    else:
        domain_key = "allowed_domains" if allowed_domains is not None else "blocked_domains"
        domains = allowed_domains if allowed_domains is not None else blocked_domains
        if domains is not None:
            if isinstance(domains, list) and all(isinstance(domain, str) for domain in domains):
                target["filters"] = {domain_key: copy.deepcopy(domains)}
                report.mark(path + "/" + domain_key, PRESERVATION_SEMANTIC, target_base + "/filters/" + domain_key, subtree=True)
            else:
                report.mark(path + "/" + domain_key, PRESERVATION_UNSUPPORTED, detail=f"{domain_key} must be a string array", subtree=True)
                raise AnthropicResponsesConversionError(f"Anthropic web search {domain_key} must be a string array", report)

    if "user_location" in tool:
        location = tool["user_location"]
        allowed_keys = {"type", "city", "region", "country", "timezone"}
        if (
            isinstance(location, dict)
            and set(location).issubset(allowed_keys)
            and location.get("type") == "approximate"
            and any(
                isinstance(location.get(key), str) and location.get(key)
                for key in ("city", "region", "country", "timezone")
            )
            and all(
                key == "type" or isinstance(value, str)
                for key, value in location.items()
            )
        ):
            target["user_location"] = copy.deepcopy(location)
            report.mark(path + "/user_location", PRESERVATION_EXACT, target_base + "/user_location", subtree=True)
        else:
            report.mark(path + "/user_location", PRESERVATION_UNSUPPORTED, detail="user_location must be an approximate location object", subtree=True)
            raise AnthropicResponsesConversionError("Anthropic web search user_location is invalid", report)

    if "max_uses" in tool:
        max_uses = tool["max_uses"]
        if not isinstance(max_uses, int) or isinstance(max_uses, bool) or max_uses <= 0:
            report.mark(path + "/max_uses", PRESERVATION_UNSUPPORTED, detail="max_uses must be a positive integer")
            raise AnthropicResponsesConversionError("Anthropic web search max_uses must be a positive integer", report)
        report.mark(path + "/max_uses", PRESERVATION_APPROXIMATION, detail="Responses web search has no equivalent per-request hard use cap")
    for extension in ("provider", "cache_control", "defer_loading", "allowed_callers"):
        if extension in tool:
            disposition = PRESERVATION_UNSUPPORTED if extension == "provider" else PRESERVATION_SIDECAR
            report.mark(path + "/" + extension, disposition, detail=f"Web search extension {extension} has no verified Responses mapping", subtree=True)
    return target


def _convert_tools(
    tools: Any,
    profile: ResponsesWireProfile,
    report: ConversionReport,
    name_codec: IdentifierCodec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(tools, list):
        report.mark("/tools", PRESERVATION_UNSUPPORTED, detail="tools must be an array", subtree=True)
        return [], []
    function_tools: List[Dict[str, Any]] = []
    server_tools: List[Dict[str, Any]] = []
    for index, tool in enumerate(tools):
        path = f"/tools/{index}"
        if not isinstance(tool, dict):
            report.mark(path, PRESERVATION_UNSUPPORTED, detail="tool definition must be an object", subtree=True)
            continue
        tool_type = tool.get("type")
        if isinstance(tool_type, str) and tool_type.startswith("web_search_"):
            if "input_schema" in tool:
                report.mark(path, PRESERVATION_UNSUPPORTED, detail="Anthropic web search tools cannot define input_schema", subtree=True)
                raise AnthropicResponsesConversionError("Anthropic web search tools cannot define input_schema", report)
            if profile.native_server_tools_in_input:
                target_base = f"/input/0/tools/{len(server_tools)}"
            else:
                top_level_function_count = 0 if profile.tools_in_input else len(function_tools)
                target_base = f"/tools/{top_level_function_count + len(server_tools)}"
            converted_search = _convert_web_search_tool(
                tool,
                path,
                target_base,
                profile,
                report,
            )
            if converted_search is not None:
                server_tools.append(converted_search)
            continue
        # Other Anthropic server tools have a provider-specific `type` and may
        # not include input_schema. They are not silently coerced to functions.
        if tool_type and tool_type != "custom" and "input_schema" not in tool:
            report.mark(path, PRESERVATION_UNSUPPORTED, detail=f"Unsupported Anthropic server tool type: {tool_type}", subtree=True)
            continue
        original_name = str(tool.get("name") or "")
        encoded_name = name_codec.encode(original_name, "name")
        strict = tool.get("strict", False)
        if not isinstance(strict, bool):
            report.mark(path + "/strict", PRESERVATION_UNSUPPORTED, detail="strict must be a boolean")
            raise AnthropicResponsesConversionError("Anthropic tool strict must be a boolean", report)
        target: Dict[str, Any] = {
            "type": "function",
            "name": encoded_name,
            "description": str(tool.get("description") or ""),
            "parameters": copy.deepcopy(tool.get("input_schema") or {"type": "object", "properties": {}}),
            "strict": strict,
        }
        function_tools.append(target)
        target_base = f"/tools/{len(function_tools)-1}"
        report.mark(path + "/name", PRESERVATION_EXACT if encoded_name == original_name else PRESERVATION_SEMANTIC, target_base + "/name")
        if "description" in tool:
            report.mark(path + "/description", PRESERVATION_EXACT, target_base + "/description")
        if "input_schema" in tool:
            report.mark(path + "/input_schema", PRESERVATION_EXACT, target_base + "/parameters", subtree=True)
        if "strict" in tool:
            report.mark(path + "/strict", PRESERVATION_EXACT, target_base + "/strict")
        if "type" in tool:
            report.mark(path + "/type", PRESERVATION_SEMANTIC, target_base + "/type")
        for extension in ("cache_control", "defer_loading", "allowed_callers"):
            if extension in tool:
                report.mark(path + "/" + extension, PRESERVATION_SIDECAR, detail=f"Tool extension {extension} has no verified wire mapping", subtree=True)
    return function_tools, server_tools


def _mark_known_output_config(payload: Dict[str, Any], report: ConversionReport, responses: Dict[str, Any]) -> None:
    output_config = payload.get("output_config")
    if not isinstance(output_config, dict):
        return
    if "effort" in output_config:
        # Marked by _map_reasoning_effort unless invalid; avoid an unaccounted leaf.
        if not any(record.source_path == "/output_config/effort" for record in report.records):
            report.mark("/output_config/effort", PRESERVATION_UNSUPPORTED, detail="No reasoning effort mapping was selected")
    if "format" in output_config:
        format_value = output_config["format"]
        if not isinstance(format_value, dict):
            report.mark("/output_config/format", PRESERVATION_UNSUPPORTED, detail="output_config.format must be an object", subtree=True)
            raise AnthropicResponsesConversionError("Anthropic output_config.format must be an object", report)
        format_type = format_value.get("type")
        if format_type != "json_schema":
            report.mark("/output_config/format/type", PRESERVATION_UNSUPPORTED, detail="Only json_schema output format is supported")
            raise AnthropicResponsesConversionError("Unsupported Anthropic output_config.format type", report)
        schema = format_value.get("schema")
        if not isinstance(schema, dict):
            report.mark("/output_config/format/schema", PRESERVATION_UNSUPPORTED, detail="json_schema format requires an object schema", subtree=True)
            raise AnthropicResponsesConversionError("Anthropic json_schema output format requires an object schema", report)

        normalized: Dict[str, Any] = {
            "type": "json_schema",
            "schema": copy.deepcopy(schema),
        }
        report.mark("/output_config/format/type", PRESERVATION_EXACT, "/text/format/type")
        report.mark("/output_config/format/schema", PRESERVATION_EXACT, "/text/format/schema", subtree=True)

        explicit_name = format_value.get("name")
        if isinstance(explicit_name, str) and explicit_name and len(explicit_name) <= 64 and _IDENTIFIER_RE.fullmatch(explicit_name):
            normalized["name"] = explicit_name
            report.mark("/output_config/format/name", PRESERVATION_EXACT, "/text/format/name")
        else:
            digest_source = _canonical_json(schema)
            normalized["name"] = "ghc_schema_" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
            if "name" in format_value:
                report.mark("/output_config/format/name", PRESERVATION_APPROXIMATION, "/text/format/name", "Invalid schema name replaced with a deterministic safe identifier")
            else:
                report.mark("/output_config/format", PRESERVATION_SEMANTIC, "/text/format/name", "Generated required Responses schema name from the canonical schema")

        for key, expected_type in (("description", str), ("strict", bool)):
            if key not in format_value:
                continue
            value = format_value[key]
            if not isinstance(value, expected_type):
                report.mark("/output_config/format/" + key, PRESERVATION_UNSUPPORTED, detail=f"json_schema {key} has an invalid type", subtree=True)
                raise AnthropicResponsesConversionError(f"Anthropic json_schema output format {key} has an invalid type", report)
            normalized[key] = copy.deepcopy(value)
            report.mark("/output_config/format/" + key, PRESERVATION_EXACT, "/text/format/" + key, subtree=True)

        allowed_keys = {"type", "name", "description", "schema", "strict"}
        unknown_keys = [key for key in format_value if key not in allowed_keys]
        if unknown_keys:
            for key in unknown_keys:
                report.mark("/output_config/format/" + _pointer_escape(str(key)), PRESERVATION_UNSUPPORTED, detail="Unknown json_schema format field", subtree=True)
            raise AnthropicResponsesConversionError(
                "Anthropic json_schema output format contains unsupported fields",
                report,
            )
        responses.setdefault("text", {})["format"] = normalized


def convert_anthropic_to_responses(
    payload: Dict[str, Any],
    *,
    wire_profile: str = "copilot_responses_lite",
    mode: str = MODE_COMPATIBILITY,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> AnthropicToResponsesResult:
    """Convert one Anthropic Messages request to a Responses request."""
    if not isinstance(payload, dict):
        report = ConversionReport("anthropic_to_responses")
        report.mark("/", PRESERVATION_UNSUPPORTED, detail="Request body must be an object", subtree=True)
        raise AnthropicResponsesConversionError("Anthropic request body must be an object", report)
    profile = get_wire_profile(wire_profile)
    report = ConversionReport("anthropic_to_responses")
    name_codec = IdentifierCodec()
    call_id_codec = IdentifierCodec()
    input_items: List[Dict[str, Any]] = []
    model = str(payload.get("model") or "")
    stream_value = payload.get("stream", False)
    if not isinstance(stream_value, bool):
        report.mark(
            "/stream",
            PRESERVATION_UNSUPPORTED,
            detail="stream must be a boolean",
        )
        raise AnthropicResponsesConversionError(
            "Anthropic request field 'stream' must be a boolean",
            report,
        )
    responses: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
        "stream": stream_value,
        "include": ["reasoning.encrypted_content"],
    }
    if "model" in payload:
        report.mark("/model", PRESERVATION_EXACT, "/model")
    if "stream" in payload:
        report.mark("/stream", PRESERVATION_EXACT, "/stream")

    if "system" in payload:
        _convert_system(payload["system"], input_items, report, profile)

    function_tools, server_tools = (
        _convert_tools(payload.get("tools"), profile, report, name_codec)
        if "tools" in payload else ([], [])
    )
    if function_tools:
        if profile.tools_in_input:
            input_items.insert(0, {"type": "additional_tools", "role": "developer", "tools": function_tools})
            # Existing target paths in records are descriptive only; the insert
            # does not alter preservation semantics.
        else:
            responses.setdefault("tools", []).extend(function_tools)
    if server_tools:
        if profile.native_server_tools_in_input:
            input_items.insert(0, {"type": "additional_tools", "role": "developer", "tools": server_tools})
        else:
            responses.setdefault("tools", []).extend(server_tools)
    tools = function_tools + server_tools

    messages = payload.get("messages")
    messages_error: Optional[str] = None
    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            path = f"/messages/{message_index}"
            if not isinstance(message, dict):
                report.mark(path, PRESERVATION_UNSUPPORTED, detail="message must be an object", subtree=True)
                continue
            _append_message_items(
                input_items,
                message,
                message_index,
                report,
                profile,
                name_codec,
                call_id_codec,
                model,
            )
    elif "messages" in payload:
        messages_error = "Anthropic request field 'messages' must be an array"
        report.mark(
            "/messages",
            PRESERVATION_UNSUPPORTED,
            detail="messages must be an array",
            subtree=True,
        )
    else:
        messages_error = "Anthropic request field 'messages' is required"
        report.mark(
            "/messages",
            PRESERVATION_UNSUPPORTED,
            detail="messages is a required field",
            subtree=True,
        )

    if "tools" in payload and isinstance(payload.get("tools"), list) and not payload.get("tools"):
        report.mark("/tools", PRESERVATION_EXACT, "/tools", subtree=True)

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        chosen_name = str(tool_choice.get("name") or "")
        available_names = {
            name_codec.decode(str(tool.get("name") or ""))
            for tool in function_tools
            if isinstance(tool, dict)
        }
        if server_tools:
            available_names.add("web_search")
        if chosen_name not in available_names:
            report.mark(
                "/tool_choice/name",
                PRESERVATION_UNSUPPORTED,
                detail="tool_choice names a tool that was not converted",
            )
            raise AnthropicResponsesConversionError(
                "Anthropic tool_choice names an unavailable tool",
                report,
            )
    if tool_choice is not None:
        responses["tool_choice"] = _convert_tool_choice(
            tool_choice,
            name_codec,
            report,
            has_native_web_search=bool(server_tools),
        )
        disable_parallel = tool_choice.get("disable_parallel_tool_use") if isinstance(tool_choice, dict) else None
        if disable_parallel is not None:
            responses["parallel_tool_calls"] = not disable_parallel
    elif tools:
        responses["tool_choice"] = "auto"
    if "parallel_tool_calls" not in responses and tools:
        responses["parallel_tool_calls"] = True

    effort = _map_reasoning_effort(payload, profile, report)
    reasoning_options: Dict[str, Any] = {}
    if effort:
        reasoning_options["effort"] = effort
    if profile.supports_reasoning_context and (
        effort or "context_management" in payload
    ):
        reasoning_options["context"] = "all_turns"
    if reasoning_options:
        responses["reasoning"] = reasoning_options

    if "thinking" in payload and isinstance(payload.get("thinking"), dict):
        for key in payload["thinking"]:
            path = "/thinking/" + _pointer_escape(key)
            if not any(record.source_path == path or (record.subtree and path.startswith(record.source_path + "/")) for record in report.records):
                report.mark(path, PRESERVATION_SIDECAR, detail="Thinking extension preserved in conversion report", subtree=True)

    if "context_management" in payload:
        context = payload["context_management"]
        known = False
        if isinstance(context, dict) and isinstance(context.get("edits"), list):
            known = all(
                isinstance(edit, dict)
                and edit.get("type") == "clear_thinking_20251015"
                and edit.get("keep") == "all"
                for edit in context["edits"]
            )
        can_map_context = known and profile.supports_reasoning_context
        report.mark(
            "/context_management",
            PRESERVATION_SEMANTIC if can_map_context else PRESERVATION_UNSUPPORTED,
            "/reasoning/context" if can_map_context else None,
            (
                "Mapped clear_thinking keep=all to all_turns"
                if can_map_context
                else "The selected Responses profile has no reasoning context mapping"
                if known
                else "Unknown context-management edit"
            ),
            subtree=True,
        )

    if "max_tokens" in payload:
        if profile.supports_max_output_tokens:
            responses["max_output_tokens"] = payload["max_tokens"]
            report.mark("/max_tokens", PRESERVATION_SEMANTIC, "/max_output_tokens")
        else:
            report.mark("/max_tokens", PRESERVATION_APPROXIMATION, detail=f"{profile.name} does not accept max_output_tokens")
    for source_name, supported in (("temperature", profile.supports_temperature), ("top_p", profile.supports_top_p)):
        if source_name in payload:
            if supported:
                responses[source_name] = payload[source_name]
                report.mark("/" + source_name, PRESERVATION_EXACT, "/" + source_name)
            else:
                report.mark("/" + source_name, PRESERVATION_APPROXIMATION, detail=f"{profile.name} has no verified {source_name} support")
    if "top_k" in payload:
        report.mark("/top_k", PRESERVATION_UNSUPPORTED, detail="Responses API has no top_k parameter")

    stop_sequences: List[str] = []
    if "stop_sequences" in payload:
        raw_stops = payload["stop_sequences"]
        if isinstance(raw_stops, list) and all(isinstance(item, str) for item in raw_stops):
            stop_sequences = list(raw_stops)
            report.mark("/stop_sequences", PRESERVATION_APPROXIMATION, detail="Stop sequences are enforced on proxy output", subtree=True)
        else:
            report.mark("/stop_sequences", PRESERVATION_UNSUPPORTED, detail="stop_sequences must be a string array", subtree=True)

    if "metadata" in payload:
        metadata = payload["metadata"]
        if isinstance(metadata, dict):
            normalized: Dict[str, str] = {}
            for key, value in metadata.items():
                normalized[str(key)] = value if isinstance(value, str) else _canonical_json(value)
                source_path = "/metadata/" + _pointer_escape(str(key))
                target_path = source_path
                if isinstance(value, str):
                    report.mark(source_path, PRESERVATION_EXACT, target_path, subtree=True)
                else:
                    report.mark(
                        source_path,
                        PRESERVATION_SIDECAR,
                        target_path,
                        "Metadata JSON type represented canonically on the wire and recorded in conversion diagnostics; wire value is canonical JSON text",
                        subtree=True,
                    )
            responses["metadata"] = normalized
        else:
            report.mark("/metadata", PRESERVATION_UNSUPPORTED, detail="metadata must be an object", subtree=True)

    if session_id:
        tenant_scope = str(tenant_id or "anonymous")
        cache_scope = f"{tenant_scope}\x00{session_id}\x00{model}"
        responses["prompt_cache_key"] = hashlib.sha256(
            cache_scope.encode("utf-8")
        ).hexdigest()
        if profile.name == "copilot_responses_lite":
            metadata_scope = f"{tenant_scope}\x00{session_id}"
            responses["client_metadata"] = {
                "session_id": hashlib.sha256(
                    metadata_scope.encode("utf-8")
                ).hexdigest()
            }

    if "service_tier" in payload:
        value = payload["service_tier"]
        mapping = {"auto": "auto", "standard_only": "default", "default": "default", "priority": "priority"}
        if value in mapping:
            responses["service_tier"] = mapping[value]
            report.mark("/service_tier", PRESERVATION_EXACT if mapping[value] == value else PRESERVATION_SEMANTIC, "/service_tier")
        else:
            report.mark("/service_tier", PRESERVATION_UNSUPPORTED, detail=f"Unknown service tier: {value}")

    _mark_known_output_config(payload, report, responses)
    if profile.default_text_verbosity:
        responses.setdefault("text", {}).setdefault("verbosity", profile.default_text_verbosity)

    report.finalize(payload, mode)
    # Compatibility mode may continue past unknown optional extensions, but a
    # missing/malformed core collection cannot be converted or safely iterated
    # by the route.  Reject it before any upstream request in every mode.
    if messages_error is not None:
        raise AnthropicResponsesConversionError(messages_error, report)
    return AnthropicToResponsesResult(
        payload=responses,
        report=report,
        name_codec=name_codec,
        call_id_codec=call_id_codec,
        stop_sequences=stop_sequences,
        wire_profile=profile.name,
    )


def _find_first_stop(text: str, stop_sequences: Sequence[str]) -> Tuple[Optional[int], Optional[str]]:
    best_index: Optional[int] = None
    best_stop: Optional[str] = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (best_index is None or index < best_index or (index == best_index and len(stop) > len(best_stop or ""))):
            best_index = index
            best_stop = stop
    return best_index, best_stop


def _truncate_blocks_at_stop(
    blocks: List[Dict[str, Any]],
    stop_sequences: Sequence[str],
    group_ids: Optional[Sequence[Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not stop_sequences:
        return blocks, None
    if group_ids is None:
        # Without source-item information, each text block is its own segment;
        # this is safer than matching across a hidden tool/item boundary.
        group_ids = list(range(len(blocks)))
    segments: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    current_group: Any = object()
    for block_index, block in enumerate(blocks):
        group = group_ids[block_index] if block_index < len(group_ids) else block_index
        if block.get("type") != "text" or not current or group != current_group:
            if current:
                segments.append(current)
                current = []
        if block.get("type") != "text":
            current_group = object()
            continue
        current_group = group
        current.append((block_index, str(block.get("text") or "")))
    if current:
        segments.append(current)

    for segment in segments:
        carried = ""
        text_positions: List[Tuple[int, int, int]] = []
        for block_index, text in segment:
            start = len(carried)
            carried += text
            text_positions.append((block_index, start, len(carried)))
        stop_index, matched = _find_first_stop(carried, stop_sequences)
        if stop_index is None:
            continue
        for block_index, start, end in text_positions:
            if stop_index >= end:
                continue
            result = copy.deepcopy(blocks[: block_index + 1])
            result[block_index]["text"] = str(
                result[block_index].get("text") or ""
            )[: stop_index - start]
            # The matching text block may become empty, but every block after
            # it is post-stop output (including tool calls).
            return result, matched
    return blocks, None


_KNOWN_RESPONSE_SIDECAR_FIELDS = {
    "object", "created_at", "completed_at", "status", "background", "billing", "error",
    "copilot_usage", "output_text",
    "incomplete_details", "instructions", "max_output_tokens", "max_tool_calls",
    "parallel_tool_calls", "previous_response_id", "prompt_cache_key",
    "prompt_cache_retention", "reasoning", "safety_identifier", "service_tier",
    "store", "temperature", "text", "tool_choice", "tools", "top_logprobs",
    "top_p", "truncation", "user", "metadata", "client_metadata",
    "frequency_penalty", "presence_penalty", "moderation",
}


def _usage_token_count(
    value: Any,
    path: str,
    report: ConversionReport,
) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        report.mark(
            path,
            PRESERVATION_UNSUPPORTED,
            detail="Responses token counts must be non-negative integers",
        )
        raise AnthropicResponsesConversionError(
            "Responses usage contains an invalid token count",
            report,
        )
    return value


def _responses_usage_to_anthropic(
    usage: Any,
    tool_usage: Any,
    report: ConversionReport,
) -> Dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    total_input = _usage_token_count(
        usage.get("input_tokens"), "/usage/input_tokens", report
    )
    cached = _usage_token_count(
        details.get("cached_tokens"),
        "/usage/input_tokens_details/cached_tokens",
        report,
    )
    cache_write = _usage_token_count(
        details.get("cache_write_tokens"),
        "/usage/input_tokens_details/cache_write_tokens",
        report,
    )
    output_tokens = _usage_token_count(
        usage.get("output_tokens"), "/usage/output_tokens", report
    )
    result: Dict[str, Any] = {
        "input_tokens": max(0, total_input - cached - cache_write),
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cached,
    }
    reasoning_tokens = _usage_token_count(
        output_details.get("reasoning_tokens"),
        "/usage/output_tokens_details/reasoning_tokens",
        report,
    )
    if reasoning_tokens:
        result["output_tokens_details"] = {"thinking_tokens": reasoning_tokens}
    if isinstance(tool_usage, dict):
        web_search = tool_usage.get("web_search")
        if isinstance(web_search, dict):
            raw_count = web_search.get("num_requests")
            requests_count = (
                raw_count
                if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                else 0
            )
            if requests_count > 0:
                result["server_tool_use"] = {
                    "web_search_requests": requests_count,
                }
    return result


def _mark_usage_preservation(usage: Any, report: ConversionReport) -> None:
    if not isinstance(usage, dict):
        report.mark(
            "/usage",
            PRESERVATION_UNSUPPORTED,
            detail="Responses usage must be an object",
            subtree=True,
        )
        return
    if not usage:
        report.mark(
            "/usage",
            PRESERVATION_SEMANTIC,
            "/usage",
            "Empty usage object projected to zero-valued Anthropic usage",
            subtree=True,
        )
        return
    direct = {
        "input_tokens": "/usage/input_tokens",
        "output_tokens": "/usage/output_tokens",
    }
    for key, value in usage.items():
        path = "/usage/" + _pointer_escape(str(key))
        if key in direct:
            report.mark(
                path,
                PRESERVATION_SEMANTIC,
                direct[key],
                "Token accounting is projected into Anthropic usage fields",
                subtree=True,
            )
        elif key == "input_tokens_details" and isinstance(value, dict):
            if not value:
                report.mark(
                    path,
                    PRESERVATION_SIDECAR,
                    detail="Empty input token details retained in the conversion diagnostics",
                    subtree=True,
                )
            for detail_key in value:
                detail_path = path + "/" + _pointer_escape(str(detail_key))
                if detail_key in ("cached_tokens", "cache_write_tokens"):
                    target = (
                        "/usage/cache_read_input_tokens"
                        if detail_key == "cached_tokens"
                        else "/usage/cache_creation_input_tokens"
                    )
                    report.mark(
                        detail_path,
                        PRESERVATION_SEMANTIC,
                        target,
                        subtree=True,
                    )
                else:
                    report.mark(
                        detail_path,
                        PRESERVATION_SIDECAR,
                        detail="Unprojected input token detail retained in the conversion diagnostics",
                        subtree=True,
                    )
        elif key == "output_tokens_details" and isinstance(value, dict):
            if not value:
                report.mark(
                    path,
                    PRESERVATION_SIDECAR,
                    detail="Empty output token details retained in the conversion diagnostics",
                    subtree=True,
                )
            for detail_key in value:
                detail_path = path + "/" + _pointer_escape(str(detail_key))
                if detail_key == "reasoning_tokens":
                    report.mark(
                        detail_path,
                        PRESERVATION_SEMANTIC,
                        "/usage/output_tokens_details/thinking_tokens",
                        subtree=True,
                    )
                else:
                    report.mark(
                        detail_path,
                        PRESERVATION_SIDECAR,
                        detail="Unprojected output token detail retained in the conversion diagnostics",
                        subtree=True,
                    )
        else:
            report.mark(
                path,
                PRESERVATION_SIDECAR,
                detail="Responses usage extension retained in the conversion diagnostics",
                subtree=True,
            )


def _response_object(value: Dict[str, Any]) -> Dict[str, Any]:
    if value.get("type") in ("response.completed", "response.incomplete", "response.failed") and isinstance(value.get("response"), dict):
        return value["response"]
    return value


def _strict_function_arguments(
    raw_arguments: Any,
    report: ConversionReport,
    path: str,
) -> Dict[str, Any]:
    """Decode a Responses function argument string as one strict JSON object.

    Sending a scalar, array, malformed JSON, or an object with duplicate keys
    as an Anthropic ``tool_use.input`` can make the CLI execute a call with a
    different contract.  This is therefore a fail-closed boundary in both
    compatibility modes, not a best-effort projection.
    """

    try:
        if not isinstance(raw_arguments, str):
            raise StrictJSONError("Function arguments must be a JSON string")
        parsed = parse_strict_json_bytes(raw_arguments.encode("utf-8", errors="strict"))
        if not isinstance(parsed, dict):
            raise StrictJSONError("Function arguments must decode to a JSON object")
    except (StrictJSONError, UnicodeEncodeError) as exc:
        report.mark(
            path,
            PRESERVATION_UNSUPPORTED,
            detail="Function arguments are not a strict JSON object",
            subtree=True,
        )
        raise AnthropicResponsesConversionError(
            "Responses function arguments are not a strict JSON object",
            report,
        ) from exc
    return parsed


def convert_responses_to_anthropic(
    response_value: Dict[str, Any],
    *,
    original_model: str,
    reasoning_model: str,
    wire_profile: str,
    name_codec: Optional[IdentifierCodec] = None,
    call_id_codec: Optional[IdentifierCodec] = None,
    stop_sequences: Optional[Sequence[str]] = None,
    mode: str = MODE_COMPATIBILITY,
) -> ResponsesToAnthropicResult:
    """Convert a terminal Responses object/event into an Anthropic message."""
    if not isinstance(response_value, dict):
        report = ConversionReport("responses_to_anthropic")
        report.mark("/", PRESERVATION_UNSUPPORTED, detail="Responses body must be an object", subtree=True)
        raise AnthropicResponsesConversionError("Responses body must be an object", report)
    terminal_event_type = response_value.get("type")
    response = _response_object(response_value)
    report = ConversionReport("responses_to_anthropic")
    response_id_value = response.get("id")
    if not isinstance(response_id_value, str) or not response_id_value:
        report.mark("/id", PRESERVATION_UNSUPPORTED, detail="Responses body requires a non-empty string id")
        raise AnthropicResponsesConversionError("Responses body requires a non-empty string id", report)
    if not isinstance(response.get("output"), list):
        report.mark("/output", PRESERVATION_UNSUPPORTED, detail="Responses output must be an array", subtree=True)
        raise AnthropicResponsesConversionError("Responses output must be an array", report)
    name_codec = name_codec or IdentifierCodec()
    call_id_codec = call_id_codec or IdentifierCodec()
    content: List[Dict[str, Any]] = []
    content_group_ids: List[Any] = []
    seen_non_reasoning_output = False
    has_tool_use = False
    has_refusal = False

    output = response.get("output")
    if isinstance(output, list):
        for index, item in enumerate(output):
            path = f"/output/{index}"
            if not isinstance(item, dict):
                report.mark(path, PRESERVATION_UNSUPPORTED, detail="output item must be an object", subtree=True)
                continue
            item_type = item.get("type")
            if item_type == "reasoning":
                if seen_non_reasoning_output:
                    report.mark(
                        path,
                        PRESERVATION_UNSUPPORTED,
                        detail="Reasoning appeared after non-reasoning output",
                        subtree=True,
                    )
                    raise AnthropicResponsesConversionError(
                        "Responses returned reasoning after visible output",
                        report,
                    )
                summary = item.get("summary")
                summary_text = ""
                if isinstance(summary, list):
                    summary_text = "".join(
                        str(part.get("text") or "")
                        for part in summary
                        if isinstance(part, dict)
                        and part.get("type") in ("summary_text", "reasoning_text")
                    )
                encrypted_content = item.get("encrypted_content")
                if not isinstance(encrypted_content, str):
                    encrypted_content = None
                item_id = item.get("id") if wire_profile == "public_responses" else None
                if not isinstance(item_id, str):
                    item_id = None
                try:
                    carrier = build_reasoning_carrier(
                        model=reasoning_model,
                        wire_profile=wire_profile,
                        item_id=item_id,
                        encrypted_content=encrypted_content,
                    )
                except ValueError:
                    carrier = build_reasoning_carrier(
                        model=reasoning_model,
                        wire_profile=wire_profile,
                        item_id=item_id,
                        encrypted_content=None,
                    )
                    report.mark(
                        path + "/encrypted_content",
                        PRESERVATION_APPROXIMATION,
                        detail="Encrypted reasoning exceeded the carrier size limit and was dropped",
                    )
                content.append({
                    "type": "thinking",
                    "thinking": summary_text,
                    "signature": carrier,
                })
                content_group_ids.append(index)
                report.mark(
                    path,
                    PRESERVATION_SEMANTIC,
                    detail="Responses reasoning carried in an Anthropic thinking block",
                    subtree=True,
                )
            elif item_type == "message":
                seen_non_reasoning_output = True
                report.mark(path + "/type", PRESERVATION_SEMANTIC)
                if "role" in item:
                    report.mark(path + "/role", PRESERVATION_EXACT)
                if "phase" in item:
                    report.mark(path + "/phase", PRESERVATION_SIDECAR, detail="Assistant phase retained for diagnostics")
                parts = item.get("content")
                if not isinstance(parts, list):
                    report.mark(path + "/content", PRESERVATION_UNSUPPORTED, detail="message output content must be an array", subtree=True)
                    continue
                for part_index, part in enumerate(parts):
                    part_path = f"{path}/content/{part_index}"
                    if not isinstance(part, dict):
                        report.mark(part_path, PRESERVATION_UNSUPPORTED, subtree=True)
                        continue
                    if part.get("type") == "output_text":
                        content.append({"type": "text", "text": str(part.get("text") or "")})
                        content_group_ids.append(index)
                        report.mark(part_path + "/type", PRESERVATION_SEMANTIC)
                        report.mark(part_path + "/text", PRESERVATION_EXACT)
                        for extension in ("annotations", "logprobs"):
                            if extension in part:
                                report.mark(part_path + "/" + extension, PRESERVATION_SIDECAR, detail=f"Responses {extension} has no Anthropic text-block equivalent", subtree=True)
                    elif part.get("type") == "refusal":
                        has_refusal = True
                        content.append({"type": "text", "text": str(part.get("refusal") or "")})
                        content_group_ids.append(index)
                        report.mark(part_path, PRESERVATION_APPROXIMATION, detail="Refusal projected as Anthropic text", subtree=True)
                    else:
                        report.mark(part_path, PRESERVATION_UNSUPPORTED, detail=f"Unknown message content type: {part.get('type')}", subtree=True)
                for key in item:
                    if key not in {"type", "role", "phase", "content"}:
                        report.mark(path + "/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Output message metadata retained for diagnostics", subtree=True)
            elif item_type == "web_search_call":
                seen_non_reasoning_output = True
                report.mark(path, PRESERVATION_SIDECAR, detail="Native web search execution retained for diagnostics and omitted from visible Anthropic content", subtree=True)
            elif item_type in ("function_call", "custom_tool_call"):
                seen_non_reasoning_output = True
                encoded_id = item.get("call_id")
                if not isinstance(encoded_id, str) or not encoded_id:
                    report.mark(
                        path + "/call_id",
                        PRESERVATION_UNSUPPORTED,
                        detail="Tool call requires a non-empty string call_id",
                    )
                    raise AnthropicResponsesConversionError(
                        "Responses tool call requires a non-empty call_id",
                        report,
                    )
                encoded_name = item.get("name")
                if not isinstance(encoded_name, str) or not encoded_name:
                    report.mark(
                        path + "/name",
                        PRESERVATION_UNSUPPORTED,
                        detail="Tool call requires a non-empty string name",
                    )
                    raise AnthropicResponsesConversionError(
                        "Responses tool call requires a non-empty name",
                        report,
                    )
                original_id = call_id_codec.decode(encoded_id)
                original_name = name_codec.decode(encoded_name)
                raw_arguments = item.get("arguments") if item_type == "function_call" else item.get("input")
                argument_path = path + ("/arguments" if item_type == "function_call" else "/input")
                if item_type == "function_call":
                    parsed_arguments = _strict_function_arguments(
                        raw_arguments,
                        report,
                        argument_path,
                    )
                    report.mark(
                        argument_path,
                        PRESERVATION_SEMANTIC,
                        detail="Strict JSON argument object projected as Anthropic tool input",
                        subtree=True,
                    )
                else:
                    if not isinstance(raw_arguments, str):
                        report.mark(
                            argument_path,
                            PRESERVATION_UNSUPPORTED,
                            detail="Custom tool input must be a string",
                            subtree=True,
                        )
                        raise AnthropicResponsesConversionError(
                            "Responses custom tool input is not a string",
                            report,
                        )
                    # Custom tools accept grammar-defined free-form text while
                    # Anthropic tool inputs must be objects.  Preserve the raw
                    # string in the same reversible envelope used by SSE.
                    parsed_arguments = {"input": raw_arguments}
                    report.mark(
                        argument_path,
                        PRESERVATION_APPROXIMATION,
                        detail="Custom tool input wrapped in a reversible Anthropic object",
                        subtree=True,
                    )
                content.append({"type": "tool_use", "id": original_id, "name": original_name, "input": parsed_arguments})
                content_group_ids.append(("tool", index))
                has_tool_use = True
                report.mark(path + "/type", PRESERVATION_SEMANTIC)
                if "call_id" in item:
                    report.mark(path + "/call_id", PRESERVATION_EXACT if original_id == encoded_id else PRESERVATION_SEMANTIC)
                if "id" in item:
                    report.mark(path + "/id", PRESERVATION_SIDECAR, detail="Responses item id retained for diagnostics")
                if "name" in item:
                    report.mark(path + "/name", PRESERVATION_EXACT if original_name == encoded_name else PRESERVATION_SEMANTIC)
                for key in item:
                    if key not in {"type", "call_id", "id", "name", "arguments", "input"}:
                        report.mark(path + "/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Tool-call metadata retained for diagnostics", subtree=True)
            else:
                report.mark(
                    path,
                    PRESERVATION_UNSUPPORTED,
                    detail="Unknown Responses output item type recorded and skipped",
                    subtree=True,
                )
                if mode == MODE_LOSSLESS_REQUIRED:
                    raise AnthropicResponsesConversionError(
                        "Responses output item type is not safely representable",
                        report,
                    )
                # Skipping one unrepresentable item keeps the rest of a
                # finished answer; refusing it delivers nothing at all.
                seen_non_reasoning_output = True

    active_stop_sequences = list(stop_sequences or [])
    content, matched_stop = _truncate_blocks_at_stop(
        content, active_stop_sequences, content_group_ids
    )
    incomplete_value = response.get("incomplete_details")
    incomplete = incomplete_value if isinstance(incomplete_value, dict) else {}
    is_incomplete = (
        response.get("status") == "incomplete"
        or terminal_event_type == "response.incomplete"
    )
    incomplete_stop_reason: Optional[str] = None
    if is_incomplete:
        incomplete_reason = incomplete.get("reason")
        if incomplete_reason == "max_output_tokens":
            incomplete_stop_reason = "max_tokens"
        elif incomplete_reason in ("content_filter", "safety", "policy"):
            incomplete_stop_reason = "refusal"
        else:
            error_path = (
                "/incomplete_details/reason"
                if isinstance(incomplete_value, dict)
                else "/incomplete_details"
            )
            report.mark(
                error_path,
                PRESERVATION_UNSUPPORTED,
                detail="Unknown or missing incomplete response reason",
                subtree=not isinstance(incomplete_value, dict),
            )
            if mode == MODE_LOSSLESS_REQUIRED:
                raise AnthropicResponsesConversionError(
                    "Responses incomplete reason is not safely representable",
                    report,
                )
            # "Incomplete" always means the answer was cut short; reporting the
            # closest Anthropic truncation reason beats discarding the content.
            incomplete_stop_reason = "max_tokens"
    if matched_stop:
        stop_reason = "stop_sequence"
    elif incomplete_stop_reason is not None:
        stop_reason = incomplete_stop_reason
    elif has_tool_use:
        stop_reason = "tool_use"
    elif has_refusal:
        stop_reason = "refusal"
    else:
        stop_reason = "end_turn"

    response_id = str(response.get("id") or "")
    message_id = response_id if response_id.startswith("msg_") else "msg_" + hashlib.sha256(response_id.encode("utf-8")).hexdigest()[:24]
    anthropic = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": _responses_usage_to_anthropic(
            response.get("usage"),
            response.get("tool_usage"),
            report,
        ),
    }
    if "usage" in response:
        _mark_usage_preservation(response.get("usage"), report)
    if "tool_usage" in response:
        tool_usage = response.get("tool_usage")
        if not isinstance(tool_usage, dict):
            report.mark(
                "/tool_usage",
                PRESERVATION_UNSUPPORTED,
                detail="Responses tool_usage must be an object",
                subtree=True,
            )
        else:
            web_search_usage = tool_usage.get("web_search")
            if web_search_usage is not None:
                requests_count = web_search_usage.get("num_requests") if isinstance(web_search_usage, dict) else None
                if isinstance(requests_count, int) and not isinstance(requests_count, bool) and requests_count >= 0:
                    report.mark(
                        "/tool_usage/web_search/num_requests",
                        PRESERVATION_SEMANTIC,
                        "/usage/server_tool_use/web_search_requests",
                    )
                    for key in web_search_usage:
                        if key != "num_requests":
                            report.mark(
                                "/tool_usage/web_search/" + _pointer_escape(str(key)),
                                PRESERVATION_SIDECAR,
                                detail="Unprojected web search usage retained in conversion diagnostics",
                                subtree=True,
                            )
                else:
                    report.mark(
                        "/tool_usage/web_search",
                        PRESERVATION_UNSUPPORTED,
                        detail="Responses web search usage must contain a non-negative integer num_requests",
                        subtree=True,
                    )
            for key in tool_usage:
                if key != "web_search":
                    report.mark(
                        "/tool_usage/" + _pointer_escape(str(key)),
                        PRESERVATION_SIDECAR,
                        detail="Unprojected Responses tool usage retained in conversion diagnostics",
                        subtree=True,
                    )
    for key in ("id", "model", "output"):
        if key in response and not any(record.source_path == "/" + key or record.source_path.startswith("/" + key + "/") for record in report.records):
            report.mark(
                "/" + key,
                PRESERVATION_SEMANTIC,
                detail=(
                    "Upstream model is projected to the client-requested model"
                    if key == "model" else None
                ),
                subtree=(key == "output"),
            )
    for key in _KNOWN_RESPONSE_SIDECAR_FIELDS:
        if key in response:
            report.mark("/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Responses metadata retained in conversion diagnostics", subtree=True)
    report.finalize(response, mode)
    return ResponsesToAnthropicResult(
        response=anthropic,
        report=report,
        matched_stop_sequence=matched_stop,
    )


_RESPONSES_ERROR_STATUS_BY_CODE = {
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "insufficient_quota": 429,
    "quota_exceeded": 429,
    "tokens_exceeded": 429,
    "overloaded": 529,
    "overloaded_error": 529,
    "server_overloaded": 529,
    "invalid_request_error": 400,
    "invalid_request": 400,
    "context_length_exceeded": 400,
    "invalid_prompt": 400,
    "string_above_max_length": 400,
    "authentication_error": 401,
    "invalid_api_key": 401,
    "unauthorized": 401,
    "permission_error": 403,
    "permission_denied": 403,
    "model_not_found": 404,
    "not_found_error": 404,
    "request_too_large": 413,
    "server_error": 500,
    "api_error": 500,
}


def responses_error_status(error_value: Any, default: int = 500) -> int:
    """Map an upstream Responses error to the HTTP class it really is.

    Collapsing every upstream failure into one status hides retryable
    conditions (rate limits, overload) from clients that back off on them.
    """

    error: Any = error_value
    if isinstance(error, dict) and isinstance(error.get("error"), dict):
        error = error["error"]
    if not isinstance(error, dict):
        return default
    status = error.get("status") or error.get("status_code") or error.get("http_status")
    if isinstance(status, bool):
        status = None
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    if isinstance(status, int) and 400 <= status <= 599:
        return status
    for key in ("code", "type", "param"):
        value = error.get(key)
        if isinstance(value, str):
            mapped = _RESPONSES_ERROR_STATUS_BY_CODE.get(value.strip().lower())
            if mapped is not None:
                return mapped
    return default


def anthropic_error_from_responses(error_value: Any, status_code: int = 500) -> Dict[str, Any]:
    """Return an Anthropic-shaped error without leaking a foreign envelope."""
    if isinstance(error_value, dict) and isinstance(error_value.get("error"), dict):
        error = error_value["error"]
    elif isinstance(error_value, dict):
        error = error_value
    else:
        error = {"message": str(error_value)}
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }
    error_type = mapping.get(int(status_code), "api_error")
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": str(error.get("message") or error.get("code") or "Upstream request failed"),
        },
    }
