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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"Invalid JSON at character {exc.pos}: {exc.msg}") from exc
    if text[end:].strip():
        raise StrictJSONError(f"Trailing data after JSON value at character {end}")

    def validate_unicode(node: Any, path: str = "$") -> None:
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
                validate_unicode(key, path + ".<key>")
                validate_unicode(child, path + ".<value>")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                validate_unicode(child, f"{path}[{index}]")

    validate_unicode(value)
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
    """Audit trail for one direction of a protocol conversion."""

    direction: str
    sidecar_available: bool = False
    records: List[PreservationRecord] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    unaccounted_paths: List[str] = field(default_factory=list)
    _marked_paths: Set[str] = field(default_factory=set, repr=False)
    _marked_subtrees: Set[str] = field(default_factory=set, repr=False)

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
        for existing in self.records:
            if (
                existing.source_path,
                existing.disposition,
                existing.target_path,
                existing.detail,
                existing.subtree,
            ) == key:
                return
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
        elif disposition == PRESERVATION_SIDECAR and not self.sidecar_available:
            warning = {
                "code": "conversion.sidecar_unavailable",
                "path": source_path,
                "action": "not_stored",
            }
            if warning not in self.warnings:
                self.warnings.append(warning)

    def _is_accounted(self, path: str) -> bool:
        if path in self._marked_paths:
            return True
        return any(path == prefix or path.startswith(prefix + "/") for prefix in self._marked_subtrees)

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
            if (
                record.disposition in (
                    PRESERVATION_APPROXIMATION,
                    PRESERVATION_UNSUPPORTED,
                )
                or (
                    record.disposition == PRESERVATION_SIDECAR
                    and not self.sidecar_available
                )
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
            "sidecar_available": self.sidecar_available,
            "records": [record.to_dict() for record in self.records],
            "warnings": copy.deepcopy(self.warnings),
            "unaccounted_paths": list(self.unaccounted_paths),
        }


@dataclass(frozen=True)
class ResponsesWireProfile:
    name: str
    tools_in_input: bool
    supports_prompt_cache_breakpoint: bool
    supports_temperature: bool
    supports_top_p: bool
    supports_max_output_tokens: bool
    reasoning_efforts: Tuple[str, ...]
    default_text_verbosity: Optional[str] = None


WIRE_PROFILES: Dict[str, ResponsesWireProfile] = {
    "public_responses": ResponsesWireProfile(
        name="public_responses",
        tools_in_input=False,
        supports_prompt_cache_breakpoint=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_output_tokens=True,
        reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    ),
    "copilot_responses_lite": ResponsesWireProfile(
        name="copilot_responses_lite",
        tools_in_input=True,
        # The supplied dump proves prompt_cache_key but not explicit breakpoints.
        supports_prompt_cache_breakpoint=False,
        supports_temperature=False,
        supports_top_p=False,
        supports_max_output_tokens=True,
        reasoning_efforts=("none", "low", "medium", "high", "xhigh", "max"),
        default_text_verbosity="low",
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

    def encode(self, value: str, kind: str = "id") -> str:
        value = str(value or "")
        if value in self._original_to_encoded:
            return self._original_to_encoded[value]
        if value and len(value) <= self.max_length and _IDENTIFIER_RE.fullmatch(value):
            encoded = value
        else:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_")
            prefix = "ghc_call_" if kind == "call" else "ghc_tool_"
            room = max(0, self.max_length - len(prefix) - len(digest) - 1)
            encoded = f"{prefix}{safe[:room]}_{digest}" if room else f"{prefix}{digest}"
            encoded = encoded[:self.max_length]
        self._encoded_to_original[encoded] = value
        self._original_to_encoded[value] = encoded
        return encoded

    def decode(self, value: str) -> str:
        return self._encoded_to_original.get(value, value)

    def mapping(self) -> Dict[str, str]:
        return dict(self._encoded_to_original)

    def restore_mapping(self, encoded_to_original: Dict[str, str]) -> None:
        """Merge a trusted replay-sidecar mapping into this request codec."""
        if not isinstance(encoded_to_original, dict):
            raise TypeError("identifier mapping must be an object")
        for encoded, original in encoded_to_original.items():
            if not isinstance(encoded, str) or not isinstance(original, str):
                raise TypeError("identifier mapping entries must be strings")
            existing_original = self._encoded_to_original.get(encoded)
            existing_encoded = self._original_to_encoded.get(original)
            if existing_original not in (None, original) or existing_encoded not in (None, encoded):
                raise ValueError("identifier replay mapping conflicts with request mapping")
            self._encoded_to_original[encoded] = original
            self._original_to_encoded[original] = encoded


ReplayResolver = Callable[[int, Dict[str, Any]], Optional[List[Dict[str, Any]]]]


def prepare_replay_items_for_wire(
    items: Sequence[Dict[str, Any]],
    wire_profile: str,
) -> List[Dict[str, Any]]:
    """Encode stored terminal output items for a subsequent Responses input.

    The replay store deliberately retains the complete terminal objects.  The
    Copilot Responses Lite dialect observed in the supplied dump accepts a
    smaller input projection, however: terminal-only ``id``/``status`` fields
    are removed, while encrypted reasoning, assistant ``phase`` and tool
    identity remain byte-for-byte JSON values.  Public Responses accepts the
    normal output-item representation and therefore receives a deep copy.
    """
    profile = get_wire_profile(wire_profile)
    copied = copy.deepcopy(list(items))
    if profile.name != "copilot_responses_lite":
        return copied

    allowed_by_type = {
        "reasoning": {"type", "summary", "encrypted_content"},
        "message": {"type", "role", "content", "phase"},
        "function_call": {
            "type", "call_id", "name", "arguments", "namespace",
        },
        # The dump demonstrates that Lite requires status on replayed custom
        # calls even though it rejects terminal ids.
        "custom_tool_call": {
            "type", "call_id", "name", "input", "status", "namespace",
        },
    }
    projected: List[Dict[str, Any]] = []
    for item in copied:
        if not isinstance(item, dict):
            raise ValueError("Replay output item must be an object")
        item_type = item.get("type")
        allowed = allowed_by_type.get(item_type)
        if allowed is None:
            raise ValueError(f"Unsupported replay output item type: {item_type}")
        encoded = {key: value for key, value in item.items() if key in allowed}
        if item_type == "message" and isinstance(encoded.get("content"), list):
            parts: List[Any] = []
            for part in encoded["content"]:
                if isinstance(part, dict) and part.get("type") in ("output_text", "input_text"):
                    parts.append({
                        key: copy.deepcopy(part[key])
                        for key in ("type", "text")
                        if key in part
                    })
                else:
                    # Unknown content is rejected earlier by response profile
                    # auditing.  Keeping it here avoids a second, silent loss
                    # if a caller uses this helper independently.
                    parts.append(copy.deepcopy(part))
            encoded["content"] = parts
        projected.append(encoded)
    return projected


@dataclass
class AnthropicToResponsesResult:
    payload: Dict[str, Any]
    report: ConversionReport
    name_codec: IdentifierCodec
    call_id_codec: IdentifierCodec
    stop_sequences: List[str]
    replay_misses: List[int]
    wire_profile: str


@dataclass
class ResponsesToAnthropicResult:
    response: Dict[str, Any]
    report: ConversionReport
    replay_items: List[Dict[str, Any]]
    visible_assistant_message: Dict[str, Any]
    matched_stop_sequence: Optional[str] = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


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
        if role == "assistant":
            item["phase"] = "commentary" if has_tool_use else "final_answer"
            report.mark(base, PRESERVATION_APPROXIMATION, detail="Assistant phase inferred because replay state was unavailable")
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
        elif block_type in ("thinking", "redacted_thinking"):
            # Foreign provider signatures are not valid OpenAI encrypted reasoning.
            report.mark(path, PRESERVATION_SIDECAR, detail="Provider-specific thinking block retained only in audit sidecar", subtree=True)
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
        part = {"type": "input_text", "text": str(block.get("text") or "")}
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


def _convert_tool_choice(value: Any, name_codec: IdentifierCodec, report: ConversionReport) -> Any:
    if isinstance(value, str):
        report.mark("/tool_choice", PRESERVATION_EXACT, "/tool_choice")
        return value
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
        encoded = name_codec.encode(original, "name")
        result = {"type": "function", "name": encoded}
        report.mark("/tool_choice/name", PRESERVATION_EXACT if encoded == original else PRESERVATION_SEMANTIC, "/tool_choice/name")
    else:
        result = "auto"
        report.mark("/tool_choice/type", PRESERVATION_UNSUPPORTED, detail=f"Unknown tool_choice type: {choice_type}")
    if "disable_parallel_tool_use" in value:
        report.mark("/tool_choice/disable_parallel_tool_use", PRESERVATION_SEMANTIC, "/parallel_tool_calls")
    return result


def _convert_tools(
    tools: Any,
    profile: ResponsesWireProfile,
    report: ConversionReport,
    name_codec: IdentifierCodec,
) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        report.mark("/tools", PRESERVATION_UNSUPPORTED, detail="tools must be an array", subtree=True)
        return []
    converted: List[Dict[str, Any]] = []
    for index, tool in enumerate(tools):
        path = f"/tools/{index}"
        if not isinstance(tool, dict):
            report.mark(path, PRESERVATION_UNSUPPORTED, detail="tool definition must be an object", subtree=True)
            continue
        # Anthropic server tools have a provider-specific `type` and may not
        # include input_schema. They are not silently coerced to functions.
        if tool.get("type") and tool.get("type") != "custom" and "input_schema" not in tool:
            report.mark(path, PRESERVATION_UNSUPPORTED, detail=f"Unsupported Anthropic server tool type: {tool.get('type')}", subtree=True)
            continue
        original_name = str(tool.get("name") or "")
        encoded_name = name_codec.encode(original_name, "name")
        target: Dict[str, Any] = {
            "type": "function",
            "name": encoded_name,
            "description": str(tool.get("description") or ""),
            "parameters": copy.deepcopy(tool.get("input_schema") or {"type": "object", "properties": {}}),
            "strict": False,
        }
        converted.append(target)
        target_base = f"/tools/{len(converted)-1}"
        report.mark(path + "/name", PRESERVATION_EXACT if encoded_name == original_name else PRESERVATION_SEMANTIC, target_base + "/name")
        if "description" in tool:
            report.mark(path + "/description", PRESERVATION_EXACT, target_base + "/description")
        if "input_schema" in tool:
            report.mark(path + "/input_schema", PRESERVATION_EXACT, target_base + "/parameters", subtree=True)
        if "type" in tool:
            report.mark(path + "/type", PRESERVATION_SEMANTIC, target_base + "/type")
        for extension in ("cache_control", "defer_loading", "allowed_callers"):
            if extension in tool:
                report.mark(path + "/" + extension, PRESERVATION_SIDECAR, detail=f"Tool extension {extension} has no verified wire mapping", subtree=True)
    return converted


def _mark_known_output_config(payload: Dict[str, Any], report: ConversionReport, responses: Dict[str, Any]) -> None:
    output_config = payload.get("output_config")
    if not isinstance(output_config, dict):
        return
    if "effort" in output_config:
        # Marked by _map_reasoning_effort unless invalid; avoid an unaccounted leaf.
        if not any(record.source_path == "/output_config/effort" for record in report.records):
            report.mark("/output_config/effort", PRESERVATION_UNSUPPORTED, detail="No reasoning effort mapping was selected")
    if "format" in output_config:
        responses.setdefault("text", {})["format"] = copy.deepcopy(output_config["format"])
        report.mark("/output_config/format", PRESERVATION_SEMANTIC, "/text/format", subtree=True)


def convert_anthropic_to_responses(
    payload: Dict[str, Any],
    *,
    wire_profile: str = "copilot_responses_lite",
    mode: str = MODE_COMPATIBILITY,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    replay_resolver: Optional[ReplayResolver] = None,
    sidecar_available: bool = False,
) -> AnthropicToResponsesResult:
    """Convert one Anthropic Messages request to a Responses request."""
    if not isinstance(payload, dict):
        report = ConversionReport("anthropic_to_responses")
        report.mark("/", PRESERVATION_UNSUPPORTED, detail="Request body must be an object", subtree=True)
        raise AnthropicResponsesConversionError("Anthropic request body must be an object", report)
    profile = get_wire_profile(wire_profile)
    report = ConversionReport(
        "anthropic_to_responses", sidecar_available=bool(sidecar_available)
    )
    name_codec = IdentifierCodec()
    call_id_codec = IdentifierCodec()
    input_items: List[Dict[str, Any]] = []
    model = str(payload.get("model") or "")
    responses: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
        "stream": bool(payload.get("stream", False)),
        "include": ["reasoning.encrypted_content"],
    }
    if "model" in payload:
        report.mark("/model", PRESERVATION_EXACT, "/model")
    if "stream" in payload:
        report.mark("/stream", PRESERVATION_EXACT, "/stream")

    if "system" in payload:
        _convert_system(payload["system"], input_items, report, profile)

    tools = _convert_tools(payload.get("tools"), profile, report, name_codec) if "tools" in payload else []
    if tools:
        if profile.tools_in_input:
            input_items.insert(0, {"type": "additional_tools", "role": "developer", "tools": tools})
            # Existing target paths in records are descriptive only; the insert
            # does not alter preservation semantics.
        else:
            responses["tools"] = tools

    replay_misses: List[int] = []
    messages = payload.get("messages")
    messages_error: Optional[str] = None
    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            path = f"/messages/{message_index}"
            if not isinstance(message, dict):
                report.mark(path, PRESERVATION_UNSUPPORTED, detail="message must be an object", subtree=True)
                continue
            replay_items = replay_resolver(message_index, message) if replay_resolver and message.get("role") == "assistant" else None
            if replay_items:
                input_items.extend(copy.deepcopy(replay_items))
                report.mark(path + "/role", PRESERVATION_SIDECAR, detail="Assistant turn restored from replay store")
                report.mark(path + "/content", PRESERVATION_SIDECAR, detail="Visible projection matched replay state", subtree=True)
                continue
            if message.get("role") == "assistant" and replay_resolver is not None:
                replay_misses.append(message_index)
            _append_message_items(input_items, message, message_index, report, profile, name_codec, call_id_codec)
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
    if tool_choice is not None:
        responses["tool_choice"] = _convert_tool_choice(tool_choice, name_codec, report)
        disable_parallel = tool_choice.get("disable_parallel_tool_use") if isinstance(tool_choice, dict) else None
        if disable_parallel is not None:
            responses["parallel_tool_calls"] = not bool(disable_parallel)
    elif tools:
        responses["tool_choice"] = "auto"
    if "parallel_tool_calls" not in responses and tools:
        responses["parallel_tool_calls"] = True

    effort = _map_reasoning_effort(payload, profile, report)
    if effort:
        responses["reasoning"] = {"effort": effort, "context": "all_turns"}
    elif "context_management" in payload:
        responses["reasoning"] = {"context": "all_turns"}

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
        report.mark(
            "/context_management",
            PRESERVATION_SEMANTIC if known else PRESERVATION_UNSUPPORTED,
            "/reasoning/context" if known else None,
            "Mapped clear_thinking keep=all to all_turns" if known else "Unknown context-management edit",
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
                        "Metadata JSON type retained in the full request sidecar; wire value is canonical JSON text",
                        subtree=True,
                    )
            responses["metadata"] = normalized
        else:
            report.mark("/metadata", PRESERVATION_UNSUPPORTED, detail="metadata must be an object", subtree=True)

    if session_id:
        cache_scope = f"{tenant_id or 'anonymous'}\x00{session_id}\x00{model}"
        responses["prompt_cache_key"] = hashlib.sha256(cache_scope.encode("utf-8")).hexdigest()
        if profile.name == "copilot_responses_lite":
            responses["client_metadata"] = {"session_id": session_id}

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
        replay_misses=replay_misses,
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


def _truncate_replay_items_at_stop(
    items: List[Dict[str, Any]],
    stop_sequences: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Apply the client-visible stop boundary to reusable Responses items.

    The unmodified terminal response remains available in the audit snapshot;
    replay state must contain only what the Anthropic client actually observed.
    Otherwise a later turn can feed post-stop text or hidden tool calls back to
    the model even though they were never part of the visible conversation.
    """
    if not stop_sequences:
        return copy.deepcopy(items), None

    for item_index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts = item.get("content")
        if not isinstance(parts, list):
            continue
        carried = ""
        text_positions: List[Tuple[int, str, int, int]] = []
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "output_text":
                text_key = "text"
            elif part_type == "refusal":
                text_key = "refusal"
            else:
                continue
            text_value = str(part.get(text_key) or "")
            start = len(carried)
            carried += text_value
            text_positions.append(
                (part_index, text_key, start, len(carried))
            )
        stop_index, matched = _find_first_stop(carried, stop_sequences)
        if stop_index is None:
            continue
        for part_index, text_key, start, end in text_positions:
            if stop_index >= end:
                continue
            projected = copy.deepcopy(items[: item_index + 1])
            message = projected[-1]
            original_parts = message.get("content")
            if not isinstance(original_parts, list):
                return projected, matched
            message["content"] = copy.deepcopy(original_parts[: part_index + 1])
            matching_part = message["content"][-1]
            matching_part[text_key] = str(matching_part.get(text_key) or "")[
                : stop_index - start
            ]
            return projected, matched

    return copy.deepcopy(items), None


_KNOWN_RESPONSE_SIDECAR_FIELDS = {
    "object", "created_at", "status", "background", "billing", "error",
    "incomplete_details", "instructions", "max_output_tokens", "max_tool_calls",
    "parallel_tool_calls", "previous_response_id", "prompt_cache_key",
    "prompt_cache_retention", "reasoning", "safety_identifier", "service_tier",
    "store", "temperature", "text", "tool_choice", "tools", "top_logprobs",
    "top_p", "truncation", "user", "metadata", "client_metadata",
}


def _responses_usage_to_anthropic(usage: Any) -> Dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    total_input = int(usage.get("input_tokens") or 0)
    cached = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    result: Dict[str, Any] = {
        "input_tokens": max(0, total_input - cached - cache_write),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cached,
    }
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    if reasoning_tokens:
        result["output_tokens_details"] = {"thinking_tokens": reasoning_tokens}
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
                    detail="Empty input token details retained in encrypted audit",
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
                        detail="Unprojected input token detail retained in encrypted audit",
                        subtree=True,
                    )
        elif key == "output_tokens_details" and isinstance(value, dict):
            if not value:
                report.mark(
                    path,
                    PRESERVATION_SIDECAR,
                    detail="Empty output token details retained in encrypted audit",
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
                        detail="Unprojected output token detail retained in encrypted audit",
                        subtree=True,
                    )
        else:
            report.mark(
                path,
                PRESERVATION_SIDECAR,
                detail="Responses usage extension retained in encrypted audit",
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
    name_codec: Optional[IdentifierCodec] = None,
    call_id_codec: Optional[IdentifierCodec] = None,
    stop_sequences: Optional[Sequence[str]] = None,
    mode: str = MODE_COMPATIBILITY,
    sidecar_available: bool = False,
) -> ResponsesToAnthropicResult:
    """Convert a terminal Responses object/event into an Anthropic message."""
    if not isinstance(response_value, dict):
        report = ConversionReport("responses_to_anthropic")
        report.mark("/", PRESERVATION_UNSUPPORTED, detail="Responses body must be an object", subtree=True)
        raise AnthropicResponsesConversionError("Responses body must be an object", report)
    terminal_event_type = response_value.get("type")
    response = _response_object(response_value)
    report = ConversionReport(
        "responses_to_anthropic", sidecar_available=bool(sidecar_available)
    )
    name_codec = name_codec or IdentifierCodec()
    call_id_codec = call_id_codec or IdentifierCodec()
    content: List[Dict[str, Any]] = []
    content_group_ids: List[Any] = []
    replay_items: List[Dict[str, Any]] = []
    has_tool_use = False
    has_refusal = False

    output = response.get("output")
    if isinstance(output, list):
        for index, item in enumerate(output):
            path = f"/output/{index}"
            if not isinstance(item, dict):
                report.mark(path, PRESERVATION_UNSUPPORTED, detail="output item must be an object", subtree=True)
                continue
            replay_items.append(copy.deepcopy(item))
            item_type = item.get("type")
            if item_type == "reasoning":
                report.mark(path, PRESERVATION_SIDECAR, detail="Encrypted reasoning is retained for replay and not exposed as Claude thinking", subtree=True)
            elif item_type == "message":
                report.mark(path + "/type", PRESERVATION_SEMANTIC)
                if "role" in item:
                    report.mark(path + "/role", PRESERVATION_EXACT)
                if "phase" in item:
                    report.mark(path + "/phase", PRESERVATION_SIDECAR, detail="Assistant phase retained for replay")
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
                        report.mark(path + "/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Output message metadata retained for replay", subtree=True)
            elif item_type in ("function_call", "custom_tool_call"):
                has_tool_use = True
                encoded_id = str(item.get("call_id") or item.get("id") or "")
                encoded_name = str(item.get("name") or "")
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
                report.mark(path + "/type", PRESERVATION_SEMANTIC)
                if "call_id" in item:
                    report.mark(path + "/call_id", PRESERVATION_EXACT if original_id == encoded_id else PRESERVATION_SEMANTIC)
                if "id" in item:
                    report.mark(path + "/id", PRESERVATION_SIDECAR, detail="Responses item id retained for replay")
                if "name" in item:
                    report.mark(path + "/name", PRESERVATION_EXACT if original_name == encoded_name else PRESERVATION_SEMANTIC)
                for key in item:
                    if key not in {"type", "call_id", "id", "name", "arguments", "input"}:
                        report.mark(path + "/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Tool-call metadata retained for replay", subtree=True)
            else:
                report.mark(path, PRESERVATION_UNSUPPORTED, detail=f"Unknown Responses output item type: {item_type}", subtree=True)
    else:
        report.mark("/output", PRESERVATION_UNSUPPORTED, detail="Responses output must be an array", subtree=True)

    active_stop_sequences = list(stop_sequences or [])
    content, matched_stop = _truncate_blocks_at_stop(
        content, active_stop_sequences, content_group_ids
    )
    if matched_stop:
        replay_items, replay_stop = _truncate_replay_items_at_stop(
            replay_items, active_stop_sequences
        )
        if replay_stop != matched_stop:
            report.mark(
                "/output",
                PRESERVATION_UNSUPPORTED,
                detail="Visible stop boundary could not be applied to replay items",
                subtree=True,
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
            raise AnthropicResponsesConversionError(
                "Responses incomplete reason is not safely representable",
                report,
            )
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
        "usage": _responses_usage_to_anthropic(response.get("usage")),
    }
    visible = {"role": "assistant", "content": copy.deepcopy(content)}

    if "usage" in response:
        _mark_usage_preservation(response.get("usage"), report)
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
            report.mark("/" + _pointer_escape(key), PRESERVATION_SIDECAR, detail="Responses metadata retained in audit/replay sidecar", subtree=True)
    report.finalize(response, mode)
    return ResponsesToAnthropicResult(
        response=anthropic,
        report=report,
        replay_items=replay_items,
        visible_assistant_message=visible,
        matched_stop_sequence=matched_stop,
    )


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
