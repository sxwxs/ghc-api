"""Content-safe projections for the in-memory dashboard cache.

The encrypted replay store is the authoritative full-fidelity audit trail.
Dashboard records intentionally retain protocol shape and digests, but never
opaque reasoning state or fields from an unknown Responses discriminator.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

from .compat_profiles import audit_responses_event, audit_responses_item


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        return repr(type(value).__name__).encode("ascii")


def redacted_value(value: Any, reason: str) -> Dict[str, Any]:
    raw = _canonical_bytes(value)
    return {
        "_redacted": True,
        "_reason": reason,
        "_size": len(raw),
        "_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _redact_reasoning_item(item: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(item)
    for key in ("encrypted_content", "summary", "content"):
        if key in result and result[key] is not None:
            result[key] = redacted_value(result[key], "opaque reasoning state")
    return result


def _redact_web_search_call(item: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(item)
    if "action" in result:
        result["action"] = redacted_value(result["action"], "web search request details")
    return result


def redact_responses_value_for_cache(value: Any) -> Any:
    """Redact hidden state while retaining known Responses protocol shape."""

    if isinstance(value, list):
        return [redact_responses_value_for_cache(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    value_type = value.get("type")
    if value_type == "reasoning":
        return _redact_reasoning_item(value)
    if value_type == "web_search_call":
        return _redact_web_search_call(value)

    result: Dict[str, Any] = {}
    for key, item in value.items():
        if key in ("encrypted_content", "signature"):
            result[key] = redacted_value(item, "opaque model state")
        elif key == "annotations" and isinstance(item, list):
            result[key] = redacted_value(item, "web search citation details")
        elif key == "output" and isinstance(item, list):
            result[key] = [
                _redact_reasoning_item(part)
                if isinstance(part, dict) and part.get("type") == "reasoning"
                else _redact_web_search_call(part)
                if isinstance(part, dict) and part.get("type") == "web_search_call"
                else redact_responses_value_for_cache(part)
                for part in item
            ]
        elif key == "item" and isinstance(item, dict) and item.get("type") == "reasoning":
            result[key] = _redact_reasoning_item(item)
        elif key == "item" and isinstance(item, dict) and item.get("type") == "web_search_call":
            result[key] = _redact_web_search_call(item)
        else:
            result[key] = redact_responses_value_for_cache(item)
    return result


def redact_responses_event_for_cache(event: Any) -> Any:
    """Return a safe dashboard projection for one upstream SSE event."""

    audit = audit_responses_event(event, mode="compatibility")
    if audit.warnings:
        event_type = event.get("type") if isinstance(event, dict) else None
        result = redacted_value(event, "unknown, invalid, or drifted Responses event")
        if isinstance(event_type, str):
            result["type"] = event_type
        return result

    if isinstance(event, dict) and event.get("type") == "response.output_text.annotation.added":
        result = {
            key: copy.deepcopy(item)
            for key, item in event.items()
            if key in {"type", "sequence_number", "output_index", "content_index", "item_id", "annotation_index"}
        }
        if "annotation" in event:
            result["annotation"] = redacted_value(event["annotation"], "web search citation details")
        return result

    if isinstance(event, dict) and str(event.get("type") or "").startswith(
        "response.reasoning"
    ):
        safe_keys = {
            "type",
            "sequence_number",
            "response_id",
            "item_id",
            "output_index",
            "content_index",
            "summary_index",
        }
        result = {
            key: copy.deepcopy(item)
            for key, item in event.items()
            if key in safe_keys
        }
        hidden = {key: item for key, item in event.items() if key not in safe_keys}
        if hidden:
            result["payload"] = redacted_value(hidden, "reasoning stream payload")
        return result
    return redact_responses_value_for_cache(event)


def redact_responses_item_for_cache(item: Any) -> Any:
    audit = audit_responses_item(item, mode="compatibility")
    if audit.warnings:
        return redacted_value(item, "unknown, invalid, or drifted Responses item")
    return redact_responses_value_for_cache(item)


def redact_responses_response_for_cache(response: Any) -> Any:
    """Project a non-stream Responses body without retaining drift payloads."""

    if not isinstance(response, dict):
        return redacted_value(response, "invalid Responses body")
    event_type = {
        "completed": "response.completed",
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }.get(str(response.get("status") or ""), "response.unknown_status")
    audit = audit_responses_event({
        "type": event_type,
        "sequence_number": 0,
        "response": response,
    })
    if audit.warnings:
        return redacted_value(response, "unknown, invalid, or drifted Responses body")
    return redact_responses_value_for_cache(response)
