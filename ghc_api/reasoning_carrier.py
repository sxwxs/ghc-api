"""Stateless Responses-reasoning carrier for Anthropic Messages history.

Responses returns an opaque ``encrypted_content`` value that must be echoed in a
later Responses request to preserve reasoning state.  Anthropic Messages has no
matching field, but assistant ``thinking.signature`` is itself an opaque string
that well-behaved clients preserve in conversation history.  This module uses a
namespaced, versioned envelope in that field so the state travels with the
assistant message instead of being kept in a server-side database.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


REASONING_CARRIER_PREFIX = "ghc-api:responses-reasoning:v1:"
MAX_REASONING_CARRIER_CHARS = 1024 * 1024
_BASE64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ReasoningCarrier:
    model: str
    wire_profile: str
    item_id: Optional[str]
    encrypted_content: Optional[str]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")


def build_reasoning_carrier(
    *,
    model: str,
    wire_profile: str,
    encrypted_content: Optional[str],
    item_id: Optional[str] = None,
) -> str:
    """Build a client-carried envelope for one Responses reasoning item."""
    if not isinstance(model, str) or not model:
        raise ValueError("reasoning carrier model must be a non-empty string")
    if not isinstance(wire_profile, str) or not wire_profile:
        raise ValueError("reasoning carrier wire_profile must be a non-empty string")
    if item_id is not None and not isinstance(item_id, str):
        raise TypeError("reasoning carrier item_id must be a string or None")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise TypeError("reasoning carrier encrypted_content must be a string or None")
    payload = _canonical_json_bytes({
        "encrypted_content": encrypted_content,
        "item_id": item_id,
        "model": model,
        "wire_profile": wire_profile,
    })
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    signature = REASONING_CARRIER_PREFIX + encoded
    if len(signature) > MAX_REASONING_CARRIER_CHARS:
        raise ValueError("reasoning carrier exceeds the maximum supported size")
    return signature


def is_reasoning_carrier(signature: Any) -> bool:
    return isinstance(signature, str) and signature.startswith(REASONING_CARRIER_PREFIX)


def parse_reasoning_carrier(signature: Any) -> Optional[ReasoningCarrier]:
    """Strictly decode one carrier, returning ``None`` for foreign signatures.

    A namespaced but malformed value raises ``ValueError``.  This distinction
    lets callers drop corrupt opaque state while still preserving the visible
    reasoning summary and reporting a compatibility warning.
    """
    if not is_reasoning_carrier(signature):
        return None
    if len(signature) > MAX_REASONING_CARRIER_CHARS:
        raise ValueError("reasoning carrier exceeds the maximum supported size")
    encoded = signature[len(REASONING_CARRIER_PREFIX):]
    if not encoded or not _BASE64URL_RE.fullmatch(encoded) or len(encoded) % 4 == 1:
        raise ValueError("reasoning carrier has invalid base64url encoding")
    padded = encoded + "=" * ((4 - len(encoded) % 4) % 4)
    try:
        raw = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
    except Exception as exc:
        raise ValueError("reasoning carrier has invalid base64url encoding") from exc
    canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if canonical != encoded:
        raise ValueError("reasoning carrier is not canonically encoded")

    def reject_constant(value: str) -> None:
        raise ValueError("reasoning carrier contains a non-finite number")

    def unique_object(pairs: list[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("reasoning carrier contains a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("reasoning carrier payload is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "encrypted_content", "item_id", "model", "wire_profile"
    }:
        raise ValueError("reasoning carrier payload has an unsupported shape")
    model = value.get("model")
    wire_profile = value.get("wire_profile")
    item_id = value.get("item_id")
    encrypted_content = value.get("encrypted_content")
    if not isinstance(model, str) or not model:
        raise ValueError("reasoning carrier model is invalid")
    if not isinstance(wire_profile, str) or not wire_profile:
        raise ValueError("reasoning carrier wire_profile is invalid")
    if item_id is not None and not isinstance(item_id, str):
        raise ValueError("reasoning carrier item_id is invalid")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise ValueError("reasoning carrier encrypted_content is invalid")
    return ReasoningCarrier(model, wire_profile, item_id, encrypted_content)


def redact_reasoning_carriers_for_cache(value: Any) -> Any:
    """Replace carrier payloads with bounded diagnostics before caching."""
    if isinstance(value, list):
        return [redact_reasoning_carriers_for_cache(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result: Dict[str, Any] = {}
    for key, child in value.items():
        if key == "signature" and is_reasoning_carrier(child):
            encoded = str(child).encode("utf-8", errors="strict")
            result[key] = (
                "[Responses reasoning carrier: "
                f"{len(encoded)} bytes, sha256={hashlib.sha256(encoded).hexdigest()}]"
            )
        else:
            result[key] = redact_reasoning_carriers_for_cache(child)
    return result


def strip_reasoning_carriers_from_messages_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Remove synthetic reasoning blocks before a native Anthropic request.

    The carrier signature is meaningful only when the next hop is Responses.
    Sending it to a native Anthropic model would present a forged thinking
    signature and can make the upstream reject the request.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, 0
    changed = False
    removed = 0
    new_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            new_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            new_messages.append(message)
            continue
        filtered = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and is_reasoning_carrier(block.get("signature"))
            ):
                removed += 1
                changed = True
                continue
            filtered.append(block)
        if not filtered and len(filtered) != len(content):
            # A synthetic encrypted-only turn has no native Anthropic content
            # once its forged signature is removed. Dropping the empty turn is
            # safer than forwarding an invalid empty assistant message.
            continue
        new_messages.append(
            {**message, "content": filtered} if len(filtered) != len(content) else message
        )
    if not changed:
        return payload, 0
    result = copy.copy(payload)
    result["messages"] = new_messages
    return result, removed


__all__ = [
    "MAX_REASONING_CARRIER_CHARS",
    "REASONING_CARRIER_PREFIX",
    "ReasoningCarrier",
    "build_reasoning_carrier",
    "is_reasoning_carrier",
    "parse_reasoning_carrier",
    "redact_reasoning_carriers_for_cache",
    "strip_reasoning_carriers_from_messages_payload",
]
