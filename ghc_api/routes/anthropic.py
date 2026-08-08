"""
Anthropic-compatible API routes
"""

import base64
import copy
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Any, List, Optional, Tuple

import requests
from flask import Blueprint, Response, g, jsonify, request, stream_with_context

from ..api_helpers import (
    ensure_copilot_token,
    get_copilot_base_url,
    get_copilot_headers,
    anthropic_responses_wire_profile,
    supports_direct_anthropic_api,
    supports_responses_api,
    supported_reasoning_efforts,
    count_tokens,
)
from ..anthropic_responses import (
    MODE_COMPATIBILITY,
    MODE_LOSSLESS_REQUIRED,
    AnthropicResponsesConversionError,
    ResponsesToAnthropicResult,
    StrictJSONError,
    anthropic_error_from_responses,
    convert_anthropic_to_responses,
    convert_responses_to_anthropic,
    parse_strict_json_bytes,
    prepare_replay_items_for_wire,
)
from ..compat_profiles import (
    CLAUDE_CLI_TOOL_CONTRACT_BASELINES,
    audit_anthropic_request,
    audit_responses_event,
)
from ..compat_redaction import (
    redact_responses_response_for_cache,
    redact_responses_value_for_cache,
)
from ..cache import cache
from ..counters import counters
from ..auth import redact_auth_headers
from ..streaming import AnthropicStreamState, reconstruct_openai_response_from_chunks, translate_chunk_to_anthropic_events
from ..sse import (
    AnthropicDirectStreamHandler,
    AnthropicDirectStreamHandlerWithRecovery,
    AnthropicResponsesStreamHandler,
)
from ..sse.keepalive import (
    BackgroundResult,
    KEEPALIVE,
    iter_lines_with_keepalive,
    wait_result_with_keepalive,
)
from ..translator import (
    translate_anthropic_to_openai,
    translate_model_name,
    translate_openai_to_anthropic,
    apply_system_prompt_filters,
    apply_tool_result_suffix_filter,
)
from ..utils import log_error_request, is_orphaned_tool_result_error, remove_orphaned_tool_results, extract_orphaned_tool_use_ids, log_tool_result_cleanup, log_connection_retry, get_client_ip
from ..state import state
from ..reasoning_replay import (
    ReplayEncryptionConfigurationError,
    ReplayQuotaExceededError,
    ReasoningReplayStore,
)
from ..utils import get_config_dir
from ..web_search import has_web_search_tool, is_web_search_unsupported_error, apply_web_search_fallback


def _current_user_id() -> str:
    """Read user_id from flask.g; falls back to anonymous outside a request context."""
    return getattr(g, "user_id", "anonymous") or "anonymous"

anthropic_bp = Blueprint('anthropic', __name__)


def _compatibility_warning(
    code: str,
    path: str = "/",
    action: str = "warning",
    **safe_fields: Any,
) -> Dict[str, Any]:
    """Build a warning that cannot accidentally contain request content."""
    warning: Dict[str, Any] = {
        "code": str(code),
        "path": str(path),
        "action": str(action),
    }
    for key in ("profile", "cli_version", "observed_type", "fingerprint"):
        value = safe_fields.get(key)
        if isinstance(value, str) and value:
            warning[key] = value
    return warning


def _sanitise_compatibility_warning(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return _compatibility_warning("compatibility.invalid_warning")
    return _compatibility_warning(
        value.get("code") or "compatibility.warning",
        value.get("path") or "/",
        value.get("action") or "warning",
        profile=value.get("profile"),
        cli_version=value.get("cli_version") or value.get("version"),
        observed_type=value.get("observed_type"),
        fingerprint=value.get("fingerprint"),
    )


def _merge_compatibility_warnings(*groups: Any) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        if not group:
            continue
        for raw in group:
            warning = _sanitise_compatibility_warning(raw)
            key = tuple(sorted(warning.items()))
            if key not in seen:
                seen.add(key)
                merged.append(warning)
    return merged


_warning_log_lock = threading.Lock()
_warning_log_times: Dict[Tuple[str, str, str], float] = {}


def _log_compatibility_warnings(
    request_id: str,
    warnings: List[Dict[str, Any]],
) -> None:
    """Rate-limit console noise while retaining every warning in request cache."""
    now = time.time()
    for warning in warnings:
        code = str(warning.get("code") or "compatibility.warning")
        path = str(warning.get("path") or "/")
        profile = str(warning.get("profile") or "")
        cli_version = str(warning.get("cli_version") or "")
        fingerprint = str(warning.get("fingerprint") or "")
        key = (code, path, profile + "\x00" + cli_version + "\x00" + fingerprint)
        with _warning_log_lock:
            last = _warning_log_times.get(key, 0.0)
            should_log = now - last >= 300
            if should_log:
                _warning_log_times[key] = now
            # Bound a long-running process even under adversarial field drift.
            if len(_warning_log_times) > 4096:
                oldest = sorted(_warning_log_times.items(), key=lambda item: item[1])[:1024]
                for old_key, _ in oldest:
                    _warning_log_times.pop(old_key, None)
        counters.incr("compat." + code.replace(".", "_"))
        if should_log:
            print(
                "WARNING [AnthropicResponsesCompat] "
                f"code={code} request_id={request_id} path={path}"
                + (f" profile={profile}" if profile else "")
                + (f" cli_version={cli_version}" if cli_version else "")
            )


def _compatibility_header_value(warnings: List[Dict[str, Any]]) -> Optional[str]:
    codes = sorted({str(item.get("code")) for item in warnings if item.get("code")})
    if not codes:
        return None
    # This is a diagnostic hint, not the authoritative report. Keep it within
    # conservative proxy header limits; the complete list stays in cache.
    return ",".join(codes)[:1024]


def _set_compatibility_headers(response: Response, warnings: List[Dict[str, Any]]) -> Response:
    value = _compatibility_header_value(warnings)
    if value:
        response.headers["X-GHC-Compatibility-Warnings"] = value
    return response


def _anthropic_json_error(
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    warnings: Optional[List[Dict[str, Any]]] = None,
) -> Response:
    response = jsonify({
        "type": "error",
        "error": {"type": error_type, "message": str(message)},
    })
    response.status_code = status_code
    return _set_compatibility_headers(response, warnings or [])


def _header_value(headers: Dict[str, Any], *names: str) -> Optional[str]:
    lowered = {str(key).lower(): value for key, value in (headers or {}).items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_anthropic_session_id(
    headers: Dict[str, Any], payload: Dict[str, Any]
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []
    header_session = _header_value(
        headers,
        "X-Claude-Code-Session-Id",
        "X-Session-Id",
    )
    metadata_session: Optional[str] = None
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        user_value = metadata.get("user_id")
        if isinstance(user_value, str):
            try:
                parsed = json.loads(user_value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("session_id") is not None:
                candidate = str(parsed.get("session_id") or "").strip()
                metadata_session = candidate or None
    if header_session and metadata_session and header_session != metadata_session:
        warnings.append(_compatibility_warning(
            "replay.session_identity_mismatch", "/metadata/user_id", "disabled"
        ))
        return None, warnings
    session_id = header_session or metadata_session
    if not session_id:
        warnings.append(_compatibility_warning(
            "replay.session_identity_missing", "/metadata", "disabled"
        ))
    return session_id, warnings


_replay_store_lock = threading.RLock()
_replay_store: Optional[ReasoningReplayStore] = None
_replay_store_signature: Optional[Tuple[Any, ...]] = None


def _configured_replay_path() -> str:
    configured = str(getattr(state, "anthropic_responses_replay_path", "") or "").strip()
    if configured:
        return configured
    return str(Path(get_config_dir()) / "anthropic-responses-replay.sqlite3")


def _reset_anthropic_responses_replay_store() -> None:
    """Close the lazy replay store (also used by runtime-config tests)."""
    global _replay_store, _replay_store_signature
    with _replay_store_lock:
        if _replay_store is not None:
            _replay_store.close()
        _replay_store = None
        _replay_store_signature = None


def _get_anthropic_responses_replay_store() -> Tuple[ReasoningReplayStore, str]:
    global _replay_store, _replay_store_signature
    path = _configured_replay_path()
    ttl = int(getattr(state, "anthropic_responses_replay_ttl_seconds", 86400))
    key_env = str(getattr(
        state, "anthropic_responses_replay_encryption_key_env", ""
    ) or "").strip()
    encryption_key = os.environ.get(key_env) if key_env else None
    key_fingerprint = (
        hashlib.sha256(encryption_key.encode("utf-8")).hexdigest()
        if encryption_key else ""
    )
    signature = (str(Path(path).expanduser()) if path != ":memory:" else path, ttl, key_env, key_fingerprint)
    with _replay_store_lock:
        if _replay_store is not None and signature == _replay_store_signature:
            return _replay_store, path
        if _replay_store is not None:
            _replay_store.close()
        _replay_store = ReasoningReplayStore(
            path,
            ttl_seconds=ttl,
            encryption_key=encryption_key,
            require_encryption=True,
        )
        _replay_store_signature = signature
        return _replay_store, path


def _assistant_visible_blocks(message: Dict[str, Any]) -> List[Any]:
    content = message.get("content")
    if isinstance(content, list):
        return copy.deepcopy(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


@dataclass
class _ReplayContext:
    mode: str
    wire_profile: str
    model: str
    tenant_id: Optional[str]
    session_id: Optional[str]
    store: Optional[ReasoningReplayStore] = None
    max_bytes: int = 0
    max_tenant_bytes: int = 0
    max_record_bytes: int = 0
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    parent_replay_id: Optional[str] = None
    matched_profiles: List[Dict[str, Any]] = field(default_factory=list)
    fatal_message: Optional[str] = None

    def warn(self, code: str, path: str = "/", action: str = "warning") -> None:
        warning = _compatibility_warning(code, path, action, profile=self.wire_profile)
        if warning not in self.warnings:
            self.warnings.append(warning)

    def fail_if_lossless(self, message: str) -> None:
        if self.mode == MODE_LOSSLESS_REQUIRED and self.fatal_message is None:
            self.fatal_message = message


def _create_replay_context(
    *,
    payload: Dict[str, Any],
    request_headers: Dict[str, Any],
    user_id: str,
    model: str,
    wire_profile: str,
    mode: str,
) -> _ReplayContext:
    session_id, session_warnings = _extract_anthropic_session_id(request_headers, payload)
    trusted_single_user = bool(getattr(
        state, "anthropic_responses_replay_trusted_single_user", False
    ))
    require_trusted = bool(getattr(
        state, "anthropic_responses_replay_require_trusted_tenant", True
    ))
    if bool(getattr(state, "enable_auth", False)) and user_id and user_id != "anonymous":
        tenant_id: Optional[str] = user_id
    elif trusted_single_user:
        tenant_id = "trusted-single-user"
    elif not require_trusted:
        tenant_id = user_id or "anonymous"
    else:
        tenant_id = None

    context = _ReplayContext(
        mode=mode,
        wire_profile=wire_profile,
        model=model,
        tenant_id=tenant_id,
        session_id=session_id,
        max_bytes=int(getattr(state, "anthropic_responses_replay_max_bytes", 0) or 0),
        max_tenant_bytes=int(getattr(
            state, "anthropic_responses_replay_max_tenant_bytes", 0
        ) or 0),
        max_record_bytes=int(getattr(
            state, "anthropic_responses_replay_max_record_bytes", 0
        ) or 0),
        warnings=list(session_warnings),
    )
    if session_warnings:
        context.fail_if_lossless("A stable Claude Code session identity is required for lossless reasoning replay")
    if tenant_id is None:
        context.warn("replay.trusted_tenant_missing", "/", "disabled")
        context.fail_if_lossless("A trusted tenant identity is required for lossless reasoning replay")
    if session_id is None or tenant_id is None:
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        return context

    try:
        store, _ = _get_anthropic_responses_replay_store()
        context.store = store
        store.purge(expired_only=True)
    except ReplayEncryptionConfigurationError:
        context.warn("replay.encryption_required", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("Replay encryption is required but is not configured")
        return context
    except Exception:
        context.warn("replay.store_unavailable", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("The reasoning replay store is unavailable")
        return context

    if (
        context.max_bytes <= 0
        or context.max_tenant_bytes <= 0
        or context.max_record_bytes <= 0
    ):
        context.warn("replay.invalid_quota", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("Replay store byte quotas must be positive")
        context.store = None
    elif context.max_record_bytes > context.max_tenant_bytes:
        context.warn("replay.invalid_quota", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("Replay per-record quota exceeds the tenant quota")
        context.store = None
    elif context.max_tenant_bytes > context.max_bytes:
        context.warn("replay.invalid_quota", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("Replay tenant quota exceeds the total quota")
        context.store = None
    elif store.logical_size_bytes() >= context.max_bytes:
        context.warn("replay.quota_exhausted", "/", "disabled")
        context.warn("audit.full_snapshot_unavailable", "/", "not_stored")
        context.fail_if_lossless("Reasoning replay store byte quota is exhausted")
        context.store = None
    return context


def _replay_resolver(context: _ReplayContext):
    if context.store is None or not context.tenant_id or not context.session_id:
        return None

    def resolve(message_index: int, message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        path = f"/messages/{message_index}"
        try:
            lookup = context.store.get(
                tenant_id=context.tenant_id,
                session_id=context.session_id,
                model=context.model,
                assistant_visible_blocks=_assistant_visible_blocks(message),
            )
        except Exception:
            context.warn("replay.lookup_failed", path, "miss")
            context.fail_if_lossless("Reasoning replay lookup failed")
            return None
        if lookup.issues:
            context.warn("replay.integrity_failed", path, "miss")
            context.fail_if_lossless("Reasoning replay state failed integrity validation")
            return None
        candidates = [
            record for record in lookup.records
            if record.parent_replay_id == context.parent_replay_id
        ]
        if len(candidates) > 1:
            context.warn("replay.ambiguous", path, "miss")
            context.fail_if_lossless("Reasoning replay state is ambiguous after a retry or fork")
            return None
        if not candidates:
            context.warn("replay.state_miss", path, "miss")
            context.fail_if_lossless("Required reasoning replay state was not found")
            return None
        record = candidates[0]
        profile_data = record.profile if isinstance(record.profile, dict) else {}
        if profile_data.get("name") != context.wire_profile:
            context.warn("replay.profile_mismatch", path, "miss")
            context.fail_if_lossless("Reasoning replay state belongs to a different wire profile")
            return None
        try:
            wire_items = prepare_replay_items_for_wire(
                record.output_items, context.wire_profile
            )
        except Exception:
            context.warn("replay.wire_encoding_failed", path, "miss")
            context.fail_if_lossless("Stored reasoning state cannot be encoded for this wire profile")
            return None
        context.parent_replay_id = record.replay_id
        context.matched_profiles.append({
            "name": profile_data.get("name"),
            "name_codec": copy.deepcopy(profile_data.get("name_codec", {})),
            "call_id_codec": copy.deepcopy(profile_data.get("call_id_codec", {})),
        })
        return wire_items

    return resolve


def _restore_replay_identifier_mappings(context: _ReplayContext, conversion: Any) -> None:
    try:
        for profile in context.matched_profiles:
            conversion.name_codec.restore_mapping(profile.get("name_codec", {}))
            conversion.call_id_codec.restore_mapping(profile.get("call_id_codec", {}))
    except Exception:
        context.warn("replay.identifier_mapping_conflict", "/messages", "miss")
        context.fail_if_lossless("Replay identifier mapping conflicts with this request")


def _persist_replay_result(
    context: _ReplayContext,
    conversion: Any,
    result: ResponsesToAnthropicResult,
    audit_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    if context.store is None or not context.tenant_id or not context.session_id:
        return
    profile = {
        "name": context.wire_profile,
        "name_codec": conversion.name_codec.mapping(),
        "call_id_codec": conversion.call_id_codec.mapping(),
    }
    if audit_snapshot is not None:
        profile["audit_snapshot"] = copy.deepcopy(audit_snapshot)
    try:
        record = context.store.put(
            tenant_id=context.tenant_id,
            session_id=context.session_id,
            model=context.model,
            output_items=result.replay_items,
            assistant_visible_blocks=result.visible_assistant_message.get("content", []),
            profile=profile,
            parent_replay_id=context.parent_replay_id,
            max_record_bytes=context.max_record_bytes,
            max_tenant_bytes=context.max_tenant_bytes,
            max_total_bytes=context.max_bytes,
        )
        context.parent_replay_id = record.replay_id
    except ReplayQuotaExceededError:
        context.warn("replay.quota_exhausted", "/response/output", "not_stored")
        context.warn("audit.full_snapshot_unavailable", "/response", "not_stored")
        context.fail_if_lossless("Reasoning replay result exceeded a configured byte quota")
    except Exception:
        context.warn("replay.write_failed", "/response/output", "not_stored")
        context.warn("audit.full_snapshot_unavailable", "/response", "not_stored")
        context.fail_if_lossless("Reasoning replay result could not be persisted")


def _persist_audit_only_snapshot(
    context: _ReplayContext,
    request_id: str,
    snapshot: Dict[str, Any],
) -> None:
    """Persist a failed/unknown exchange without making it replay-eligible."""
    if context.store is None or not context.tenant_id or not context.session_id:
        return
    try:
        context.store.put(
            tenant_id=context.tenant_id,
            session_id=context.session_id,
            model=context.model,
            output_items=[],
            assistant_visible_blocks=[{
                "type": "ghc_compatibility_audit_record",
                "request_id": request_id,
            }],
            profile={
                "name": context.wire_profile,
                "audit_only": True,
                "audit_snapshot": copy.deepcopy(snapshot),
            },
            max_record_bytes=context.max_record_bytes,
            max_tenant_bytes=context.max_tenant_bytes,
            max_total_bytes=context.max_bytes,
        )
    except ReplayQuotaExceededError:
        context.warn("audit.quota_exhausted", "/response", "not_stored")
    except Exception:
        context.warn("audit.write_failed", "/response", "not_stored")


def _compatibility_audit_snapshot(
    *,
    original_request_raw: bytes,
    original_request_body: Dict[str, Any],
    request_headers: Dict[str, Any],
    responses_payload: Dict[str, Any],
    conversion: Any,
    request_audit: Any,
    warnings: List[Dict[str, Any]],
    raw_response_events: Optional[List[str]] = None,
    raw_response_sse_lines: Optional[List[str]] = None,
    upstream_response_raw: Optional[bytes] = None,
    response_report: Any = None,
) -> Dict[str, Any]:
    """Complete encrypted sidecar payload stored with a replay DAG node."""
    snapshot: Dict[str, Any] = {
        "version": 1,
        "request_raw_base64": base64.b64encode(original_request_raw).decode("ascii"),
        "parsed_request": copy.deepcopy(original_request_body),
        "request_headers": copy.deepcopy(request_headers),
        "responses_request": copy.deepcopy(responses_payload),
        "request_conversion_report": conversion.report.to_dict(),
        "request_compatibility_audit": request_audit.to_dict(),
        "compatibility_warnings": copy.deepcopy(warnings),
    }
    if raw_response_events is not None:
        snapshot["raw_response_events"] = list(raw_response_events)
    if raw_response_sse_lines is not None:
        snapshot["raw_response_sse_lines"] = list(raw_response_sse_lines)
    if upstream_response_raw is not None:
        snapshot["upstream_response_raw_base64"] = base64.b64encode(
            upstream_response_raw
        ).decode("ascii")
    if response_report is not None:
        snapshot["response_conversion_report"] = response_report.to_dict()
    return snapshot


def _upstream_response_bytes(response: requests.Response) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    return str(getattr(response, "text", "") or "").encode("utf-8")


# Fields supported by Copilot's Anthropic API endpoint.
# output_config is gated upstream by apply_effort_policy, so it only reaches the
# filter when the target model supports the requested effort value.
COPILOT_SUPPORTED_FIELDS = {
    "model", "messages", "max_tokens", "system", "metadata",
    "stop_sequences", "stream", "temperature", "top_p", "top_k",
    "tools", "tool_choice", "thinking", "service_tier", "output_config",
}

def apply_effort_policy(payload: Dict, translated_model: str) -> Dict:
    """Keep output_config.effort only if Copilot reports the model supports
    that exact value; otherwise drop output_config. No value normalization."""
    output_config = payload.get("output_config")
    if not isinstance(output_config, dict):
        return payload
    eff = output_config.get("effort")
    if eff is None:
        return payload

    supported = supported_reasoning_efforts(translated_model)
    if eff in supported:
        return payload  # forward as-is

    print(f"[Effort] Model {translated_model} does not support effort={eff} "
          f"(supported: {sorted(supported) or 'none'}); dropping output_config")
    counters.incr("mod.effort_policy_filter")
    return {k: v for k, v in payload.items() if k != "output_config"}


def translate_thinking_enabled_to_adaptive(payload: Dict, translated_model: str) -> Dict:
    """Translate legacy thinking.type=enabled to the new adaptive protocol when
    the target model only accepts the new one.

    Newer Copilot-served Claude models (Opus 4.7+) reject thinking.type=enabled
    with a 400 ("Use thinking.type.adaptive and output_config.effort"). This shim
    auto-translates so old clients keep working without code changes.

    Trigger: thinking.type=="enabled" AND the model reports a non-empty
    reasoning_effort capability. Models without reasoning_effort are treated as
    old-protocol and pass through unchanged.

    Mapping (budget_tokens -> effort): <4096 low, <16384 medium, >=16384 high.
    xhigh/max are intentionally not auto-selected since not every effort-aware
    model supports them; apply_effort_policy gates the result either way.

    A client-supplied output_config.effort always wins over the mapped value.
    max_tokens is bumped to preserve response headroom that the original
    budget_tokens implied (mirrors adjust_max_tokens_for_thinking, since
    budget_tokens is dropped by this translation).
    """
    thinking = payload.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return payload

    if not supported_reasoning_efforts(translated_model):
        return payload

    budget_tokens = thinking.get("budget_tokens") or 0
    if budget_tokens and budget_tokens < 4096:
        mapped_effort = "low"
    elif budget_tokens < 16384:
        mapped_effort = "medium"
    else:
        mapped_effort = "high"

    new_payload = {**payload, "thinking": {"type": "adaptive"}}

    existing_oc = payload.get("output_config")
    if isinstance(existing_oc, dict) and existing_oc.get("effort") is not None:
        print(f"[ThinkingTranslate] {translated_model}: enabled(budget={budget_tokens}) "
              f"-> adaptive; preserving client effort={existing_oc.get('effort')}")
    else:
        merged_oc = dict(existing_oc) if isinstance(existing_oc, dict) else {}
        merged_oc["effort"] = mapped_effort
        new_payload["output_config"] = merged_oc
        print(f"[ThinkingTranslate] {translated_model}: enabled(budget={budget_tokens}) "
              f"-> adaptive + effort={mapped_effort}")

    max_tokens = new_payload.get("max_tokens", 0)
    if budget_tokens and max_tokens <= budget_tokens:
        response_buffer = min(16384, budget_tokens)
        new_max_tokens = budget_tokens + response_buffer
        print(f"[ThinkingTranslate] Adjusted max_tokens: {max_tokens} -> {new_max_tokens} "
              f"(preserving headroom from original budget_tokens={budget_tokens})")
        new_payload["max_tokens"] = new_max_tokens

    counters.incr("mod.thinking_protocol_translate")
    return new_payload


def _remove_scope_from_ephemeral_cache_control(block: Dict) -> None:
    """Remove 'scope' key from a block's cache_control if type is 'ephemeral'."""
    cc = block.get("cache_control")
    if cc and isinstance(cc, dict) and cc.get("type") == "ephemeral" and "scope" in cc:
        print(f"[DirectAnthropic] Removing `scope` from cache_control in content block: {cc}")
        cc.pop("scope")
        counters.incr("mod.cache_control_scope_removal")


def _remove_scope_from_cache_control_in_payload(payload: Dict) -> None:
    """Remove 'scope' from ephemeral cache_control in tools, message content blocks and system prompt blocks."""
    # Handle tool definitions
    for tool in payload.get("tools", []):
        if isinstance(tool, dict):
            _remove_scope_from_ephemeral_cache_control(tool)

    # Handle system prompt blocks
    system = payload.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                _remove_scope_from_ephemeral_cache_control(block)

    # Handle message content blocks
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    _remove_scope_from_ephemeral_cache_control(block)


def filter_payload_for_copilot(payload: Dict) -> Dict:
    """Filter payload to only include fields supported by Copilot's Anthropic API."""
    filtered = {}
    unsupported_fields = []

    for key, value in payload.items():
        if key in COPILOT_SUPPORTED_FIELDS:
            filtered[key] = copy.deepcopy(value)
        else:
            unsupported_fields.append(key)

    if unsupported_fields:
        print(f"[DirectAnthropic] Filtered unsupported fields: {', '.join(unsupported_fields)}")
        counters.incr("mod.unsupported_field_filter")

    # Remove scope from ephemeral cache_control in tools, messages, and system prompt
    _remove_scope_from_cache_control_in_payload(filtered)

    return filtered


def adjust_max_tokens_for_thinking(payload: Dict) -> Dict:
    """Adjust max_tokens if thinking is enabled.

    According to Anthropic docs, max_tokens must be greater than thinking.budget_tokens.
    """
    thinking = payload.get("thinking")
    if not thinking:
        return payload

    budget_tokens = thinking.get("budget_tokens")
    if not budget_tokens:
        return payload

    max_tokens = payload.get("max_tokens", 0)
    if max_tokens <= budget_tokens:
        response_buffer = min(16384, budget_tokens)
        new_max_tokens = budget_tokens + response_buffer
        print(f"[DirectAnthropic] Adjusted max_tokens: {max_tokens} → {new_max_tokens} (thinking.budget_tokens={budget_tokens})")
        counters.incr("mod.thinking_max_tokens_adjust")
        return {**payload, "max_tokens": new_max_tokens}

    return payload


def get_anthropic_headers(enable_vision: bool = False) -> Dict[str, str]:
    """Get headers for direct Anthropic API requests to Copilot."""
    headers = get_copilot_headers(enable_vision)
    headers["anthropic-version"] = "2023-06-01"
    return headers


def _direct_anthropic_stream_handler_cls():
    return (
        AnthropicDirectStreamHandlerWithRecovery
        if state.enable_tool_call_recovery
        else AnthropicDirectStreamHandler
    )


def _start_direct_anthropic_post(headers: Dict, payload: Dict, stream: bool):
    return BackgroundResult(lambda: requests.post(
        f"{get_copilot_base_url()}/v1/messages",
        headers=headers,
        json=payload,
        timeout=state.upstream_read_timeout,
        stream=stream,
    ))


def _anthropic_ping_event() -> str:
    return 'event: ping\ndata: {"type": "ping"}\n\n'


def _parse_response_body(response: requests.Response):
    try:
        return response.json()
    except Exception:
        return response.text


def _anthropic_error_event(response_body, status_code: int) -> str:
    if isinstance(response_body, dict):
        if response_body.get("type") == "error" and isinstance(response_body.get("error"), dict):
            event = response_body
        elif isinstance(response_body.get("error"), dict):
            event = {"type": "error", "error": response_body["error"]}
        else:
            event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": json.dumps(response_body),
                },
            }
    else:
        event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(response_body) or f"Upstream returned HTTP {status_code}",
            },
        }
    return f"event: error\ndata: {json.dumps(event)}\n\n"


def _complete_direct_anthropic_stream_cache(
    request_id: str,
    request_body: Dict,
    response_body,
    status_code: int,
    request_size: int,
    start_time: float,
    original_model: str,
    translated_model: str,
    user_id: str,
) -> None:
    duration = round(time.time() - start_time, 2)
    cache.complete_request(request_id, {
        "request_body": request_body,
        "response_body": response_body,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "status_code": status_code,
        "request_size": request_size,
        "response_size": len(json.dumps(response_body)),
        "input_tokens": 0,
        "output_tokens": 0,
        "duration": duration,
        "user_id": user_id,
    })


def _wait_anthropic_response_with_ping(pending_response, emit_initial_ping: bool = False):
    if emit_initial_ping:
        counters.incr("ping_sent")
        yield _anthropic_ping_event()

    result_iter = wait_result_with_keepalive(
        pending_response,
        state.sse_keepalive_interval,
    )
    while True:
        try:
            item = next(result_iter)
        except StopIteration as stop:
            return stop.value
        if item is KEEPALIVE:
            counters.incr("ping_sent")
            yield _anthropic_ping_event()


def _stream_pending_direct_anthropic_request(
    pending_response,
    headers: Dict,
    anthropic_payload: Dict,
    current_payload: Dict,
    filtered_payload: Dict,
    filtered_request_size: int,
    request_id: str,
    start_time: float,
    original_model: str,
    translated_model: str,
    original_request_body: Dict = None,
    request_headers: Dict = None,
    client_ip: str = None,
    user_id: str = "anonymous",
    cleanup_log_entry=None,
    attempt: int = 0,
    conn_attempt: int = 0,
) -> Response:
    """Continue a direct Anthropic stream after the first upstream call idles.

    The route has already waited one keepalive interval for response headers.
    Returning this ``Response`` commits to SSE semantics and keeps the client
    alive while the upstream response object is still pending.
    """
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": filtered_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "request_size": filtered_request_size,
        "user_id": user_id,
    })

    def generate() -> Generator[str, None, None]:
        nonlocal cleanup_log_entry
        active_payload = current_payload
        active_filtered_payload = filtered_payload
        active_request_size = filtered_request_size
        active_pending = pending_response
        last_response = None
        last_response_body = None

        try:
            cache.update_request_state(request_id, cache.STATE_SENDING)

            max_retries = 3
            for active_attempt in range(attempt, max_retries + 1):
                if active_attempt != attempt:
                    active_filtered_payload = filter_payload_for_copilot(active_payload)
                    active_filtered_payload = adjust_max_tokens_for_thinking(active_filtered_payload)
                    active_request_size = len(json.dumps(active_filtered_payload))
                    cache.update_request_state(
                        request_id,
                        cache.STATE_SENDING,
                        request_body=active_filtered_payload,
                        request_size=active_request_size,
                    )
                    active_pending = None

                connection_retries = state.max_connection_retries
                first_conn_attempt = conn_attempt if active_attempt == attempt else 0
                last_connection_error = None

                for active_conn_attempt in range(first_conn_attempt, connection_retries + 1):
                    try:
                        if active_pending is None:
                            active_pending = _start_direct_anthropic_post(
                                headers,
                                active_filtered_payload,
                                stream=True,
                            )
                            response = yield from _wait_anthropic_response_with_ping(active_pending)
                        else:
                            response = yield from _wait_anthropic_response_with_ping(
                                active_pending,
                                emit_initial_ping=True,
                            )
                        active_pending = None

                        if response.ok:
                            handler_cls = _direct_anthropic_stream_handler_cls()
                            handler = handler_cls(
                                response=response,
                                request_id=request_id,
                                request_size=active_request_size,
                                start_time=start_time,
                                original_model=original_model,
                                translated_model=translated_model,
                                request_body_for_cache=active_filtered_payload,
                                original_request_body=original_request_body,
                                request_headers=request_headers,
                                client_ip=client_ip,
                                user_id=user_id,
                            )
                            handler._cache_seeded = True
                            yield from handler._generate()
                            return

                        last_connection_error = None
                        break
                    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                        active_pending = None
                        last_connection_error = e
                        log_connection_retry(request_id, "/v1/messages", active_conn_attempt, connection_retries, e)
                        ensure_copilot_token()
                        if active_conn_attempt < connection_retries:
                            print(f"[Direct Anthropic] Connection error (attempt {active_conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                            time.sleep(min(2 ** active_conn_attempt, 8))
                            continue
                        print(f"[Direct Anthropic] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

                if last_connection_error is not None:
                    last_response_body = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}",
                        },
                    }
                    _complete_direct_anthropic_stream_cache(
                        request_id,
                        active_filtered_payload,
                        last_response_body,
                        504,
                        active_request_size,
                        start_time,
                        original_model,
                        translated_model,
                        user_id,
                    )
                    yield _anthropic_error_event(last_response_body, 504)
                    return

                last_response = response
                response_text = response.text
                log_error_request("/v1/messages", active_payload, response_text, response.status_code, client_ip)
                last_response_body = _parse_response_body(response)

                if (state.enable_web_search_proxy and
                        has_web_search_tool(active_payload) and
                        is_web_search_unsupported_error(response.status_code, response_text)):
                    print(f"[Direct Anthropic] Web search unsupported, applying search proxy fallback for request {request_id}")
                    active_payload = apply_web_search_fallback(active_payload, state.web_search_proxy_endpoint)
                    continue

                if is_orphaned_tool_result_error(response.status_code, response_text):
                    orphaned_ids = extract_orphaned_tool_use_ids(response_text)
                    if orphaned_ids:
                        print(f"[Direct Anthropic] Attempt {active_attempt + 1}: Found orphaned tool_result IDs: {orphaned_ids}")

                        if cleanup_log_entry is None:
                            cleanup_log_entry = {
                                "request_id": request_id,
                                "original_request": anthropic_payload,
                                "error_response": response_text,
                                "error_status_code": response.status_code,
                                "orphaned_ids": orphaned_ids,
                            }
                        else:
                            cleanup_log_entry["orphaned_ids"].extend(orphaned_ids)

                        cleaned_messages = remove_orphaned_tool_results(
                            active_payload.get("messages", []), orphaned_ids
                        )
                        active_payload = dict(active_payload)
                        active_payload["messages"] = cleaned_messages
                        print(f"[Direct Anthropic] Retrying with cleaned messages...")
                        continue

                _complete_direct_anthropic_stream_cache(
                    request_id,
                    active_filtered_payload,
                    last_response_body,
                    response.status_code,
                    active_request_size,
                    start_time,
                    original_model,
                    translated_model,
                    user_id,
                )
                yield _anthropic_error_event(last_response_body, response.status_code)
                return

            if cleanup_log_entry is not None and last_response is not None:
                cleanup_log_entry["modified_request"] = active_payload
                cleanup_log_entry["final_status_code"] = last_response.status_code
                cleanup_log_entry["final_response"] = last_response.text
                log_tool_result_cleanup(cleanup_log_entry)

            if last_response is not None:
                _complete_direct_anthropic_stream_cache(
                    request_id,
                    active_filtered_payload,
                    last_response_body,
                    last_response.status_code,
                    active_request_size,
                    start_time,
                    original_model,
                    translated_model,
                    user_id,
                )
                yield _anthropic_error_event(last_response_body, last_response.status_code)
                return

        except GeneratorExit:
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            print(f"[Direct Anthropic] Error while waiting for upstream stream for request {request_id}: {type(e).__name__}: {e}")
            response_body = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }
            _complete_direct_anthropic_stream_cache(
                request_id,
                active_filtered_payload,
                response_body,
                500,
                active_request_size,
                start_time,
                original_model,
                translated_model,
                user_id,
            )
            try:
                yield _anthropic_error_event(response_body, 500)
            except GeneratorExit:
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def apply_system_prompt_filters_to_payload(payload: Dict) -> Dict:
    """Apply system prompt filters to Anthropic payload.

    Handles both string and list formats of system prompt.
    - Removes strings specified in state.system_prompt_remove
    - Adds strings specified in state.system_prompt_add (if not already present)
    Returns a new payload with filtered system prompt.
    """
    system = payload.get("system")

    # If no system prompt exists but we have content to add, create one
    if not system:
        if state.system_prompt_add:
            # Create new system prompt with all add strings
            new_system = []
            for add_str in state.system_prompt_add:
                print(f"[Content Filter] Added new system prompt block: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
                new_system.append({"type": "text", "text": add_str})
                counters.incr("mod.system_prompt_add")
            if new_system:
                return {**payload, "system": new_system}
        return payload

    if isinstance(system, str):
        filtered_system = apply_system_prompt_filters(system)
        if filtered_system != system or state.system_prompt_add:
            for add_str in state.system_prompt_add:
                if add_str not in filtered_system:
                    filtered_system = filtered_system + "\n\n" + add_str
                    counters.incr("mod.system_prompt_add")
                    print(f"[Content Filter] Added to system prompt: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
            return {**payload, "system": filtered_system}
    elif isinstance(system, list):
        # Handle list format: filter each text block for removals
        new_system = []
        modified = False

        # Collect all text content for checking if add strings exist
        all_text_content = ""
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                all_text_content += block.get("text", "") + "\n"

        # Apply removal filters to each text block
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                original_text = block.get("text", "")
                if original_text.startswith("x-anthropic-billing-header:"):
                    print(f"[Content Filter] Removed billing header block from system prompt: {original_text[:50]}{'...' if len(original_text) > 50 else ''}")
                    modified = True
                    counters.incr("mod.billing_header_removal")
                    continue
                # Only apply removal filters, not additions (we'll handle additions separately)
                filtered_text = original_text
                for remove_str in state.system_prompt_remove:
                    if remove_str in filtered_text:
                        filtered_text = filtered_text.replace(remove_str, "")
                        counters.incr("mod.system_prompt_remove")
                        print(f"[Content Filter] Removed from system prompt: {remove_str[:50]}{'...' if len(remove_str) > 50 else ''}")

                if filtered_text != original_text:
                    modified = True
                    new_system.append({**block, "text": filtered_text})
                else:
                    new_system.append(block)
            else:
                print(f"[Content Filter] Non-text system block passed through without modification. type={block.get('type') if isinstance(block, dict) else type(block)}")
                new_system.append(block)

        # Check if each add string exists in the combined text content
        # If not, add a new text block for it
        for add_str in state.system_prompt_add:
            if add_str not in all_text_content:
                print(f"[Content Filter] Added new system prompt block: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
                new_system.append({"type": "text", "text": add_str})
                counters.incr("mod.system_prompt_add")
                modified = True

        if modified:
            return {**payload, "system": new_system}

    return payload


def apply_tool_result_suffix_filter_to_payload(payload: Dict) -> Dict:
    """Apply tool result suffix filters to Anthropic payload.

    Removes trailing suffixes from tool_result content in messages.
    Returns a new payload with filtered tool results.
    """
    messages = payload.get("messages", [])
    if not messages:
        return payload

    new_messages = []
    modified = False

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content = []
        msg_modified = False

        for block in content:
            if block.get("type") == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, str):
                    filtered_content = apply_tool_result_suffix_filter(tool_content)
                    if filtered_content != tool_content:
                        msg_modified = True
                        new_content.append({**block, "content": filtered_content})
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            else:
                new_content.append(block)

        if msg_modified:
            modified = True
            new_messages.append({**msg, "content": new_content})
        else:
            new_messages.append(msg)

    if modified:
        return {**payload, "messages": new_messages}

    return payload


@anthropic_bp.route("/v1/messages/count_tokens", methods=["POST"])
def anthropic_count_tokens():
    """Handle Anthropic token counting API.

    This endpoint counts tokens in the request payload for context window management.
    """
    try:
        ensure_copilot_token()
        payload = request.get_json()

        model_id = payload.get("model", "")

        # Translate model name (e.g., "claude-opus-4-6" -> "claude-opus-4.4")
        translated_model = translate_model_name(model_id)
        if translated_model != model_id:
            print(f"[count_tokens] Model name translated: {model_id} -> {translated_model}")
            model_id = translated_model

        # Find the model in cached models
        selected_model = None
        if state.models and state.models.get("data"):
            selected_model = next(
                (m for m in state.models["data"] if m.get("id") == model_id),
                None
            )

        if not selected_model:
            print(f"[count_tokens] Model {model_id} not found, returning default token count")
            return jsonify({"input_tokens": 1})

        # Count tokens from system prompt
        total_tokens = 0
        system = payload.get("system")
        if system:
            if isinstance(system, str):
                total_tokens += count_tokens(system, model_id)
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_tokens += count_tokens(block.get("text", ""), model_id)

        # Count tokens from messages
        for msg in payload.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += count_tokens(content, model_id)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        total_tokens += count_tokens(block.get("text", ""), model_id)
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            total_tokens += count_tokens(tool_content, model_id)
                    elif block.get("type") == "tool_use":
                        # Count tool input as JSON string
                        tool_input = block.get("input", {})
                        total_tokens += count_tokens(json.dumps(tool_input), model_id)

        # Count tokens from tools definitions
        tools = payload.get("tools", [])
        if tools:
            # Add base overhead for tool use capability (similar to copilot-api-js)
            if model_id.startswith("claude"):
                total_tokens += 346
            elif model_id.startswith("grok"):
                total_tokens += 480

            # Count tool definition tokens
            for tool in tools:
                total_tokens += count_tokens(tool.get("name", ""), model_id)
                total_tokens += count_tokens(tool.get("description", ""), model_id)
                input_schema = tool.get("input_schema", {})
                total_tokens += count_tokens(json.dumps(input_schema), model_id)

        # Apply buffer for non-Anthropic vendors (similar to copilot-api-js)
        vendor = selected_model.get("vendor", "")
        if vendor != "Anthropic":
            if model_id.startswith("grok"):
                total_tokens = int(total_tokens * 1.03)
            else:
                total_tokens = int(total_tokens * 1.05)
        return jsonify({"input_tokens": total_tokens})

    except Exception as e:
        print(f"[count_tokens] Error: {e}")
        return jsonify({"input_tokens": 1})


@anthropic_bp.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    """Handle Anthropic messages API"""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Capture the exact wire body before Flask normalises JSON. Duplicate keys,
    # non-finite numbers, invalid UTF-8 and trailing data are ambiguous and are
    # rejected rather than silently changed by a permissive parser.
    original_request_raw = request.get_data(cache=True)
    try:
        anthropic_payload = parse_strict_json_bytes(original_request_raw)
    except StrictJSONError as exc:
        return _anthropic_json_error(str(exc), 400)
    if not isinstance(anthropic_payload, dict):
        return _anthropic_json_error("Anthropic request body must be a JSON object", 400)

    ensure_copilot_token()

    # Capture incoming request headers (auth values redacted before caching).
    request_headers = redact_auth_headers(dict(request.headers))
    client_ip = get_client_ip(request)
    user_id = _current_user_id()

    # Store original request before any modifications
    original_request_body = copy.deepcopy(anthropic_payload)

    original_model = anthropic_payload.get("model", "unknown")

    # Translate model name (applies to both paths)
    translated_model = translate_model_name(original_model)
    if translated_model != original_model:
        print(f"[Anthropic API] Model name translated: {original_model} -> {translated_model}")
        anthropic_payload = {**anthropic_payload, "model": translated_model}

    # Capability routing deliberately happens before any legacy content or
    # thinking rewrite. The Responses converter owns its loss accounting and
    # must see the original request semantics.
    use_direct_api = supports_direct_anthropic_api(translated_model)

    if use_direct_api:
        anthropic_payload = apply_system_prompt_filters_to_payload(anthropic_payload)
        anthropic_payload = apply_tool_result_suffix_filter_to_payload(anthropic_payload)
        anthropic_payload = translate_thinking_enabled_to_adaptive(anthropic_payload, translated_model)
        anthropic_payload = apply_effort_policy(anthropic_payload, translated_model)
        print(f"[Anthropic API] Using direct Anthropic API path for model: {translated_model}")
        return handle_direct_anthropic_request(anthropic_payload, request_id, start_time, original_model, translated_model, original_request_body, request_headers, client_ip=client_ip, user_id=user_id)

    if (
        bool(getattr(state, "anthropic_responses_compat_enabled", False))
        and supports_responses_api(translated_model)
    ):
        print(f"[Anthropic API] Using Responses compatibility path for model: {translated_model}")
        return handle_responses_anthropic_request(
            anthropic_payload=anthropic_payload,
            request_id=request_id,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            request_headers=request_headers,
            client_ip=client_ip,
            user_id=user_id,
        )

    anthropic_payload = apply_system_prompt_filters_to_payload(anthropic_payload)
    anthropic_payload = apply_tool_result_suffix_filter_to_payload(anthropic_payload)
    anthropic_payload = translate_thinking_enabled_to_adaptive(anthropic_payload, translated_model)
    anthropic_payload = apply_effort_policy(anthropic_payload, translated_model)
    print(f"[Anthropic API] Using OpenAI translation path for model: {translated_model}")
    return handle_translated_request(anthropic_payload, request_id, start_time, original_model, translated_model, original_request_body, request_headers, client_ip=client_ip, user_id=user_id)


def _strict_upstream_json(response: requests.Response) -> Dict[str, Any]:
    content = getattr(response, "content", None)
    if isinstance(content, bytes) and content:
        value = parse_strict_json_bytes(content)
    else:
        value = response.json()
    if not isinstance(value, dict):
        raise StrictJSONError("Upstream Responses body must be a JSON object")
    return value


def _responses_request_cache_record(
    *,
    request_headers: Dict[str, Any],
    client_ip: Optional[str],
    original_request_body: Dict[str, Any],
    original_request_raw: bytes,
    responses_payload: Dict[str, Any],
    response_body: Any,
    upstream_response_body: Any,
    status_code: int,
    request_size: int,
    response_size: int,
    start_time: float,
    original_model: str,
    translated_model: str,
    user_id: str,
    warnings: List[Dict[str, Any]],
    request_report: Any,
    response_report: Any = None,
    compatibility_audit: Any = None,
) -> Dict[str, Any]:
    usage = response_body.get("usage", {}) if isinstance(response_body, dict) else {}
    report: Dict[str, Any] = {}
    if request_report is not None:
        report["request"] = (
            request_report.to_dict()
            if hasattr(request_report, "to_dict")
            else copy.deepcopy(request_report)
        )
    if response_report is not None:
        report["response"] = response_report.to_dict()
    cached_upstream_body = redact_responses_response_for_cache(upstream_response_body)
    cache_limit = int(getattr(cache, "max_request_size", 0) or 0)
    if cache_limit > 0 and response_size > cache_limit:
        try:
            upstream_digest_source = json.dumps(
                upstream_response_body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except Exception:
            upstream_digest_source = b""
        cached_upstream_body = {
            "_truncated": True,
            "_size": response_size,
            "_reason": "upstream body exceeded dashboard cache limit",
            "_sha256": hashlib.sha256(upstream_digest_source).hexdigest(),
        }
    record = {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": redact_responses_value_for_cache(responses_payload),
        "response_body": response_body,
        "upstream_response_body": cached_upstream_body,
        "original_request_raw_sha256": hashlib.sha256(original_request_raw).hexdigest(),
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "status_code": status_code,
        "request_size": request_size,
        "response_size": response_size,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "duration": round(time.time() - start_time, 2),
        "user_id": user_id,
        "compatibility_profile": anthropic_responses_wire_profile(translated_model),
        "compatibility_warnings": warnings,
        "conversion_report": report,
    }
    if compatibility_audit is not None:
        record["compatibility_audit"] = (
            compatibility_audit.to_dict()
            if hasattr(compatibility_audit, "to_dict")
            else copy.deepcopy(compatibility_audit)
        )
    return record


def _responses_transport_error(
    *,
    message: str,
    status_code: int,
    warnings: List[Dict[str, Any]],
) -> Response:
    error_type = "api_error"
    if status_code == 504:
        error_type = "timeout_error"
    return _anthropic_json_error(message, status_code, error_type, warnings)


def _cache_responses_local_failure(
    *,
    request_id: str,
    message: str,
    status_code: int,
    request_headers: Dict[str, Any],
    client_ip: Optional[str],
    original_request_body: Dict[str, Any],
    original_request_raw: bytes,
    responses_payload: Optional[Dict[str, Any]],
    start_time: float,
    original_model: str,
    translated_model: str,
    user_id: str,
    warnings: List[Dict[str, Any]],
    request_report: Any = None,
    response_report: Any = None,
    compatibility_audit: Any = None,
    upstream_response_body: Any = None,
    response_size: int = 0,
) -> None:
    error_type = "invalid_request_error" if status_code == 400 else "api_error"
    body = {
        "type": "error",
        "error": {"type": error_type, "message": str(message)},
    }
    cache.add_request(request_id, _responses_request_cache_record(
        request_headers=request_headers,
        client_ip=client_ip,
        original_request_body=original_request_body,
        original_request_raw=original_request_raw,
        responses_payload=responses_payload or {},
        response_body=body,
        upstream_response_body=upstream_response_body,
        status_code=status_code,
        request_size=len(original_request_raw),
        response_size=response_size,
        start_time=start_time,
        original_model=original_model,
        translated_model=translated_model,
        user_id=user_id,
        warnings=warnings,
        request_report=request_report,
        response_report=response_report,
        compatibility_audit=compatibility_audit,
    ))


def _make_anthropic_responses_stream_handler(
    *,
    response: requests.Response,
    request_id: str,
    request_size: int,
    start_time: float,
    original_model: str,
    translated_model: str,
    responses_payload: Dict[str, Any],
    original_request_body: Dict[str, Any],
    original_request_raw: bytes,
    request_headers: Dict[str, Any],
    client_ip: Optional[str],
    user_id: str,
    conversion: Any,
    warnings: List[Dict[str, Any]],
    request_audit: Any,
    replay_context: _ReplayContext,
    mode: str,
) -> AnthropicResponsesStreamHandler:
    holder: Dict[str, Any] = {}

    def on_completed(result: ResponsesToAnthropicResult) -> Optional[str]:
        handler = holder.get("handler")
        if handler is not None and handler.raw_capture_truncated:
            replay_context.warn(
                "audit.capture_limit_exceeded", "/response", "not_fully_stored"
            )
            replay_context.fail_if_lossless(
                "The complete Responses stream exceeded the encrypted audit capture limit"
            )
            if replay_context.fatal_message:
                handler._compatibility_warnings = _merge_compatibility_warnings(
                    handler._compatibility_warnings, replay_context.warnings
                )
                return replay_context.fatal_message
        audit_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=_merge_compatibility_warnings(warnings, replay_context.warnings),
            raw_response_events=(handler.raw_events if handler is not None else []),
            raw_response_sse_lines=(
                handler.raw_sse_lines if handler is not None else []
            ),
            response_report=result.report,
        )
        _persist_replay_result(
            replay_context, conversion, result, audit_snapshot=audit_snapshot
        )
        if handler is not None:
            handler._compatibility_warnings = _merge_compatibility_warnings(
                handler._compatibility_warnings, replay_context.warnings
            )
        return replay_context.fatal_message

    def on_audit_finalized(
        raw_events: List[str],
        raw_sse_lines: List[str],
        terminal_result: Optional[ResponsesToAnthropicResult],
    ) -> None:
        if terminal_result is not None:
            return
        handler = holder.get("handler")
        if handler is not None and handler.raw_capture_truncated:
            replay_context.warn(
                "audit.capture_limit_exceeded", "/response", "not_fully_stored"
            )
            replay_context.fail_if_lossless(
                "The complete Responses stream exceeded the encrypted audit capture limit"
            )
        stream_warnings = _merge_compatibility_warnings(
            warnings,
            replay_context.warnings,
            (
                handler.translator.compatibility_warnings
                if handler is not None else []
            ),
        )
        snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=stream_warnings,
            raw_response_events=raw_events,
            raw_response_sse_lines=raw_sse_lines,
        )
        _persist_audit_only_snapshot(replay_context, request_id, snapshot)
        if handler is not None:
            handler._compatibility_warnings = _merge_compatibility_warnings(
                handler._compatibility_warnings, replay_context.warnings
            )

    handler = AnthropicResponsesStreamHandler(
        response=response,
        request_id=request_id,
        request_size=request_size,
        start_time=start_time,
        original_model=original_model,
        translated_model=translated_model,
        request_body_for_cache=redact_responses_value_for_cache(responses_payload),
        original_request_body=original_request_body,
        request_headers=request_headers,
        client_ip=client_ip,
        user_id=user_id,
        conversion=conversion,
        compatibility_warnings=warnings,
        compatibility_audit=request_audit.to_dict(),
        on_completed=on_completed,
        on_audit_finalized=on_audit_finalized,
        mode=mode,
        sidecar_available=replay_context.store is not None,
        max_raw_capture_bytes=max(
            1024,
            min(
                16 * 1024 * 1024,
                replay_context.max_record_bytes // 3
                if replay_context.max_record_bytes > 0 else 16 * 1024 * 1024,
            ),
        ),
    )
    holder["handler"] = handler
    return handler


def _stream_pending_anthropic_responses_request(
    pending_response: BackgroundResult,
    *,
    request_id: str,
    request_size: int,
    start_time: float,
    original_model: str,
    translated_model: str,
    responses_payload: Dict[str, Any],
    original_request_body: Dict[str, Any],
    original_request_raw: bytes,
    request_headers: Dict[str, Any],
    client_ip: Optional[str],
    user_id: str,
    conversion: Any,
    warnings: List[Dict[str, Any]],
    request_audit: Any,
    replay_context: _ReplayContext,
    mode: str,
) -> Response:
    """Commit an Anthropic SSE response while upstream headers are pending."""
    # Seed synchronously before Flask starts iterating the generator. A client
    # can disconnect after the first pre-header ping, and that attempt must
    # still have a durable dashboard lifecycle record.
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": redact_responses_value_for_cache(responses_payload),
        "model": original_model,
        "translated_model": (
            translated_model if translated_model != original_model else None
        ),
        "endpoint": "/v1/messages",
        "request_size": request_size,
        "user_id": user_id,
    })
    cache.update_request_state(request_id, cache.STATE_SENDING)

    def generate() -> Generator[str, None, None]:
        cache_finished = False
        response: Optional[requests.Response] = None
        try:
            response = yield from _wait_anthropic_response_with_ping(
                pending_response, emit_initial_ping=True
            )
            if not response.ok:
                try:
                    upstream_error = _strict_upstream_json(response)
                except Exception:
                    upstream_error = {
                        "message": "Upstream Responses API returned a non-JSON error"
                    }
                error = anthropic_error_from_responses(
                    upstream_error, response.status_code
                )
                yield _anthropic_error_event(error, response.status_code)
                snapshot = _compatibility_audit_snapshot(
                    original_request_raw=original_request_raw,
                    original_request_body=original_request_body,
                    request_headers=request_headers,
                    responses_payload=responses_payload,
                    conversion=conversion,
                    request_audit=request_audit,
                    warnings=warnings,
                    upstream_response_raw=_upstream_response_bytes(response),
                )
                _persist_audit_only_snapshot(
                    replay_context, request_id, snapshot
                )
                cache.add_request(request_id, _responses_request_cache_record(
                    request_headers=request_headers,
                    client_ip=client_ip,
                    original_request_body=original_request_body,
                    original_request_raw=original_request_raw,
                    responses_payload=responses_payload,
                    response_body=error,
                    upstream_response_body=upstream_error,
                    status_code=response.status_code,
                    request_size=request_size,
                    response_size=len(getattr(response, "text", "").encode("utf-8")),
                    start_time=start_time,
                    original_model=original_model,
                    translated_model=translated_model,
                    user_id=user_id,
                    warnings=_merge_compatibility_warnings(
                        warnings, replay_context.warnings
                    ),
                    request_report=conversion.report,
                    compatibility_audit=request_audit,
                ))
                cache_finished = True
                response.close()
                return
            handler = _make_anthropic_responses_stream_handler(
                response=response,
                request_id=request_id,
                request_size=request_size,
                start_time=start_time,
                original_model=original_model,
                translated_model=translated_model,
                responses_payload=responses_payload,
                original_request_body=original_request_body,
                original_request_raw=original_request_raw,
                request_headers=request_headers,
                client_ip=client_ip,
                user_id=user_id,
                conversion=conversion,
                warnings=warnings,
                request_audit=request_audit,
                replay_context=replay_context,
                mode=mode,
            )
            # The cache entry was seeded above; do not overwrite its timestamp
            # and SENDING state when handing ownership to the normal handler.
            handler._cache_seeded = True
            yield from handler._generate()
            cache_finished = True
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            failure_warnings = _merge_compatibility_warnings(
                warnings,
                [_compatibility_warning("responses.connection_failed", "/", "error")],
            )
            snapshot = _compatibility_audit_snapshot(
                original_request_raw=original_request_raw,
                original_request_body=original_request_body,
                request_headers=request_headers,
                responses_payload=responses_payload,
                conversion=conversion,
                request_audit=request_audit,
                warnings=failure_warnings,
                raw_response_events=[],
            )
            _persist_audit_only_snapshot(replay_context, request_id, snapshot)
            event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream Responses connection failed ({type(exc).__name__})",
                },
            }
            cache.add_request(request_id, _responses_request_cache_record(
                request_headers=request_headers,
                client_ip=client_ip,
                original_request_body=original_request_body,
                original_request_raw=original_request_raw,
                responses_payload=responses_payload,
                response_body=event,
                upstream_response_body=None,
                status_code=504,
                request_size=request_size,
                response_size=0,
                start_time=start_time,
                original_model=original_model,
                translated_model=translated_model,
                user_id=user_id,
                warnings=_merge_compatibility_warnings(
                    failure_warnings, replay_context.warnings
                ),
                request_report=conversion.report,
                compatibility_audit=request_audit,
            ))
            cache_finished = True
            yield f"event: error\ndata: {json.dumps(event)}\n\n"
        except GeneratorExit:
            pending_response.abandon()
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            disconnect_warnings = _merge_compatibility_warnings(
                warnings,
                [_compatibility_warning("responses.client_disconnected", "/", "error")],
            )
            snapshot = _compatibility_audit_snapshot(
                original_request_raw=original_request_raw,
                original_request_body=original_request_body,
                request_headers=request_headers,
                responses_payload=responses_payload,
                conversion=conversion,
                request_audit=request_audit,
                warnings=disconnect_warnings,
                raw_response_events=[],
            )
            _persist_audit_only_snapshot(replay_context, request_id, snapshot)
            cache.add_request(request_id, _responses_request_cache_record(
                request_headers=request_headers,
                client_ip=client_ip,
                original_request_body=original_request_body,
                original_request_raw=original_request_raw,
                responses_payload=responses_payload,
                response_body={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Client disconnected before upstream headers arrived",
                    },
                },
                upstream_response_body=None,
                status_code=499,
                request_size=request_size,
                response_size=0,
                start_time=start_time,
                original_model=original_model,
                translated_model=translated_model,
                user_id=user_id,
                warnings=_merge_compatibility_warnings(
                    disconnect_warnings, replay_context.warnings
                ),
                request_report=conversion.report,
                compatibility_audit=request_audit,
            ))
            cache_finished = True
            return
        except Exception:
            failure_warnings = _merge_compatibility_warnings(
                warnings,
                [_compatibility_warning("responses.pre_header_failed", "/", "error")],
            )
            snapshot = _compatibility_audit_snapshot(
                original_request_raw=original_request_raw,
                original_request_body=original_request_body,
                request_headers=request_headers,
                responses_payload=responses_payload,
                conversion=conversion,
                request_audit=request_audit,
                warnings=failure_warnings,
                raw_response_events=[],
            )
            _persist_audit_only_snapshot(replay_context, request_id, snapshot)
            event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Upstream Responses request failed before headers arrived",
                },
            }
            cache.add_request(request_id, _responses_request_cache_record(
                request_headers=request_headers,
                client_ip=client_ip,
                original_request_body=original_request_body,
                original_request_raw=original_request_raw,
                responses_payload=responses_payload,
                response_body=event,
                upstream_response_body=None,
                status_code=502,
                request_size=request_size,
                response_size=0,
                start_time=start_time,
                original_model=original_model,
                translated_model=translated_model,
                user_id=user_id,
                warnings=_merge_compatibility_warnings(
                    failure_warnings, replay_context.warnings
                ),
                request_report=conversion.report,
                compatibility_audit=request_audit,
            ))
            cache_finished = True
            yield f"event: error\ndata: {json.dumps(event)}\n\n"
        finally:
            if not cache_finished and response is None:
                pending_response.abandon()

    result = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    return _set_compatibility_headers(result, warnings)


def handle_responses_anthropic_request(
    anthropic_payload: Dict,
    request_id: str,
    start_time: float,
    original_model: str,
    translated_model: str,
    original_request_body: Dict = None,
    original_request_raw: bytes = b"",
    request_headers: Dict = None,
    client_ip: str = None,
    user_id: str = "anonymous",
) -> Response:
    """Handle Anthropic Messages directly through the Responses API.

    This path never uses the Chat Completions translator and never retries a
    protocol error with fields removed. Only byte-identical connection retries
    are permitted.
    """
    request_headers = request_headers or {}
    original_request_body = original_request_body or copy.deepcopy(anthropic_payload)
    mode = str(getattr(
        state, "anthropic_responses_compat_mode", MODE_COMPATIBILITY
    ))
    wire_profile = anthropic_responses_wire_profile(translated_model)
    request_audit = audit_anthropic_request(
        request_headers,
        original_request_body,
        mode=mode,
        baseline_manifest={"profiles": CLAUDE_CLI_TOOL_CONTRACT_BASELINES},
    )
    warnings: List[Dict[str, Any]] = _merge_compatibility_warnings(
        request_audit.warnings
    )
    replay_context = _create_replay_context(
        payload=anthropic_payload,
        request_headers=request_headers,
        user_id=user_id,
        model=translated_model,
        wire_profile=wire_profile,
        mode=mode,
    )
    warnings = _merge_compatibility_warnings(warnings, replay_context.warnings)
    if request_audit.should_fail:
        if replay_context.fatal_message is None:
            _persist_audit_only_snapshot(
                replay_context,
                request_id,
                {
                    "version": 1,
                    "rejected_before_upstream": True,
                    "request_raw_base64": base64.b64encode(
                        original_request_raw
                    ).decode("ascii"),
                    "parsed_request": copy.deepcopy(original_request_body),
                    "request_headers": copy.deepcopy(request_headers),
                    "request_compatibility_audit": request_audit.to_dict(),
                    "compatibility_warnings": copy.deepcopy(warnings),
                },
            )
            warnings = _merge_compatibility_warnings(
                warnings, replay_context.warnings
            )
        _log_compatibility_warnings(request_id, warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message="Anthropic request shape is not supported by the active compatibility profile",
            status_code=400,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=None,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            compatibility_audit=request_audit,
        )
        return _anthropic_json_error(
            "Anthropic request shape is not supported by the active compatibility profile",
            400,
            warnings=warnings,
        )
    if replay_context.fatal_message:
        _log_compatibility_warnings(request_id, warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message=replay_context.fatal_message,
            status_code=400,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=None,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            compatibility_audit=request_audit,
        )
        return _anthropic_json_error(replay_context.fatal_message, 400, warnings=warnings)

    resolver = _replay_resolver(replay_context)
    try:
        conversion = convert_anthropic_to_responses(
            anthropic_payload,
            wire_profile=wire_profile,
            mode=mode,
            session_id=replay_context.session_id,
            tenant_id=replay_context.tenant_id,
            replay_resolver=resolver,
            sidecar_available=replay_context.store is not None,
        )
    except AnthropicResponsesConversionError as exc:
        conversion_warnings = _merge_compatibility_warnings(
            warnings, exc.report.warnings
        )
        _persist_audit_only_snapshot(
            replay_context,
            request_id,
            {
                "version": 1,
                "rejected_before_upstream": True,
                "request_raw_base64": base64.b64encode(
                    original_request_raw
                ).decode("ascii"),
                "parsed_request": copy.deepcopy(original_request_body),
                "request_headers": copy.deepcopy(request_headers),
                "request_compatibility_audit": request_audit.to_dict(),
                "request_conversion_report": exc.report.to_dict(),
                "compatibility_warnings": copy.deepcopy(conversion_warnings),
            },
        )
        conversion_warnings = _merge_compatibility_warnings(
            conversion_warnings, replay_context.warnings
        )
        _log_compatibility_warnings(request_id, conversion_warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message=str(exc),
            status_code=400,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=None,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=conversion_warnings,
            request_report=exc.report,
            compatibility_audit=request_audit,
        )
        return _anthropic_json_error(str(exc), 400, warnings=conversion_warnings)
    except (TypeError, ValueError) as exc:
        _cache_responses_local_failure(
            request_id=request_id,
            message=str(exc),
            status_code=400,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=None,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            compatibility_audit=request_audit,
        )
        return _anthropic_json_error(str(exc), 400, warnings=warnings)

    _restore_replay_identifier_mappings(replay_context, conversion)
    if resolver is None and any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in anthropic_payload.get("messages", [])
    ):
        replay_context.warn("replay.unavailable_for_assistant_history", "/messages", "approximation")
        replay_context.fail_if_lossless("Assistant history requires reasoning replay in lossless_required mode")
    if conversion.replay_misses:
        replay_context.fail_if_lossless("Required assistant reasoning replay state is missing")
    warnings = _merge_compatibility_warnings(
        warnings, replay_context.warnings, conversion.report.warnings
    )
    if replay_context.fatal_message:
        _log_compatibility_warnings(request_id, warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message=replay_context.fatal_message,
            status_code=400,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=conversion.payload,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=request_audit,
        )
        return _anthropic_json_error(replay_context.fatal_message, 400, warnings=warnings)

    responses_payload = conversion.payload
    request_size = len(json.dumps(
        responses_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8"))
    enable_vision = any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") in ("image", "document")
            for part in message.get("content", [])
        )
        for message in anthropic_payload.get("messages", [])
    )
    headers = get_copilot_headers(enable_vision)
    is_agent_call = any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in anthropic_payload.get("messages", [])
    )
    headers["X-Initiator"] = "agent" if is_agent_call else "user"

    connection_retries = int(getattr(state, "max_connection_retries", 0))
    response = None
    last_connection_error = None
    for conn_attempt in range(connection_retries + 1):
        try:
            if (
                responses_payload.get("stream")
                and state.sse_keepalive_interval > 0
            ):
                pending_response = BackgroundResult(lambda: requests.post(
                    f"{get_copilot_base_url()}/v1/responses",
                    headers=headers,
                    json=responses_payload,
                    stream=True,
                    timeout=state.upstream_read_timeout,
                ))
                try:
                    response = pending_response.get(
                        timeout=state.sse_keepalive_interval
                    )
                except queue.Empty:
                    _log_compatibility_warnings(request_id, warnings)
                    return _stream_pending_anthropic_responses_request(
                        pending_response,
                        request_id=request_id,
                        request_size=request_size,
                        start_time=start_time,
                        original_model=original_model,
                        translated_model=translated_model,
                        responses_payload=responses_payload,
                        original_request_body=original_request_body,
                        original_request_raw=original_request_raw,
                        request_headers=request_headers,
                        client_ip=client_ip,
                        user_id=user_id,
                        conversion=conversion,
                        warnings=warnings,
                        request_audit=request_audit,
                        replay_context=replay_context,
                        mode=mode,
                    )
            else:
                response = requests.post(
                    f"{get_copilot_base_url()}/v1/responses",
                    headers=headers,
                    json=responses_payload,
                    stream=bool(responses_payload.get("stream")),
                    timeout=state.upstream_read_timeout,
                )
            last_connection_error = None
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_connection_error = exc
            log_connection_retry(
                request_id,
                "/v1/messages (responses)",
                conn_attempt,
                connection_retries,
                exc,
            )
            ensure_copilot_token()
            if conn_attempt < connection_retries:
                time.sleep(min(2 ** conn_attempt, 8))

    if last_connection_error is not None or response is None:
        warnings = _merge_compatibility_warnings(
            warnings,
            [_compatibility_warning("responses.connection_failed", "/", "error")],
        )
        connection_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
        )
        _persist_audit_only_snapshot(
            replay_context, request_id, connection_snapshot
        )
        warnings = _merge_compatibility_warnings(
            warnings, replay_context.warnings
        )
        connection_error_body = {
            "type": "error",
            "error": {
                "type": "timeout_error",
                "message": (
                    "Upstream Responses connection failed after "
                    f"{connection_retries + 1} attempt(s)"
                ),
            },
        }
        cache.add_request(request_id, _responses_request_cache_record(
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            response_body=connection_error_body,
            upstream_response_body=None,
            status_code=504,
            request_size=request_size,
            response_size=0,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=request_audit,
        ))
        _log_compatibility_warnings(request_id, warnings)
        return _responses_transport_error(
            message=(
                "Upstream Responses connection failed after "
                f"{connection_retries + 1} attempt(s)"
            ),
            status_code=504,
            warnings=warnings,
        )

    if not response.ok:
        try:
            upstream_error = _strict_upstream_json(response)
        except Exception:
            upstream_error = {"message": "Upstream Responses API returned a non-JSON error"}
        anthropic_error = anthropic_error_from_responses(
            upstream_error, response.status_code
        )
        error_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
            upstream_response_raw=_upstream_response_bytes(response),
        )
        _persist_audit_only_snapshot(replay_context, request_id, error_snapshot)
        response_size = len(getattr(response, "text", "").encode("utf-8"))
        warnings = _merge_compatibility_warnings(warnings, replay_context.warnings)
        cache.add_request(request_id, _responses_request_cache_record(
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            response_body=anthropic_error,
            upstream_response_body=upstream_error,
            status_code=response.status_code,
            request_size=request_size,
            response_size=response_size,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=request_audit,
        ))
        _log_compatibility_warnings(request_id, warnings)
        result = jsonify(anthropic_error)
        result.status_code = response.status_code
        return _set_compatibility_headers(result, warnings)

    if responses_payload.get("stream"):
        handler = _make_anthropic_responses_stream_handler(
            response=response,
            request_id=request_id,
            request_size=request_size,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            responses_payload=responses_payload,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            request_headers=request_headers,
            client_ip=client_ip,
            user_id=user_id,
            conversion=conversion,
            warnings=warnings,
            request_audit=request_audit,
            replay_context=replay_context,
            mode=mode,
        )
        _log_compatibility_warnings(request_id, warnings)
        return _set_compatibility_headers(handler.stream(), warnings)

    try:
        upstream_response = _strict_upstream_json(response)
    except Exception:
        warnings = _merge_compatibility_warnings(
            warnings,
            [_compatibility_warning("responses.malformed_json_body", "/response", "error")],
        )
        malformed_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
            upstream_response_raw=_upstream_response_bytes(response),
        )
        _persist_audit_only_snapshot(
            replay_context, request_id, malformed_snapshot
        )
        warnings = _merge_compatibility_warnings(
            warnings, replay_context.warnings
        )
        error_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Upstream Responses API returned malformed JSON",
            },
        }
        cache.add_request(request_id, _responses_request_cache_record(
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            response_body=error_body,
            upstream_response_body={"raw_sha256": hashlib.sha256(
                _upstream_response_bytes(response)
            ).hexdigest()},
            status_code=502,
            request_size=request_size,
            response_size=len(_upstream_response_bytes(response)),
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=request_audit,
        ))
        _log_compatibility_warnings(request_id, warnings)
        return _responses_transport_error(
            message="Upstream Responses API returned malformed JSON",
            status_code=502,
            warnings=warnings,
        )

    response_status = str(upstream_response.get("status") or "")
    response_event_type = {
        "completed": "response.completed",
        "incomplete": "response.incomplete",
        "failed": "response.failed",
    }.get(response_status, "response.unknown_status")
    response_audit = audit_responses_event(
        {
            "type": response_event_type,
            # Non-stream bodies have no SSE sequence number. Supplying a local
            # sentinel lets the shared event-shape auditor validate the actual
            # terminal response without reporting a false missing-field drift.
            "sequence_number": 0,
            "response": upstream_response,
        },
        mode=mode,
    )
    warnings = _merge_compatibility_warnings(warnings, response_audit.warnings)
    combined_audit = {
        "request": request_audit.to_dict(),
        "response": response_audit.to_dict(),
    }
    if response_audit.should_fail:
        anthropic_error = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Unsupported Responses response shape",
            },
        }
        drift_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
            upstream_response_raw=_upstream_response_bytes(response),
        )
        drift_snapshot["response_compatibility_audit"] = response_audit.to_dict()
        _persist_audit_only_snapshot(replay_context, request_id, drift_snapshot)
        warnings = _merge_compatibility_warnings(
            warnings, replay_context.warnings
        )
        cache.add_request(request_id, _responses_request_cache_record(
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            response_body=anthropic_error,
            upstream_response_body=upstream_response,
            status_code=502,
            request_size=request_size,
            response_size=len(getattr(response, "text", "").encode("utf-8")),
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=combined_audit,
        ))
        _log_compatibility_warnings(request_id, warnings)
        result = jsonify(anthropic_error)
        result.status_code = 502
        return _set_compatibility_headers(result, warnings)

    if upstream_response.get("status") == "failed" or upstream_response.get("error"):
        anthropic_error = anthropic_error_from_responses(
            upstream_response.get("error") or upstream_response, 500
        )
        failed_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
            upstream_response_raw=_upstream_response_bytes(response),
        )
        failed_snapshot["response_compatibility_audit"] = response_audit.to_dict()
        _persist_audit_only_snapshot(replay_context, request_id, failed_snapshot)
        warnings = _merge_compatibility_warnings(
            warnings, replay_context.warnings
        )
        cache.add_request(request_id, _responses_request_cache_record(
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            response_body=anthropic_error,
            upstream_response_body=upstream_response,
            status_code=500,
            request_size=request_size,
            response_size=len(getattr(response, "text", "").encode("utf-8")),
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            compatibility_audit=combined_audit,
        ))
        result = jsonify(anthropic_error)
        result.status_code = 500
        return _set_compatibility_headers(result, warnings)

    try:
        translated = convert_responses_to_anthropic(
            upstream_response,
            original_model=original_model,
            name_codec=conversion.name_codec,
            call_id_codec=conversion.call_id_codec,
            stop_sequences=conversion.stop_sequences,
            mode=mode,
            sidecar_available=replay_context.store is not None,
        )
    except AnthropicResponsesConversionError as exc:
        warnings = _merge_compatibility_warnings(warnings, exc.report.warnings)
        conversion_snapshot = _compatibility_audit_snapshot(
            original_request_raw=original_request_raw,
            original_request_body=original_request_body,
            request_headers=request_headers,
            responses_payload=responses_payload,
            conversion=conversion,
            request_audit=request_audit,
            warnings=warnings,
            upstream_response_raw=_upstream_response_bytes(response),
            response_report=exc.report,
        )
        conversion_snapshot["response_compatibility_audit"] = response_audit.to_dict()
        _persist_audit_only_snapshot(
            replay_context, request_id, conversion_snapshot
        )
        warnings = _merge_compatibility_warnings(
            warnings, replay_context.warnings
        )
        _log_compatibility_warnings(request_id, warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message=str(exc),
            status_code=502,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            response_report=exc.report,
            compatibility_audit=combined_audit,
            upstream_response_body=upstream_response,
            response_size=len(getattr(response, "text", "").encode("utf-8")),
        )
        return _responses_transport_error(
            message=str(exc), status_code=502, warnings=warnings
        )

    audit_snapshot = _compatibility_audit_snapshot(
        original_request_raw=original_request_raw,
        original_request_body=original_request_body,
        request_headers=request_headers,
        responses_payload=responses_payload,
        conversion=conversion,
        request_audit=request_audit,
        warnings=warnings,
        upstream_response_raw=_upstream_response_bytes(response),
        response_report=translated.report,
    )
    _persist_replay_result(
        replay_context, conversion, translated, audit_snapshot=audit_snapshot
    )
    warnings = _merge_compatibility_warnings(
        warnings,
        replay_context.warnings,
        translated.report.warnings,
    )
    if replay_context.fatal_message:
        _log_compatibility_warnings(request_id, warnings)
        _cache_responses_local_failure(
            request_id=request_id,
            message=replay_context.fatal_message,
            status_code=500,
            request_headers=request_headers,
            client_ip=client_ip,
            original_request_body=original_request_body,
            original_request_raw=original_request_raw,
            responses_payload=responses_payload,
            start_time=start_time,
            original_model=original_model,
            translated_model=translated_model,
            user_id=user_id,
            warnings=warnings,
            request_report=conversion.report,
            response_report=translated.report,
            compatibility_audit=combined_audit,
            upstream_response_body=upstream_response,
            response_size=len(getattr(response, "text", "").encode("utf-8")),
        )
        return _responses_transport_error(
            message=replay_context.fatal_message,
            status_code=500,
            warnings=warnings,
        )

    anthropic_response = translated.response
    cache.add_request(request_id, _responses_request_cache_record(
        request_headers=request_headers,
        client_ip=client_ip,
        original_request_body=original_request_body,
        original_request_raw=original_request_raw,
        responses_payload=responses_payload,
        response_body=anthropic_response,
        upstream_response_body=upstream_response,
        status_code=response.status_code,
        request_size=request_size,
        response_size=len(getattr(response, "text", "").encode("utf-8")),
        start_time=start_time,
        original_model=original_model,
        translated_model=translated_model,
        user_id=user_id,
        warnings=warnings,
        request_report=conversion.report,
        response_report=translated.report,
        compatibility_audit=combined_audit,
    ))
    _log_compatibility_warnings(request_id, warnings)
    return _set_compatibility_headers(jsonify(anthropic_response), warnings)


def handle_direct_anthropic_request(anthropic_payload: Dict, request_id: str, start_time: float,
                                     original_model: str, translated_model: str, original_request_body: Dict = None,
                                     request_headers: Dict = None,
                                     client_ip: str = None,
                                     user_id: str = "anonymous") -> Response:
    """Handle request using direct Anthropic API (no translation needed)."""

    # Check for vision content
    enable_vision = any(
        isinstance(msg.get("content"), list) and
        any(p.get("type") == "image" for p in msg.get("content", []))
        for msg in anthropic_payload.get("messages", [])
    )

    # Agent/user check for X-Initiator header
    is_agent_call = any(
        msg.get("role") == "assistant"
        for msg in anthropic_payload.get("messages", [])
    )

    headers = get_anthropic_headers(enable_vision)
    headers["X-Initiator"] = "agent" if is_agent_call else "user"

    max_retries = 3
    current_payload = anthropic_payload
    cleanup_log_entry = None

    for attempt in range(max_retries + 1):
        # Filter and adjust payload for Copilot
        filtered_payload = filter_payload_for_copilot(current_payload)
        filtered_payload = adjust_max_tokens_for_thinking(filtered_payload)
        filtered_request_size = len(json.dumps(filtered_payload))

        # Non-streaming request
        connection_retries = state.max_connection_retries
        last_connection_error = None
        use_streaming = filtered_payload.get("stream")
        for conn_attempt in range(connection_retries + 1):
            try:
                if use_streaming and state.sse_keepalive_interval > 0:
                    pending_response = _start_direct_anthropic_post(
                        headers,
                        filtered_payload,
                        stream=True,
                    )
                    try:
                        response = pending_response.get(timeout=state.sse_keepalive_interval)
                    except queue.Empty:
                        return _stream_pending_direct_anthropic_request(
                            pending_response=pending_response,
                            headers=headers,
                            anthropic_payload=anthropic_payload,
                            current_payload=current_payload,
                            filtered_payload=filtered_payload,
                            filtered_request_size=filtered_request_size,
                            request_id=request_id,
                            start_time=start_time,
                            original_model=original_model,
                            translated_model=translated_model,
                            original_request_body=original_request_body,
                            request_headers=request_headers,
                            client_ip=client_ip,
                            user_id=user_id,
                            cleanup_log_entry=cleanup_log_entry,
                            attempt=attempt,
                            conn_attempt=conn_attempt,
                        )
                else:
                    response = requests.post(
                        f"{get_copilot_base_url()}/v1/messages",
                        headers=headers,
                        json=filtered_payload,
                        timeout=state.upstream_read_timeout,
                        stream=use_streaming
                    )
                if use_streaming and response.ok:
                    handler_cls = _direct_anthropic_stream_handler_cls()
                    return handler_cls(
                        response=response,
                        request_id=request_id,
                        request_size=filtered_request_size,
                        start_time=start_time,
                        original_model=original_model,
                        translated_model=translated_model,
                        request_body_for_cache=filtered_payload,
                        original_request_body=original_request_body,
                        request_headers=request_headers,
                        client_ip=client_ip,
                        user_id=user_id,
                    ).stream()
                last_connection_error = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_connection_error = e
                log_connection_retry(request_id, "/v1/messages", conn_attempt, connection_retries, e)
                ensure_copilot_token()  # Refresh token in case it's a token expiration issue
                if conn_attempt < connection_retries:
                    print(f"[Direct Anthropic] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                    time.sleep(min(2 ** conn_attempt, 8))
                    continue
                else:
                    print(f"[Direct Anthropic] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

        if last_connection_error is not None:
            error_body = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}})
            return Response(error_body, status=504, mimetype="application/json")

        duration = round(time.time() - start_time, 2)

        if response.ok:
            anthropic_response = response.json()

            # Cache the request/response
            usage = anthropic_response.get("usage", {})
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
                "request_body": filtered_payload,
                "response_body": anthropic_response,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": filtered_request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "duration": duration,
                "user_id": user_id,
            })

            if cleanup_log_entry is not None:
                cleanup_log_entry["modified_request"] = current_payload
                cleanup_log_entry["final_status_code"] = response.status_code
                cleanup_log_entry["final_response"] = anthropic_response
                log_tool_result_cleanup(cleanup_log_entry)

            return jsonify(anthropic_response)
        else:
            log_error_request("/v1/messages", current_payload, response.text, response.status_code, client_ip)
            usage = {}
            try:
                anthropic_response = response.json()
            except:
                anthropic_response = response.text
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
                "request_body": filtered_payload,
                "response_body": anthropic_response,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": filtered_request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration": duration,
                "user_id": user_id,
            })

            # Handle web search unsupported error with proxy fallback
            if (state.enable_web_search_proxy and
                    has_web_search_tool(current_payload) and
                    is_web_search_unsupported_error(response.status_code, response.text)):
                print(f"[Direct Anthropic] Web search unsupported, applying search proxy fallback for request {request_id}")
                current_payload = apply_web_search_fallback(current_payload, state.web_search_proxy_endpoint)
                continue

            # Handle orphaned tool_result error with retry
            if is_orphaned_tool_result_error(response.status_code, response.text):
                orphaned_ids = extract_orphaned_tool_use_ids(response.text)
                if orphaned_ids:
                    print(f"[Direct Anthropic] Attempt {attempt + 1}: Found orphaned tool_result IDs: {orphaned_ids}")

                    if cleanup_log_entry is None:
                        cleanup_log_entry = {
                            "request_id": request_id,
                            "original_request": anthropic_payload,
                            "error_response": response.text,
                            "error_status_code": response.status_code,
                            "orphaned_ids": orphaned_ids,
                        }
                    else:
                        cleanup_log_entry["orphaned_ids"].extend(orphaned_ids)

                    cleaned_messages = remove_orphaned_tool_results(
                        current_payload.get("messages", []), orphaned_ids
                    )
                    current_payload = dict(current_payload)
                    current_payload["messages"] = cleaned_messages
                    print(f"[Direct Anthropic] Retrying with cleaned messages...")
                    continue

    # Final failure after all retries
    if cleanup_log_entry is not None:
        cleanup_log_entry["modified_request"] = current_payload
        cleanup_log_entry["final_status_code"] = response.status_code
        cleanup_log_entry["final_response"] = response.text
        log_tool_result_cleanup(cleanup_log_entry)

    return Response(response.text, status=response.status_code, mimetype="application/json")


def handle_translated_request(anthropic_payload: Dict, request_id: str, start_time: float,
                               original_model: str, translated_model: str, original_request_body: Dict = None,
                               request_headers: Dict = None,
                               client_ip: str = None,
                               user_id: str = "anonymous") -> Response:
    """Handle request using OpenAI translation path."""
    # Check for vision content
    enable_vision = any(
        isinstance(msg.get("content"), list) and
        any(p.get("type") == "image" for p in msg.get("content", []))
        for msg in anthropic_payload.get("messages", [])
    )

    max_retries = 3
    current_payload = anthropic_payload
    cleanup_log_entry = None

    for attempt in range(max_retries + 1):
        # Translate to OpenAI format; output_config is intentionally dropped here
        # since Copilot's /chat/completions effort support is unverified.
        openai_payload = translate_anthropic_to_openai(current_payload)
        openai_request_size = len(json.dumps(openai_payload))
        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in openai_payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        if anthropic_payload.get("stream"):
            return stream_anthropic_messages(openai_payload, headers, request_id,
                                            current_payload, openai_request_size, start_time,
                                            original_model, translated_model, original_request_body, request_headers,
                                            client_ip=client_ip, user_id=user_id)

        # Non-streaming request
        connection_retries = state.max_connection_retries
        last_connection_error = None
        for conn_attempt in range(connection_retries + 1):
            try:
                response = requests.post(
                    f"{get_copilot_base_url()}/chat/completions",
                    headers=headers,
                    json=openai_payload,
                    timeout=state.upstream_read_timeout,
                )
                last_connection_error = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_connection_error = e
                log_connection_retry(request_id, "/v1/messages (translated)", conn_attempt, connection_retries, e)
                ensure_copilot_token()  # Refresh token in case it's a token expiration issue
                if conn_attempt < connection_retries:
                    print(f"[Translated API] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                    time.sleep(min(2 ** conn_attempt, 8))
                    continue
                else:
                    print(f"[Translated API] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

        if last_connection_error is not None:
            error_body = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}})
            return Response(error_body, status=504, mimetype="application/json")

        duration = round(time.time() - start_time, 2)

        if response.ok:
            openai_response = response.json()
            anthropic_response = translate_openai_to_anthropic(openai_response)

            # Cache the request/response
            usage = openai_response.get("usage", {})
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
                "request_body": openai_payload,
                "response_body": anthropic_response,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": openai_request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": cached_tokens,
                "duration": duration,
                "user_id": user_id,
            })

            return jsonify(anthropic_response)
        else:
            log_error_request("/v1/messages", anthropic_payload, response.text, response.status_code, client_ip)

            # Handle web search unsupported error with proxy fallback
            if (state.enable_web_search_proxy and
                    has_web_search_tool(current_payload) and
                    is_web_search_unsupported_error(response.status_code, response.text)):
                print(f"[Translated API] Web search unsupported, applying search proxy fallback for request {request_id}")
                current_payload = apply_web_search_fallback(current_payload, state.web_search_proxy_endpoint)
                continue

            if is_orphaned_tool_result_error(response.status_code, response.text):
                orphaned_ids = extract_orphaned_tool_use_ids(response.text)
                if orphaned_ids:
                    print(f"[Anthropic API] Attempt {attempt + 1}: Found orphaned tool_result IDs: {orphaned_ids}")

                    if cleanup_log_entry is None:
                        cleanup_log_entry = {
                            "request_id": request_id,
                            "original_request": anthropic_payload,
                            "error_response": response.text,
                            "error_status_code": response.status_code,
                            "orphaned_ids": orphaned_ids,
                        }
                    else:
                        cleanup_log_entry["orphaned_ids"].extend(orphaned_ids)

                    cleaned_messages = remove_orphaned_tool_results(
                        current_payload.get("messages", []), orphaned_ids
                    )
                    current_payload = dict(current_payload)
                    current_payload["messages"] = cleaned_messages
                    print(f"[Anthropic API] Retrying with cleaned messages...")
                    continue

        if cleanup_log_entry is not None:
            cleanup_log_entry["modified_request"] = current_payload
            cleanup_log_entry["final_status_code"] = response.status_code
            cleanup_log_entry["final_response"] = response.text
            log_tool_result_cleanup(cleanup_log_entry)

        return Response(response.text, status=response.status_code, mimetype="application/json")


def stream_anthropic_messages(openai_payload: Dict, headers: Dict, request_id: str,
                              anthropic_payload: Dict, request_size: int, start_time: float,
                              original_model: str, translated_model: str, original_request_body: Dict = None,
                              request_headers: Dict = None,
                              client_ip: str = None,
                              user_id: str = "anonymous") -> Response:
    """Handle streaming Anthropic messages"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": openai_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "request_size": request_size,
        "user_id": user_id,
    })

    def generate() -> Generator[str, None, None]:
        stream_state = AnthropicStreamState()
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        total_cache_read_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False
        final_openai_payload = openai_payload
        final_request_size = request_size

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            pending_response = BackgroundResult(lambda: requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=openai_payload,
                stream=True,
                timeout=state.upstream_read_timeout,
            ))
            response = yield from _wait_anthropic_response_with_ping(pending_response)
            status_code = response.status_code

            # Handle web search unsupported error before streaming begins
            if (not response.ok and
                    state.enable_web_search_proxy and
                    has_web_search_tool(anthropic_payload) and
                    is_web_search_unsupported_error(response.status_code, response.text)):
                print(f"[Stream Anthropic] Web search unsupported, applying search proxy fallback for request {request_id}")
                modified_payload = apply_web_search_fallback(anthropic_payload, state.web_search_proxy_endpoint)
                new_openai_payload = translate_anthropic_to_openai(modified_payload)
                final_openai_payload = new_openai_payload
                final_request_size = len(json.dumps(new_openai_payload))
                cache.update_request_state(
                    request_id,
                    cache.STATE_SENDING,
                    request_body=new_openai_payload,
                    request_size=final_request_size,
                )
                pending_response = BackgroundResult(lambda: requests.post(
                    f"{get_copilot_base_url()}/chat/completions",
                    headers=headers,
                    json=new_openai_payload,
                    stream=True,
                    timeout=state.upstream_read_timeout,
                ))
                response = yield from _wait_anthropic_response_with_ping(pending_response)
                status_code = response.status_code

            for line in iter_lines_with_keepalive(response, state.sse_keepalive_interval):
                if line is KEEPALIVE:
                    counters.incr("ping_sent")
                    yield 'event: ping\ndata: {"type": "ping"}\n\n'
                    continue

                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)
                            total_cache_read_input_tokens = chunk["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)

                        # Translate to Anthropic events
                        events = translate_chunk_to_anthropic_events(chunk, stream_state)
                        for event in events:
                            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    except json.JSONDecodeError:
                        continue

                elif status_code > 399:
                    # Non-JSON error response - yield as is
                    yield f"{line}\n\n"
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream Anthropic] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this is normal, just clean up
            error_occurred = True
            print(f"[Stream Anthropic] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream Anthropic] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the OpenAI response then translate to Anthropic format
        reconstructed_openai = reconstruct_openai_response_from_chunks(response_chunks)
        anthropic_response = translate_openai_to_anthropic(reconstructed_openai) if reconstructed_openai else {}
        if error_occurred and not anthropic_response:
            anthropic_response = {"error": {"type": "api_error", "message": "Stream interrupted"}}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": final_openai_payload,
            "response_body": anthropic_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/messages",
            "status_code": status_code,
            "request_size": final_request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
            "duration": duration,
            "user_id": user_id,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
