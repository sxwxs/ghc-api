"""Independent routes for config-driven OpenAI-compatible upstream proxies."""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple

from flask import Blueprint, Response, g, jsonify, request

from ..auth import ANONYMOUS_USER_ID, redact_auth_headers, require_auth
from ..cache import cache
from ..proxy import ProxyRequestError, ProxyRuntime
from ..proxy.config import ProxyApiConfig, ProxyModelApiConfig, ProxyModelConfig, ProxyProfileConfig
from ..sse import ProxyChatCompletionsStreamHandler, ProxyResponsesStreamHandler
from ..state import state
from ..utils import get_client_ip


proxy_bp = Blueprint("configured_proxy", __name__)
proxy_runtime = ProxyRuntime()


@proxy_bp.before_request
def _proxy_auth_gate():
    """Protect only the independent /proxy namespace without changing legacy auth routing."""
    if not state.enable_auth:
        g.user_id = ANONYMOUS_USER_ID
        return None

    result = require_auth(request)
    if result.user_id is None:
        return jsonify({
            "error": result.error_code,
            "message": result.error_message,
        }), result.http_status

    g.user_id = result.user_id
    return None


def _current_user_id() -> str:
    return getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID


def _error(message: str, code: str, status: int, param: Optional[str] = None):
    error = {
        "message": message,
        "type": "invalid_request_error" if status < 500 else "proxy_error",
        "code": code,
    }
    if param:
        error["param"] = param
    return jsonify({"error": error}), status


def _resolve_target(
    profile_name: str,
    api_name: str,
    model_id: str,
) -> Optional[Tuple[ProxyProfileConfig, ProxyApiConfig, ProxyModelConfig, ProxyModelApiConfig]]:
    profile = proxy_runtime.registry.get_profile(profile_name)
    if profile is None:
        return None
    resolved = profile.resolve(api_name, model_id)
    if resolved is None:
        return None
    api, model, model_api = resolved
    return profile, api, model, model_api


def _rewrite_non_stream_model(result, public_model: str, rewrite: bool):
    if rewrite and isinstance(result, dict) and "model" in result:
        result = dict(result)
        result["model"] = public_model
    return result


def _usage_for_api(api_name: str, result) -> Tuple[int, int, int, int]:
    if not isinstance(result, dict):
        return 0, 0, 0, 0
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    if api_name == "responses":
        details = usage.get("input_tokens_details", {}) or {}
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            details.get("cached_tokens", 0),
            0,
        )
    details = usage.get("prompt_tokens_details", {}) or {}
    return (
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        0,
        details.get("cached_tokens", 0),
    )


def _cache_non_stream(
    request_id: str,
    endpoint: str,
    profile_name: str,
    api_name: str,
    original_model: str,
    translated_model: str,
    original_request_body: Dict,
    upstream_payload: Dict,
    request_headers: Dict,
    client_ip: str,
    user_id: str,
    status_code: int,
    result,
    request_size: int,
    response_size: int,
    duration: float,
) -> None:
    if status_code < 400:
        input_tokens, output_tokens, cache_creation, cache_read = _usage_for_api(api_name, result)
    else:
        input_tokens, output_tokens, cache_creation, cache_read = 0, 0, 0, 0
    cache.add_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": upstream_payload,
        "response_body": result,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": endpoint,
        "status_code": status_code,
        "request_size": request_size,
        "response_size": response_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "duration": duration,
        "user_id": user_id,
        "upstream_provider": "configured_proxy",
        "upstream_profile": profile_name,
        "upstream_api": api_name,
    })


def _handle_proxy_request(profile_name: str, api_name: str):
    start_time = time.time()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", "invalid_json", 400)

    original_model = payload.get("model")
    if not isinstance(original_model, str) or not original_model:
        return _error("The 'model' field is required.", "missing_model", 400, "model")

    target = _resolve_target(profile_name, api_name, original_model)
    if target is None:
        profile = proxy_runtime.registry.get_profile(profile_name)
        if profile is None:
            if proxy_runtime.registry.last_error:
                return _error("The configured proxy is unavailable because its private configuration is invalid.", "proxy_config_error", 503)
            return _error("Configured proxy profile not found.", "proxy_not_found", 404)
        return _error(
            f"Model '{original_model}' does not support this endpoint in the configured proxy profile.",
            "unsupported_model",
            400,
            "model",
        )

    profile, api, model, model_api = target
    original_request_body = copy.deepcopy(payload)
    request_id = str(uuid.uuid4())
    request_headers = redact_auth_headers(dict(request.headers))
    client_ip = get_client_ip(request)
    user_id = _current_user_id()
    use_streaming = bool(payload.get("stream", False))
    endpoint = f"/proxy/{profile_name}/v1/" + (
        "responses" if api_name == "responses" else "chat/completions"
    )

    try:
        upstream = proxy_runtime.post(
            profile=profile,
            api=api,
            model=model,
            model_api=model_api,
            payload=payload,
            stream=use_streaming,
        )
    except ProxyRequestError as exc:
        print(f"[Configured Proxy] Request {request_id} could not start: {exc}")
        return _error("Configured upstream request could not be started.", "upstream_unavailable", 503)

    response = upstream.response
    upstream_payload = upstream.payload
    translated_model = upstream_payload.get("model")
    if not isinstance(translated_model, str) or not translated_model:
        translated_model = original_model
    request_size = len(json.dumps(upstream_payload).encode("utf-8"))

    if use_streaming and response.ok:
        content_length = response.headers.get("Content-Length")
        content_type = response.headers.get("Content-Type", "")
        if content_length == "0":
            response.close()
            return _error("The configured upstream returned an empty streaming response.", "empty_upstream_response", 502)
        if "text/event-stream" not in content_type.lower() and not api.accept_mislabeled_sse:
            response.close()
            return _error("The configured upstream did not return an event stream.", "invalid_upstream_stream", 502)

        common = {
            "response": response,
            "request_id": request_id,
            "request_size": request_size,
            "start_time": start_time,
            "original_model": original_model,
            "translated_model": translated_model,
            "request_body_for_cache": upstream_payload,
            "original_request_body": original_request_body,
            "request_headers": request_headers,
            "client_ip": client_ip,
            "user_id": user_id,
            "endpoint": endpoint,
            "profile_name": profile_name,
            "public_model": original_model,
            "rewrite_model": api.response_model == "public",
        }
        if api_name == "responses":
            return ProxyResponsesStreamHandler(**common).stream()
        return ProxyChatCompletionsStreamHandler(**common).stream()

    duration = round(time.time() - start_time, 2)
    response_content = response.content
    response_size = len(response_content)

    if response.ok and not response_content:
        response.close()
        result = {
            "error": {
                "message": "The configured upstream returned an empty response body.",
                "type": "proxy_error",
                "code": "empty_upstream_response",
            }
        }
        _cache_non_stream(
            request_id, endpoint, profile_name, api_name, original_model,
            translated_model, original_request_body, upstream_payload, request_headers, client_ip,
            user_id, 502, result, request_size, 0, duration,
        )
        return jsonify(result), 502

    try:
        result = response.json()
        result_is_json = True
    except ValueError:
        result = response.text
        result_is_json = False

    if response.ok and result_is_json:
        result = _rewrite_non_stream_model(
            result, original_model, api.response_model == "public"
        )

    _cache_non_stream(
        request_id, endpoint, profile_name, api_name, original_model,
        translated_model, original_request_body, upstream_payload, request_headers, client_ip,
        user_id, response.status_code, result, request_size, response_size, duration,
    )

    if result_is_json:
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=response.status_code,
            content_type=response.headers.get("Content-Type", "application/json"),
        )
    return Response(
        response_content,
        status=response.status_code,
        content_type=response.headers.get("Content-Type", "text/plain"),
    )


@proxy_bp.route("/proxy/<profile_name>/v1/responses", methods=["POST"])
def proxy_responses(profile_name: str):
    return _handle_proxy_request(profile_name, "responses")


@proxy_bp.route("/proxy/<profile_name>/v1/chat/completions", methods=["POST"])
def proxy_chat_completions(profile_name: str):
    return _handle_proxy_request(profile_name, "chat_completions")


def _model_data(
    profile_name: str,
    profile: ProxyProfileConfig,
    model: ProxyModelConfig,
    include_profile: bool = False,
) -> Optional[Dict]:
    supported_endpoints = []
    if model.api_config("responses") is not None and "responses" in profile.apis:
        supported_endpoints.append("/responses")
    if model.api_config("chat_completions") is not None and "chat_completions" in profile.apis:
        supported_endpoints.append("/chat/completions")
    if not supported_endpoints:
        return None

    data = {
        "id": model.id,
        "object": "model",
        "type": "model",
        "created": 0,
        "created_at": datetime.utcfromtimestamp(0).isoformat() + "Z",
        "owned_by": "configured-proxy",
        "display_name": model.display_name,
        "supported_endpoints": supported_endpoints,
        "reasoning": model.reasoning,
        "input": list(model.input_types),
        "context_window": model.context_window,
        "max_output_tokens": model.max_output_tokens,
    }
    if include_profile:
        data["profile"] = profile_name
        data["base_url"] = f"/proxy/{profile_name}/v1"
    return data


@proxy_bp.route("/proxy/models", methods=["GET"])
def proxy_model_catalog():
    """List configured-proxy models across profiles for first-party clients."""
    snapshot = proxy_runtime.registry.snapshot()
    if not snapshot.profiles and proxy_runtime.registry.last_error:
        return _error(
            "Configured proxy models are unavailable because the private configuration is invalid.",
            "proxy_config_error",
            503,
        )

    models = []
    for profile_name, profile in snapshot.profiles.items():
        for model in profile.models.values():
            data = _model_data(profile_name, profile, model, include_profile=True)
            if data is not None:
                models.append(data)
    return jsonify({"object": "list", "data": models, "has_more": False})


@proxy_bp.route("/proxy/<profile_name>/v1/models", methods=["GET"])
def proxy_models(profile_name: str):
    profile = proxy_runtime.registry.get_profile(profile_name)
    if profile is None:
        if proxy_runtime.registry.last_error:
            return _error("The configured proxy is unavailable because its private configuration is invalid.", "proxy_config_error", 503)
        return _error("Configured proxy profile not found.", "proxy_not_found", 404)

    models = []
    for model in profile.models.values():
        data = _model_data(profile_name, profile, model)
        if data is not None:
            models.append(data)

    return jsonify({"object": "list", "data": models, "has_more": False})
