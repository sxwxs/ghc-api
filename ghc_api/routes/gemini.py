"""Experimental native Gemini and Cloud Code inference compatibility."""

import json
import os
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from ..api_helpers import supported_reasoning_efforts
from ..gemini_native import CLOUD_MODELS, IncompleteChatStream, from_chat, from_chat_stream, to_chat
from ..sse.keepalive import KEEPALIVE
from . import openai

gemini_bp = Blueprint("gemini", __name__)
BUFFERED_STREAM_MAX_BYTES = 8 * 1024 * 1024


def _error(message, code=400):
    return jsonify({"error": {"code": code, "message": message,
                             "status": "INVALID_ARGUMENT" if code == 400 else "INTERNAL"}}), code


def _generate(model=None, action=None):
    if action not in ("generateContent", "streamGenerateContent"):
        return _error("Unsupported Gemini method", 404)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("Request must be a JSON object")
    cloud = model is None
    body = data.get("request") if cloud else data
    selected = data.get("model", data.get("modelId")) if cloud else model
    if (os.environ.get("GHC_API_GEMINI_FLASH_LITE_FALLBACK") == "1"
            and selected in ("gemini-3.1-flash-lite-preview", "models/gemini-3.1-flash-lite-preview")
            and isinstance(body, dict)):
        replacement = "gemini-3.8-flash"
        catalog = current_app.make_response(openai.list_models_full())
        if catalog.status_code != 200:
            return catalog
        target = next((entry for entry in catalog.get_json()["data"]
                       if entry.get("id") == replacement), None)
        if (target is None
                or not {"/chat/completions", "/v1/chat/completions"}.intersection(
                    target.get("supported_endpoints", []))
                or "low" not in supported_reasoning_efforts(replacement)):
            current_app.logger.error("Gemini helper replacement %s lacks Chat Completions/LOW support",
                                     replacement)
            return _error("Experimental Gemini helper replacement lacks Chat Completions/LOW support", 503)
        generation = body.get("generationConfig", {})
        thinking = generation.get("thinkingConfig", {}) if isinstance(generation, dict) else None
        level = thinking.get("thinkingLevel") if isinstance(thinking, dict) else None
        mapped = isinstance(level, str) and level.upper() == "MINIMAL"
        if mapped:
            body = {**body, "generationConfig": {
                **generation, "thinkingConfig": {**thinking, "thinkingLevel": "LOW"},
            }}
        current_app.logger.warning(
            "Experimental Gemini helper substitution: %s -> %s; thinkingLevel %r -> %r; "
            "not verified safety-equivalent", selected, replacement, level, "LOW" if mapped else level,
        )
        selected = replacement
    try:
        payload = to_chat(body, selected, action == "streamGenerateContent")
    except ValueError as exc:
        return _error(str(exc))
    include_thoughts = body.get("generationConfig", {}).get("thinkingConfig", {}).get("includeThoughts", True)
    upstream = current_app.make_response(openai.handle_chat_completions(payload))
    if upstream.status_code >= 400:
        return upstream

    def wrap(value):
        return {"response": value} if cloud and "error" not in value else value

    if not payload["stream"]:
        try:
            result = from_chat(upstream.get_json(), include_thoughts)
        except (ValueError, KeyError, TypeError) as exc:
            current_app.logger.error("Invalid upstream Gemini translation: %s", exc)
            return _error("Invalid upstream completion response", 502)
        finally:
            upstream.close()
        return jsonify(wrap(result))

    @stream_with_context
    def generate():
        response = upstream
        attempts = 2 if os.environ.get("GHC_API_GEMINI_SAFE_STREAM_RETRY") == "1" else 1
        buffered = os.environ.get("GHC_API_GEMINI_BUFFER_STREAM") == "1"
        for attempt in range(attempts):
            emitted = 0
            pending = []
            pending_bytes = 0
            retry = False
            try:
                for event in from_chat_stream(response.iter_encoded(), include_thoughts):
                    if event is KEEPALIVE:
                        yield ": keepalive\n\n"
                        continue
                    error = event.get("error")
                    code = error.get("code") if isinstance(error, dict) else None
                    if (not emitted and attempt + 1 < attempts and code in (429, 502, 503, 504)
                            and error.get("retryable") is not False):
                        current_app.logger.warning(
                            "Gemini stream retry before output: model=%s attempt=%d upstream_status=%s",
                            selected, attempt + 1, code,
                        )
                        retry = True
                        break
                    encoded = "data: " + json.dumps(wrap(event)) + "\n\n"
                    if "error" in event:
                        yield encoded
                        return
                    if buffered:
                        pending_bytes += len(encoded.encode("utf-8"))
                        if pending_bytes > BUFFERED_STREAM_MAX_BYTES:
                            raise ValueError("Buffered Gemini response exceeds 8 MiB")
                        pending.append(encoded)
                    else:
                        emitted += 1
                        yield encoded
                if not retry:
                    for encoded in pending:
                        emitted += 1
                        yield encoded
                    return
            except (ValueError, KeyError, TypeError) as exc:
                retry = (isinstance(exc, IncompleteChatStream) and not emitted
                         and attempt + 1 < attempts)
                current_app.logger.log(
                    30 if retry else 40,
                    "Gemini stream translation failed: model=%s attempt=%d emitted_events=%d "
                    "buffered_events=%d error_type=%s retry=%s reason=%s",
                    selected, attempt + 1, emitted, len(pending), type(exc).__name__, retry, exc,
                )
                if not retry:
                    yield "event: error\ndata: " + json.dumps({
                        "error": {"code": 502, "message": "Invalid upstream completion stream"},
                    }) + "\n\n"
                    return
            finally:
                response.close()
            time.sleep(1)
            response = current_app.make_response(openai.handle_chat_completions(payload))
            if response.status_code >= 400:
                try:
                    current_app.logger.error("Gemini stream retry failed: model=%s upstream_status=%d",
                                             selected, response.status_code)
                    yield "event: error\ndata: " + json.dumps({
                        "error": {"code": response.status_code, "message": "Upstream stream retry failed"},
                    }) + "\n\n"
                finally:
                    response.close()
                return
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache"})


@gemini_bp.route("/v1beta/models/<model>:<action>", methods=["POST"])
@gemini_bp.route("/v1/models/<model>:<action>", methods=["POST"])
def generate_content(model, action):
    return _generate(model, action)


@gemini_bp.route("/v1internal:<action>", methods=["POST"])
def cloud_content(action):
    if action == "fetchAvailableModels":
        return models(cloud=True)
    if action not in ("generateContent", "streamGenerateContent"):
        return _error("Google account and onboarding methods are not implemented; use Google's own service", 501)
    return _generate(action=action)


@gemini_bp.route("/v1beta/models", methods=["GET"])
def models(cloud=False):
    catalog = current_app.make_response(openai.list_models())
    if catalog.status_code != 200:
        return catalog
    entries = catalog.get_json()["data"]
    if cloud:
        result = {}
        for placeholder, ident in CLOUD_MODELS.items():
            model = next((entry for entry in entries if entry["id"] == ident), None)
            if model is not None:
                result[ident] = {
                    "displayName": "Copilot " + ident,
                    "model": placeholder,
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "tokenizerType": "LLAMA_WITH_SPECIAL",
                }
                for source, target in (("context_window", "maxTokens"),
                                       ("max_output_tokens", "maxOutputTokens")):
                    if model.get(source) is not None:
                        result[ident][target] = model[source]
        return jsonify({"models": result, "agentModelSorts": [
            {"displayName": "Copilot Gemini", "groups": [{"modelIds": list(result)}]},
        ]})
    return jsonify({"models": [
        {"name": "models/" + entry["id"], "displayName": entry["display_name"],
         "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
        for entry in entries if entry["id"].startswith("gemini-")
    ]})
