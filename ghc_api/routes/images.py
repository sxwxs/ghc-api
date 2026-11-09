"""OpenAI-compatible Images API backed by Copilot's advertised image endpoints.

Image creation is billable and not idempotent. Never replay a request after a
read timeout or connection failure: the upstream may already have generated it.
"""

import copy
import json
import time
import uuid

import requests
from flask import Blueprint, Response, g, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from ..api_helpers import (
    ensure_copilot_token, fetch_models, get_copilot_base_url,
    get_copilot_headers, image_upstream_path,
)
from ..auth import redact_auth_headers
from ..cache import cache
from ..state import state
from ..translator import translate_model_name
from ..utils import get_client_ip, responses_usage_counts

images_bp = Blueprint("images", __name__)
MAX_IMAGE_REQUEST_BYTES = 50 * 1024 * 1024


def _error(message, code, param=None, status=400):
    error = {"message": message, "type": "invalid_request_error" if status < 500 else "upstream_error", "code": code}
    if param:
        error["param"] = param
    return jsonify({"error": error}), status


def _image_log_body(body):
    """Keep usage and parameters, not multi-megabyte image payloads in logs."""
    if isinstance(body, list):
        return [_image_log_body(item) for item in body]
    if isinstance(body, dict):
        return {
            key: ({"_omitted": True, "_chars": len(value)}
                  if key == "b64_json" and isinstance(value, str)
                  else _image_log_body(value))
            for key, value in body.items()
        }
    if isinstance(body, str) and body.startswith("data:image/"):
        return {"_omitted": True, "_chars": len(body)}
    return body


def check_image_upload_limit():
    # With a terminated/chunked WSGI input, read-all may return exactly the
    # LimitedStream cap instead of raising. Never accept that truncated body.
    if not request.content_length and getattr(request.stream, "is_exhausted", False):
        raise RequestEntityTooLarge()


@images_bp.errorhandler(RequestEntityTooLarge)
def _image_request_too_large(exc=None):
    return _error("Image request exceeds the 50 MiB upload limit.", "request_too_large", status=413)


@images_bp.route("/v1/images/generations", methods=["POST"])
@images_bp.route("/images/generations", methods=["POST"])
def generations():
    return _forward_image_request("generations")


@images_bp.route("/v1/images/edits", methods=["POST"])
@images_bp.route("/images/edits", methods=["POST"])
def edits():
    return _forward_image_request("edits")


def _forward_image_request(operation):
    start = time.time()
    endpoint = f"/v1/images/{operation}"
    multipart = request.mimetype == "multipart/form-data"
    try:
        if request.content_length and request.content_length > MAX_IMAGE_REQUEST_BYTES:
            raise RequestEntityTooLarge()
        if multipart:
            if operation != "edits":
                return _error("Image generation requires a JSON object.", "invalid_json")
            payload = request.form.to_dict()
            check_image_upload_limit()
            if any(len(request.form.getlist(k)) != 1 for k in ("model", "prompt") if k in request.form):
                return _error("Specify model and prompt only once.", "invalid_form")
            if not (request.files.getlist("image") or request.files.getlist("image[]")):
                return _error("At least one image file is required.", "missing_image", "image")
        else:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return _error("Request body must be a JSON object or an image-edit multipart form.", "invalid_json")

        original_model = payload.get("model")
        if not isinstance(original_model, str) or not original_model.strip():
            return _error("The 'model' field is required.", "missing_model", "model")
        if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
            return _error("The 'prompt' field must be a non-empty string.", "invalid_prompt", "prompt")
        stream = payload.get("stream", False)
        if stream not in (False, None, "false", "0"):
            return _error("This proxy currently supports non-streaming image responses only.", "unsupported_stream", "stream")

        translated_model = translate_model_name(original_model)
        try:
            if state.models is None and not fetch_models():
                return _error("Failed to load the Copilot model catalog.", "model_catalog_unavailable", status=502)
        except (requests.RequestException, RuntimeError):
            return _error("Failed to load the Copilot model catalog.", "model_catalog_unavailable", status=502)
        upstream_path = image_upstream_path(translated_model, operation)
        if upstream_path is None:
            return _error(f"Model '{original_model}' does not support image {operation}.", "unsupported_model", "model")

        try:
            ensure_copilot_token()
        except (requests.RequestException, RuntimeError):
            return _error("Copilot authentication is currently unavailable.", "upstream_auth_unavailable", status=502)
        headers = get_copilot_headers()
        headers["X-Initiator"] = "user"
        original_body = _image_log_body(copy.deepcopy(payload))
        upstream_payload = {**payload, "model": translated_model}
        upstream_payload.pop("stream", None)
        kwargs = {}
        if multipart:
            # requests supplies the multipart boundary, not application/json.
            headers.pop("Content-Type", None)
            kwargs["data"] = [
                (key, translated_model if key == "model" else value)
                for key, value in request.form.items(multi=True) if key != "stream"
            ]
            kwargs["files"] = [
                (key, (file.filename, file.stream, file.mimetype))
                for key, file in request.files.items(multi=True)
            ]
            upstream_payload["_files"] = [
                {"field": key, "filename": file.filename, "content_type": file.mimetype}
                for key, file in request.files.items(multi=True)
            ]
            original_body["_files"] = upstream_payload["_files"]
        else:
            kwargs["json"] = upstream_payload

        request_id = str(uuid.uuid4())
        record = {
            "request_headers": redact_auth_headers(dict(request.headers)),
            "client_ip": get_client_ip(request),
            "original_request_body": original_body,
            "request_body": _image_log_body(upstream_payload),
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": endpoint,
            "request_size": request.content_length or len(json.dumps(payload)),
            "user_id": getattr(g, "user_id", "anonymous") or "anonymous",
        }
        cache.start_request(request_id, record)
        cache.update_request_state(request_id, cache.STATE_SENDING)
        try:
            response = requests.post(
                f"{get_copilot_base_url()}{upstream_path}", headers=headers,
                timeout=state.upstream_read_timeout, **kwargs,
            )
        except requests.RequestException as exc:
            message = (f"Upstream image request failed ({type(exc).__name__}). It was not retried; "
                       "generation may already have completed upstream. Check usage before resubmitting.")
            result = {"error": {"message": message, "type": "upstream_error", "code": "image_request_failed"}}
            status = 504 if isinstance(exc, requests.Timeout) else 502
            cache.complete_request(request_id, {
                **record, "response_body": result, "status_code": status,
                "duration": round(time.time() - start, 2),
            })
            return jsonify(result), status

        try:
            try:
                result = response.json()
            except ValueError:
                result = response.text
            usage_counts = responses_usage_counts(result.get("usage") if isinstance(result, dict) else None)
            cache.complete_request(request_id, {
                **record,
                "response_body": _image_log_body(result),
                "status_code": response.status_code,
                "response_size": len(response.content),
                **usage_counts,
                "duration": round(time.time() - start, 2),
            })
            return Response(response.content, status=response.status_code,
                            content_type=response.headers.get("Content-Type", "application/json"))
        finally:
            response.close()
    except RequestEntityTooLarge:
        return _image_request_too_large()
