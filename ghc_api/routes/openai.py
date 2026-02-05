"""
OpenAI-compatible API routes
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Generator

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..api_helpers import ensure_copilot_token, get_copilot_base_url, get_copilot_headers
from ..cache import cache
from ..state import state
from ..streaming import reconstruct_openai_response_from_chunks

openai_bp = Blueprint('openai', __name__)


@openai_bp.route("/v1/models", methods=["GET"])
@openai_bp.route("/models", methods=["GET"])
def list_models():
    """List available models"""
    from ..api_helpers import fetch_models

    try:
        ensure_copilot_token()
        if not state.models:
            fetch_models()

        models = [
            {
                "id": m["id"],
                "object": "model",
                "type": "model",
                "created": 0,
                "created_at": datetime.utcfromtimestamp(0).isoformat() + "Z",
                "owned_by": m.get("vendor", "unknown"),
                "display_name": m.get("name", m["id"]),
            }
            for m in state.models.get("data", [])
        ]

        return jsonify({
            "object": "list",
            "data": models,
            "has_more": False,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@openai_bp.route("/v1/chat/completions", methods=["POST"])
@openai_bp.route("/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completions (OpenAI format)"""
    try:
        start_time = time.time()
        ensure_copilot_token()
        payload = request.get_json()
        request_id = str(uuid.uuid4())

        # Check for vision content
        enable_vision = False
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                if any(p.get("type") == "image_url" for p in content):
                    enable_vision = True
                    break

        # Detect agent vs user call
        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        request_body = json.dumps(payload)
        request_size = len(request_body)

        if payload.get("stream"):
            return stream_chat_completions(payload, headers, request_id, request_body, request_size, start_time)

        # Non-streaming request
        response = requests.post(
            f"{get_copilot_base_url()}/chat/completions",
            headers=headers,
            json=payload,
            timeout=1200,
        )

        duration = round(time.time() - start_time, 2)
        response_body = response.text
        response_size = len(response_body)

        if response.ok:
            result = response.json()

            # Cache the request/response
            usage = result.get("usage", {})
            cache.add_request(request_id, {
                "request_body": payload,
                "response_body": result,
                "model": payload.get("model", "unknown"),
                "endpoint": "/v1/chat/completions",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "duration": duration,
            })

            return jsonify(result)
        else:
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def stream_chat_completions(payload: Dict, headers: Dict, request_id: str,
                            request_body: str, request_size: int, start_time: float) -> Response:
    """Handle streaming chat completions"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_body": payload,
        "model": payload.get("model", "unknown"),
        "endpoint": "/v1/chat/completions",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        # Track tokens from streaming chunks
                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)

                        yield f"data: {data}\n\n"
                    except json.JSONDecodeError:
                        continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this clean up
            error_occurred = True
            print(f"[Stream] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the full response from chunks
        reconstructed_response = reconstruct_openai_response_from_chunks(response_chunks)
        if error_occurred and not reconstructed_response:
            reconstructed_response = {"error": "Stream interrupted"}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": payload,
            "response_body": reconstructed_response,
            "model": payload.get("model", "unknown"),
            "endpoint": "/v1/chat/completions",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "duration": duration,
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
