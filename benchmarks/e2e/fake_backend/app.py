"""Deterministic fake Copilot/LLM backend for ghc-api E2E benchmarks.

The response shapes are hand-authored from protocol field/type inventories. No
request-dump value is loaded, copied, echoed, logged, or needed at runtime.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from itertools import count
from typing import Dict, Iterable, List

from flask import Flask, Response, jsonify, request, stream_with_context


app = Flask(__name__)
_id_counter = count(1)
_id_lock = threading.Lock()


def _new_ids() -> Dict[str, str]:
    with _id_lock:
        n = next(_id_counter)
    suffix = f"{n:08d}"
    return {
        "response": f"resp_fake_{suffix}",
        "message": f"msg_fake_{suffix}",
        "reasoning": f"rs_fake_{suffix}",
        "function": f"fc_fake_{suffix}",
        "call": f"call_fake_{suffix}",
        "tool": f"toolu_fake_{suffix}",
        "web": f"ws_fake_{suffix}",
    }


def _benchmark_options(payload: Dict) -> Dict:
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    options = metadata.get("ghc_benchmark") if isinstance(metadata, dict) else None
    if not isinstance(options, dict):
        options = {}
    return {
        "profile": str(options.get("profile", "full")),
        "text_bytes": max(1, min(int(options.get("text_bytes", 1024)), 2 * 1024 * 1024)),
        "text_chunks": max(1, min(int(options.get("text_chunks", 16)), 4096)),
        "argument_bytes": max(2, min(int(options.get("argument_bytes", 256)), 512 * 1024)),
        "argument_chunks": max(1, min(int(options.get("argument_chunks", 4)), 1024)),
        "ttft_ms": max(0, min(int(options.get("ttft_ms", 0)), 60_000)),
        "chunk_delay_ms": max(0, min(int(options.get("chunk_delay_ms", 0)), 10_000)),
    }


def _synthetic_text(size: int) -> str:
    seed = "Offline fixture result segment confirms stable forwarding behavior without external data. "
    repeats = (size // len(seed)) + 1
    return (seed * repeats)[:size]


def _synthetic_arguments(size: int) -> str:
    base = {"path": "/isolated-fixture/tree/node-a.txt", "offset": 0, "limit": 20}
    encoded = json.dumps(base, separators=(",", ":"))
    if size <= len(encoded):
        return encoded
    base["padding"] = "x" * max(0, size - len(encoded) - len(',"padding":""'))
    return json.dumps(base, separators=(",", ":"))


def _split_text(value: str, chunks: int) -> List[str]:
    chunks = max(1, min(chunks, max(1, len(value))))
    base, extra = divmod(len(value), chunks)
    result = []
    offset = 0
    for index in range(chunks):
        width = base + (1 if index < extra else 0)
        result.append(value[offset:offset + width])
        offset += width
    return result


def _sse(events: Iterable[Dict], options: Dict, anthropic: bool = False) -> Response:
    def generate():
        first = True
        for event in events:
            delay_ms = options["ttft_ms"] if first else options["chunk_delay_ms"]
            first = False
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            event_type = event.get("type")
            event_header = f"event: {event_type}\n" if event_type else ""
            yield f"{event_header}data: {json.dumps(event, separators=(',', ':'))}\n\n"
        if not anthropic:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _usage(input_tokens: int = 256, output_tokens: int = 128) -> Dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 32},
        "output_tokens_details": {"reasoning_tokens": 48},
    }


def _copilot_usage() -> Dict:
    return {
        "token_details": [
            {"token_type": "input", "token_count": 256, "batch_size": 1, "cost_per_batch": 0},
            {"token_type": "output", "token_count": 128, "batch_size": 1, "cost_per_batch": 0},
            {"token_type": "cache_read", "token_count": 64, "batch_size": 1, "cost_per_batch": 0},
            {"token_type": "cache_write", "token_count": 32, "batch_size": 1, "cost_per_batch": 0},
        ],
        "total_nano_aiu": 0,
    }


def _tool_definition() -> Dict:
    return {
        "type": "function",
        "name": "inspect_fixture_asset",
        "description": "Inspect an isolated fixture asset used by the load harness.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": False,
        "output_schema": None,
    }


def _response_object(model: str, ids: Dict[str, str], status: str, output: List[Dict] | None = None) -> Dict:
    created = int(time.time())
    completed = created if status == "completed" else None
    return {
        "id": ids["response"],
        "object": "response",
        "created_at": created,
        "completed_at": completed,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 1024,
        "max_tool_calls": None,
        "model": model,
        "output": output or [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_retention": "in-memory",
        "store": False,
        "temperature": 0,
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_logprobs": 0,
        "truncation": "disabled",
        "tool_choice": "auto",
        "metadata": {"synthetic": True, "benchmark": "ghc-api"},
        "moderation": None,
        "safety_identifier": f"safety_{ids['response']}",
        "service_tier": "default",
        "reasoning": {"effort": "medium", "summary": "auto", "context": "synthetic", "mode": "enabled"},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tools": [_tool_definition()],
        "tool_usage": {
            "web_search": {"num_requests": 1},
            "image_gen": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
                "output_tokens_details": {"text_tokens": 0, "image_tokens": 0},
            },
        },
        "usage": _usage() if status == "completed" else None,
        "user": None,
    }


def _responses_items(ids: Dict[str, str], text: str, arguments: str, include_web: bool) -> List[Dict]:
    items = [
        {
            "id": ids["reasoning"],
            "type": "reasoning",
            "encrypted_content": "fixture_ciphertext_placeholder_v1",
            "summary": [{"type": "summary_text", "text": "The isolated fixture contains deterministic records ready for validation."}],
            "content": [],
        },
        {
            "id": ids["function"],
            "type": "function_call",
            "call_id": ids["call"],
            "name": "inspect_fixture_asset",
            "arguments": arguments,
            "status": "completed",
        },
    ]
    if include_web:
        items.append({
            "id": ids["web"],
            "type": "web_search_call",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "isolated protocol fixture reference",
                "queries": ["isolated protocol fixture reference", "offline stream event catalog"],
            },
        })
    items.append({
        "id": ids["message"],
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [{
                "type": "url_citation",
                "start_index": 0,
                "end_index": min(13, len(text)),
                "title": "Offline protocol fixture",
                "url": "https://perf-fixture.invalid/reference/streaming",
            }],
            "logprobs": [],
        }],
    })
    return items


def _responses_events(model: str, options: Dict, ids: Dict[str, str]) -> List[Dict]:
    profile = options["profile"]
    include_reasoning = profile in {"full", "reasoning_text", "tool", "web_search"}
    include_tool = profile in {"full", "tool", "web_search"}
    include_web = profile in {"full", "web_search"}
    text = _synthetic_text(options["text_bytes"])
    arguments = _synthetic_arguments(options["argument_bytes"])
    sequence = count(0)

    def event(event_type: str, **fields) -> Dict:
        return {"type": event_type, "sequence_number": next(sequence), **fields}

    events = [
        event("response.created", response=_response_object(model, ids, "in_progress")),
        event("response.in_progress", response=_response_object(model, ids, "in_progress")),
    ]

    if include_reasoning:
        reasoning = {
            "id": ids["reasoning"], "type": "reasoning",
            "encrypted_content": "fixture_ciphertext_placeholder_v1",
            "summary": [], "content": [],
        }
        events.append(event("response.output_item.added", output_index=0, item=reasoning))
        part = {"type": "summary_text", "text": ""}
        events.append(event("response.reasoning_summary_part.added", item_id=ids["reasoning"], output_index=0, summary_index=0, part=part))
        summary = "The isolated fixture contains deterministic records ready for validation."
        for delta in _split_text(summary, min(4, len(summary))):
            events.append(event("response.reasoning_summary_text.delta", item_id=ids["reasoning"], output_index=0, summary_index=0, delta=delta, obfuscation="fake"))
        events.append(event("response.reasoning_summary_text.done", item_id=ids["reasoning"], output_index=0, summary_index=0, text=summary))
        events.append(event("response.reasoning_summary_part.done", item_id=ids["reasoning"], output_index=0, summary_index=0, part={"type": "summary_text", "text": summary}))
        done_reasoning = dict(reasoning, summary=[{"type": "summary_text", "text": summary}])
        events.append(event("response.output_item.done", output_index=0, item=done_reasoning))

    output_index = 1 if include_reasoning else 0
    if include_tool:
        call_item = {"id": ids["function"], "type": "function_call", "call_id": ids["call"], "name": "inspect_fixture_asset", "arguments": "", "status": "in_progress"}
        events.append(event("response.output_item.added", output_index=output_index, item=call_item))
        for delta in _split_text(arguments, options["argument_chunks"]):
            events.append(event("response.function_call_arguments.delta", item_id=ids["function"], output_index=output_index, delta=delta, obfuscation="fake"))
        events.append(event("response.function_call_arguments.done", item_id=ids["function"], output_index=output_index, arguments=arguments))
        events.append(event("response.output_item.done", output_index=output_index, item={**call_item, "arguments": arguments, "status": "completed"}))
        output_index += 1

    if include_web:
        web_item = {"id": ids["web"], "type": "web_search_call", "status": "in_progress", "action": {"type": "search", "query": "isolated protocol fixture reference", "queries": ["isolated protocol fixture reference"]}}
        events.append(event("response.output_item.added", output_index=output_index, item=web_item))
        for state in ("in_progress", "searching", "completed"):
            events.append(event(f"response.web_search_call.{state}", item_id=ids["web"], output_index=output_index))
        events.append(event("response.output_item.done", output_index=output_index, item={**web_item, "status": "completed"}))
        output_index += 1

    message_item = {"id": ids["message"], "type": "message", "role": "assistant", "status": "in_progress", "phase": "final_answer", "content": []}
    events.append(event("response.output_item.added", output_index=output_index, item=message_item))
    part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
    events.append(event("response.content_part.added", item_id=ids["message"], output_index=output_index, content_index=0, part=part))
    for delta in _split_text(text, options["text_chunks"]):
        events.append(event("response.output_text.delta", item_id=ids["message"], output_index=output_index, content_index=0, delta=delta, logprobs=[], obfuscation="fake"))
    annotation = {"type": "url_citation", "start_index": 0, "end_index": min(13, len(text)), "title": "Offline protocol fixture", "url": "https://perf-fixture.invalid/reference/streaming"}
    events.append(event("response.output_text.annotation.added", item_id=ids["message"], output_index=output_index, content_index=0, annotation_index=0, annotation=annotation))
    events.append(event("response.output_text.done", item_id=ids["message"], output_index=output_index, content_index=0, text=text, logprobs=[]))
    final_part = {"type": "output_text", "text": text, "annotations": [annotation], "logprobs": []}
    events.append(event("response.content_part.done", item_id=ids["message"], output_index=output_index, content_index=0, part=final_part))
    events.append(event("response.output_item.done", output_index=output_index, item={**message_item, "status": "completed", "content": [final_part]}))

    output = _responses_items(ids, text, arguments, include_web)
    events.append(event("response.completed", response=_response_object(model, ids, "completed", output), copilot_usage=_copilot_usage()))
    return events


def _anthropic_response(model: str, options: Dict, ids: Dict[str, str]) -> Dict:
    text = _synthetic_text(options["text_bytes"])
    return {
        "id": ids["message"],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "thinking", "thinking": "I will validate the isolated fixture and produce a deterministic result."},
            {"type": "text", "text": text},
            {"type": "tool_use", "id": ids["tool"], "name": "inspect_fixture_asset", "input": {"path": "/isolated-fixture/tree/node-a.txt", "offset": 0, "limit": 20}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 128,
            "output_tokens": 96,
            "cache_creation_input_tokens": 32,
            "cache_read_input_tokens": 64,
        },
    }


def _anthropic_events(model: str, options: Dict, ids: Dict[str, str]) -> List[Dict]:
    profile = options["profile"]
    include_thinking = profile in {"full", "reasoning_text", "tool"}
    include_tool = profile in {"full", "tool"}
    text = _synthetic_text(options["text_bytes"])
    events = [{
        "type": "message_start",
        "message": {
            "id": ids["message"], "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None, "stop_details": None,
            "usage": {
                "input_tokens": 128, "output_tokens": 0,
                "cache_creation_input_tokens": 32, "cache_read_input_tokens": 64,
                "cache_creation": {"ephemeral_5m_input_tokens": 32, "ephemeral_1h_input_tokens": 0},
            },
        },
    }]
    index = 0
    if include_thinking:
        events.append({"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": ""}})
        events.append({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": "I will validate the isolated fixture and produce a deterministic result."}})
        events.append({"type": "content_block_stop", "index": index})
        index += 1

    events.append({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
    for delta in _split_text(text, options["text_chunks"]):
        events.append({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": delta}})
    events.append({"type": "content_block_stop", "index": index})
    index += 1

    if include_tool:
        events.append({"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": ids["tool"], "name": "inspect_fixture_asset", "input": {}}})
        arguments = _synthetic_arguments(options["argument_bytes"])
        for delta in _split_text(arguments, options["argument_chunks"]):
            events.append({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": delta}})
        events.append({"type": "content_block_stop", "index": index})

    events.extend([
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if include_tool else "end_turn", "stop_sequence": None, "stop_details": None},
            "usage": {"input_tokens": 128, "output_tokens": 96, "cache_creation_input_tokens": 32, "cache_read_input_tokens": 64},
            "copilot_usage": _copilot_usage(),
        },
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {"inputTokenCount": 128, "outputTokenCount": 96, "invocationLatency": 25, "firstByteLatency": 5},
        },
    ])
    return events


@app.get("/health")
def health():
    return jsonify({"ok": True, "synthetic": True})


@app.get("/copilot_internal/v2/token")
def token():
    return jsonify({
        "token": "fake_copilot_token_for_benchmark_only",
        "refresh_in": 86400,
        "expires_at": int(time.time()) + 86400,
    })


@app.get("/copilot_internal/user")
def copilot_user():
    return jsonify({
        "access_type_sku": "individual",
        "analytics_tracking_id": "fake_tracking_id",
        "assigned_date": "2026-01-01",
        "can_signup_for_limited": False,
        "chat_enabled": True,
        "copilot_plan": "individual",
        "organization_login_list": [],
        "organization_list": [],
        "quota_reset_date": "2099-01-01",
        "quota_snapshots": {},
        "token_based_billing": False,
    })


@app.get("/user")
def github_user():
    return jsonify({
        "login": "fake-benchmark-user",
        "id": 1,
        "name": "Fake Benchmark User",
        "email": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "two_factor_authentication": True,
    })


@app.get("/repos/microsoft/vscode/releases/latest")
def vscode_release():
    return jsonify({"tag_name": "1.104.3"})


@app.get("/models")
def models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "fake-opus",
                "name": "Fake Opus",
                "object": "model",
                "vendor": "Anthropic",
                "version": "benchmark-v1",
                "preview": False,
                "model_picker_enabled": True,
                "model_picker_category": "chat",
                "is_chat_default": False,
                "is_chat_fallback": False,
                "supported_endpoints": ["/v1/messages", "/responses"],
                "capabilities": {
                    "family": "claude-opus",
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high"], "streaming": True},
                    "limits": {"max_context_window_tokens": 200000, "max_prompt_tokens": 190000, "max_output_tokens": 32000},
                },
            },
            {
                "id": "fake-gpt",
                "name": "Fake GPT",
                "object": "model",
                "vendor": "OpenAI",
                "version": "benchmark-v1",
                "preview": False,
                "model_picker_enabled": True,
                "model_picker_category": "chat",
                "is_chat_default": False,
                "is_chat_fallback": False,
                "supported_endpoints": ["/responses"],
                "capabilities": {
                    "family": "gpt",
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high"], "streaming": True},
                    "limits": {"max_context_window_tokens": 200000, "max_prompt_tokens": 190000, "max_output_tokens": 32000},
                },
            },
        ],
    })


@app.post("/v1/messages")
def messages():
    payload = request.get_json(silent=True) or {}
    options = _benchmark_options(payload)
    ids = _new_ids()
    model = str(payload.get("model") or "fake-opus")
    if payload.get("stream"):
        return _sse(_anthropic_events(model, options, ids), options, anthropic=True)
    return jsonify(_anthropic_response(model, options, ids))


@app.post("/v1/responses")
@app.post("/responses")
def responses():
    payload = request.get_json(silent=True) or {}
    options = _benchmark_options(payload)
    ids = _new_ids()
    model = str(payload.get("model") or "fake-gpt")
    events = _responses_events(model, options, ids)
    if payload.get("stream"):
        return _sse(events, options)
    return jsonify(events[-1]["response"])


@app.post("/chat/completions")
def chat_completions():
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model") or "fake-gpt")
    ids = _new_ids()
    text = _synthetic_text(1024)
    if payload.get("stream"):
        chunks = [
            {"id": ids["response"], "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": ids["response"], "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
            {"id": ids["response"], "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 128, "completion_tokens": 96, "total_tokens": 224}},
        ]
        return _sse(chunks, _benchmark_options(payload))
    return jsonify({
        "id": ids["response"], "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 128, "completion_tokens": 96, "total_tokens": 224},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18400)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
