import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.proxy.affinity import ProxyAffinityStore
from ghc_api.proxy.client import ProxyRuntime, transform_payload
from ghc_api.proxy.config import ProxyConfigError, ProxyRegistry, parse_proxy_config
from ghc_api.proxy.kimi_k3 import (
    ProxyPayloadError,
    fold_chat_messages,
    parse_native_tool_calls,
)
from ghc_api.routes import proxy as proxy_routes
from ghc_api.state import state


CONFIG = """
proxies:
  coding-models:
    auth: {type: none}
    apis:
      chat_completions:
        upstream_url: https://papyrus.example.test/chat/completions
        request_model: upstream
        response_model: public
        accept_mislabeled_sse: true
    models:
      kimi-k3-proxy:
        display_name: Kimi K3 (Configured Proxy)
        reasoning: true
        input: [text]
        context_window: 128000
        max_output_tokens: 16384
        headers:
          papyrus-model-name: Kimi-K3-Eval
        apis:
          chat_completions:
            upstream_model: Kimi-K3-Eval
            compatibility: kimi_k3_papyrus
      ordinary-proxy:
        apis:
          chat_completions:
            upstream_model: Ordinary-Upstream
"""


class FakeResponse:
    def __init__(self, payload=None, *, lines=None, content_type="application/json", status=200):
        self._payload = payload
        self._lines = lines or []
        self.status_code = status
        self.ok = status < 400
        self.headers = {"Content-Type": content_type}
        self.content = json.dumps(payload).encode() if payload is not None else b"stream"
        self.text = self.content.decode(errors="replace")
        self.closed = False

    def json(self):
        if self._payload is None:
            raise ValueError
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


@pytest.fixture()
def profile():
    return parse_proxy_config(yaml.safe_load(CONFIG)).profiles["coding-models"]


def test_config_accepts_known_and_rejects_unknown_compatibility(profile):
    assert profile.models["kimi-k3-proxy"].apis["chat_completions"].compatibility == "kimi_k3_papyrus"
    config = yaml.safe_load(CONFIG)
    config["proxies"]["coding-models"]["models"]["kimi-k3-proxy"]["apis"]["chat_completions"]["compatibility"] = "future_mode"
    with pytest.raises(ProxyConfigError, match="compatibility"):
        parse_proxy_config(config)
    config["proxies"]["coding-models"]["models"]["kimi-k3-proxy"]["apis"]["chat_completions"]["compatibility"] = ["kimi_k3_papyrus"]
    with pytest.raises(ProxyConfigError, match="compatibility"):
        parse_proxy_config(config)


def test_model_without_compatibility_is_unchanged(profile):
    api, model, model_api = profile.resolve("chat_completions", "ordinary-proxy")
    payload = {"model": "ordinary-proxy", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    result = transform_payload(payload, api, model, model_api)
    assert result["model"] == "Ordinary-Upstream"
    assert result["messages"] == payload["messages"]


def test_text_blocks_are_joined_and_full_context_is_folded(profile):
    api, model, model_api = profile.resolve("chat_completions", "kimi-k3-proxy")
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    payload = {
        "model": "kimi-k3-proxy",
        "stream": True,
        "max_tokens": 200,
        "tool_choice": "auto",
        "tools": tools,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "be "}, {"type": "text", "text": "careful"}]},
            {"role": "developer", "content": "edit files"},
            {"role": "user", "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]},
            {"role": "assistant", "reasoning_content": "need read", "content": "checking", "tool_calls": [{"id": "call_old", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.py\"}"}}]},
            {"role": "tool", "tool_call_id": "call_old", "content": "file text"},
            {"role": "user", "content": "continue"},
        ],
    }
    result = transform_payload(payload, api, model, model_api)
    folded = result["messages"][0]["content"]

    assert result["model"] == "Kimi-K3-Eval"
    assert result["messages"][0]["role"] == "user"
    assert "be careful" in folded
    assert folded.index("hello") < folded.index("need read") < folded.index("call_old") < folded.index("file text") < folded.index("continue")
    assert "name=\"read\"" in folded
    assert result["tools"] == tools
    assert result["tool_choice"] == "auto"
    assert result["max_tokens"] == 200
    assert result["stream"] is True


@pytest.mark.parametrize("block", [
    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
    {"type": "future_content", "value": "secret"},
])
def test_image_and_unknown_blocks_are_rejected(block):
    with pytest.raises(ProxyPayloadError, match="text-only"):
        fold_chat_messages({"messages": [{"role": "user", "content": [block]}]})


def test_native_tool_parser_supports_multiple_calls_and_all_types():
    native = """tool=\"all_types\" index=\"1\"<|sep|>
key=\"s\" type=\"string\"<|sep|>hello
key=\"n\" type=\"number\"<|sep|>2.5
key=\"i\" type=\"integer\"<|sep|>3
key=\"b\" type=\"boolean\"<|sep|>true
key=\"z\" type=\"null\"<|sep|>null
key=\"o\" type=\"object\"<|sep|>{\"x\":1}
key=\"a\" type=\"array\"<|sep|>[1,2]
tool=\"ping\" index=\"2\"<|sep|>"""
    calls = parse_native_tool_calls(native, ["all_types", "ping"])
    assert calls is not None
    assert [call["function"]["name"] for call in calls] == ["all_types", "ping"]
    assert len({call["id"] for call in calls}) == 2
    assert calls == parse_native_tool_calls(native, ["all_types", "ping"])
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "s": "hello", "n": 2.5, "i": 3, "b": True, "z": None,
        "o": {"x": 1}, "a": [1, 2],
    }


@pytest.mark.parametrize("native,allowed", [
    ('tool="undeclared" index="1"<|sep|>', ["other"]),
    ('ordinary tool="other" index="1"<|sep|>', ["other"]),
    ('tool="other" index="1"<|sep|> trailing', ["other"]),
    ('tool="other" index="1"<|sep|>\nkey="x" type="number"<|sep|>not-a-number', ["other"]),
    ('tool="other" index="1"<|sep|>\nkey="x" type="object"<|sep|>[]', ["other"]),
])
def test_native_tool_parser_fails_closed(native, allowed):
    assert parse_native_tool_calls(native, allowed) is None


@pytest.fixture()
def kimi_app():
    previous_auth = state.enable_auth
    state.enable_auth = False
    with cache.lock:
        cache.cache.clear()
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    path = root / "proxies.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    runtime = ProxyRuntime(ProxyRegistry(path), ProxyAffinityStore(root / "affinity.json"))
    patcher = mock.patch.object(proxy_routes, "proxy_runtime", runtime)
    patcher.start()
    app = create_app()
    try:
        yield app
    finally:
        patcher.stop()
        temp.cleanup()
        state.enable_auth = previous_auth
        with cache.lock:
            cache.cache.clear()


def test_route_rejects_non_text_before_post(kimi_app):
    with mock.patch("ghc_api.proxy.client.requests.post") as post:
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "kimi-k3-proxy",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            })
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "unsupported_content"
    post.assert_not_called()


def test_non_stream_thinking_and_native_tool_are_converted_and_cached(kimi_app):
    native = '<think>use add</think> tool="add" index="1"<|sep|>\n key="a" type="number"<|sep|>2\n key="b" type="integer"<|sep|>3'
    upstream_body = {
        "id": "chat-1", "model": "Kimi-K3-Eval",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": native}, "finish_reason": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    upstream = FakeResponse(upstream_body)
    tools = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]

    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "kimi-k3-proxy", "messages": [{"role": "user", "content": "sum"}], "tools": tools,
            })

    assert response.status_code == 200
    body = response.get_json()
    message = body["choices"][0]["message"]
    assert body["model"] == "kimi-k3-proxy"
    assert message["reasoning_content"] == "use add"
    assert message["content"] == ""
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"a": 2, "b": 3}
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    sent = post.call_args.kwargs["json"]
    assert sent["model"] == "Kimi-K3-Eval"
    assert len(sent["messages"]) == 1

    cached = next(iter(cache.cache.values()))
    assert cached["original_request_body"]["model"] == "kimi-k3-proxy"
    assert cached["request_body"] == sent
    assert cached["raw_response_body"] == upstream_body
    assert cached["public_response_body"] == body
    assert cached["model"] == "kimi-k3-proxy"
    assert cached["translated_model"] == "Kimi-K3-Eval"


def test_non_stream_malformed_or_undeclared_tool_remains_text(kimi_app):
    text = '<think>reason</think>tool="danger" index="1"<|sep|>\nkey="x" type="number"<|sep|>oops'
    upstream = FakeResponse({
        "id": "chat-2", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]
    })
    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "kimi-k3-proxy", "messages": [{"role": "user", "content": "go"}],
                "tools": [{"type": "function", "function": {"name": "safe"}}],
            })
    choice = response.get_json()["choices"][0]
    assert choice["message"]["reasoning_content"] == "reason"
    assert choice["message"]["content"].startswith('tool="danger"')
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


def _sse_json(body):
    return [
        json.loads(line[6:]) for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_stream_cross_chunk_thinking_standard_sse_and_synthesized_finish(kimi_app):
    lines = [
        b"event: message",
        b'data: {"id":"chat-s","object":"chat.completion.chunk","model":"Kimi-K3-Eval","choices":[{"index":0,"delta":{"role":"assistant","content":"<thi"},"finish_reason":null}]}',
        b'data: {"id":"chat-s","object":"chat.completion.chunk","model":"Kimi-K3-Eval","choices":[{"index":0,"delta":{"content":"nk>deep</thi"},"finish_reason":null}]}',
        b'data: {"id":"chat-s","object":"chat.completion.chunk","model":"Kimi-K3-Eval","choices":[{"index":0,"delta":{"content":"nk>answer"},"finish_reason":null}]}',
        b'[DONE]',
    ]
    upstream = FakeResponse(lines=lines, content_type="application/json")
    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "kimi-k3-proxy", "messages": [{"role": "user", "content": "go"}], "stream": True,
            })
            body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.content_type.startswith("text/event-stream")
    assert "<think>" not in body and "</think>" not in body
    events = _sse_json(body)
    reasoning = "".join(choice["delta"].get("reasoning_content", "") for event in events for choice in event.get("choices", []))
    content = "".join(choice["delta"].get("content", "") for event in events for choice in event.get("choices", []))
    assert reasoning == "deep"
    assert content == "answer"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["model"] == "kimi-k3-proxy"
    assert body.endswith("data: [DONE]\n\n")
    assert all(frame.startswith("data: ") for frame in body.strip().split("\n\n"))

    cached = next(iter(cache.cache.values()))
    assert cached["raw_events"]
    assert "[DONE]" in cached["raw_sse_lines"]
    assert cached["public_events"][-1] == json.dumps(events[-1], ensure_ascii=False, separators=(",", ":"))


def test_stream_native_multiple_tools_and_tool_finish(kimi_app):
    native = 'tool="add" index="1"<|sep|>\nkey="a" type="number"<|sep|>2\ntool="ping" index="2"<|sep|>'
    event = {"id": "chat-tools", "choices": [{"index": 0, "delta": {"content": native}, "finish_reason": None}]}
    upstream = FakeResponse(lines=[f"data: {json.dumps(event)}".encode(), b"data: [DONE]"], content_type="text/event-stream")
    tools = [
        {"type": "function", "function": {"name": "add"}},
        {"type": "function", "function": {"name": "ping"}},
    ]
    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "kimi-k3-proxy", "messages": [{"role": "user", "content": "go"}],
                "stream": True, "tools": tools,
            })
            body = response.get_data(as_text=True)
    events = _sse_json(body)
    tool_event = next(event for event in events if event["choices"][0]["delta"].get("tool_calls"))
    assert [call["function"]["name"] for call in tool_event["choices"][0]["delta"]["tool_calls"]] == ["add", "ping"]
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert body.endswith("data: [DONE]\n\n")


def test_ordinary_model_streaming_is_not_changed(kimi_app):
    raw = '{"id":"ordinary","choices":[{"index":0,"delta":{"content":"<think>visible</think>"}}]}'
    upstream = FakeResponse(lines=[f"data: {raw}".encode(), b"data: [DONE]"], content_type="text/event-stream")
    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with kimi_app.test_client() as client:
            response = client.post("/proxy/coding-models/v1/chat/completions", json={
                "model": "ordinary-proxy", "messages": [{"role": "user", "content": "go"}], "stream": True,
            })
            body = response.get_data(as_text=True)
    assert raw in body
    assert "<think>visible</think>" in body
