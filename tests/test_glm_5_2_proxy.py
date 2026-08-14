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
from ghc_api.proxy.glm_5_2 import declared_tool_schemas, parse_native_tool_calls
from ghc_api.routes import proxy as proxy_routes
from ghc_api.state import state


CONFIG = """
proxies:
  glm-5-2:
    auth: {type: none}
    headers:
      x-api-key: test-key
    apis:
      chat_completions:
        upstream_url: http://glm.example.test/v1/chat/completions
        request_model: omit
        response_model: public
        accept_mislabeled_sse: true
    models:
      glm-5.2-proxy:
        display_name: GLM 5.2
        reasoning: true
        input: [text]
        apis:
          chat_completions:
            upstream_model: null
            compatibility: glm_5_2_nvfp4
      ordinary-proxy:
        apis:
          chat_completions:
            upstream_model: null
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
    return parse_proxy_config(yaml.safe_load(CONFIG)).profiles["glm-5-2"]


def test_config_accepts_glm_compatibility_only_for_chat_completions(profile):
    assert profile.models["glm-5.2-proxy"].apis["chat_completions"].compatibility == "glm_5_2_nvfp4"
    config = yaml.safe_load(CONFIG)
    api = config["proxies"]["glm-5-2"]["models"]["glm-5.2-proxy"]["apis"].pop("chat_completions")
    config["proxies"]["glm-5-2"]["apis"]["responses"] = {
        "upstream_url": "http://glm.example.test/v1/responses"
    }
    config["proxies"]["glm-5-2"]["models"]["glm-5.2-proxy"]["apis"]["responses"] = api
    with pytest.raises(ProxyConfigError, match="only valid for chat_completions"):
        parse_proxy_config(config)


def test_request_folds_pi_text_blocks_and_preserves_tools(profile):
    api, model, model_api = profile.resolve("chat_completions", "glm-5.2-proxy")
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    payload = {
        "model": "glm-5.2-proxy",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [{"type": "text", "text": "Say "}, {"type": "text", "text": "hello"}]},
        ],
        "tools": tools,
        "stream": True,
    }

    result = transform_payload(payload, api, model, model_api)

    assert "model" not in result
    assert result["messages"][0]["role"] == "user"
    assert isinstance(result["messages"][0]["content"], str)
    assert "Be concise." in result["messages"][0]["content"]
    assert "Say hello" in result["messages"][0]["content"]
    assert result["tools"] == tools


def test_ordinary_route_does_not_fold_content(profile):
    api, model, model_api = profile.resolve("chat_completions", "ordinary-proxy")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    result = transform_payload({"model": "ordinary-proxy", "messages": messages}, api, model, model_api)
    assert result["messages"] == messages


def test_native_tool_parser_handles_schema_types_and_discards_observation():
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["text", "count"],
            },
        },
    }]
    native = (
        "<tool_call>echo"
        "<arg_key>text</arg_key><arg_value>hello</arg_value>"
        "<arg_key>count</arg_key><arg_value>2</arg_value>"
        "</tool_call><|observation|><|observation|>hallucinated result"
    )

    calls = parse_native_tool_calls(native, declared_tool_schemas(tools))

    assert calls is not None
    assert calls[0]["function"]["name"] == "echo"
    assert json.loads(calls[0]["function"]["arguments"]) == {"count": 2, "text": "hello"}


@pytest.mark.parametrize("native", [
    "ordinary text <tool_call>echo</tool_call>",
    "<tool_call>undeclared</tool_call>",
    "<tool_call>echo<arg_key>count</arg_key><arg_value>not-an-int</arg_value></tool_call>",
    "<tool_call>echo<arg_key>unknown</arg_key><arg_value>x</arg_value></tool_call>",
])
def test_native_tool_parser_fails_closed(native):
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["text", "count"],
            },
        },
    }]
    assert parse_native_tool_calls(native, declared_tool_schemas(tools)) is None


@pytest.fixture()
def glm_app():
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


def _sse_json(body):
    return [
        json.loads(line[6:]) for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_non_stream_response_splits_thinking_and_converts_native_tool(glm_app):
    content = (
        "<think>need echo</think>"
        "<tool_call>echo<arg_key>text</arg_key><arg_value>hello</arg_value></tool_call>"
        "<|observation|>fake"
    )
    upstream = FakeResponse({
        "id": "glm-1",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
    })
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]

    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with glm_app.test_client() as client:
            response = client.post("/proxy/glm-5-2/v1/chat/completions", json={
                "model": "glm-5.2-proxy",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "echo"}]}],
                "tools": tools,
            })

    choice = response.get_json()["choices"][0]
    assert response.status_code == 200
    assert choice["message"]["reasoning_content"] == "need echo"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "echo"
    assert choice["finish_reason"] == "tool_calls"


def test_stream_response_is_standard_and_synthesizes_tool_finish(glm_app):
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    chunks = ["<thi", "nk>need echo</think><tool_call>echo<arg_key>text</arg_key>", "<arg_value>hello</arg_value></tool_call><|observation|>fake"]
    lines = [
        ("data: " + json.dumps({
            "id": "glm-stream",
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        })).encode()
        for chunk in chunks
    ] + [b"[DONE]"]
    upstream = FakeResponse(lines=lines, content_type="application/json")

    with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
        with glm_app.test_client() as client:
            response = client.post("/proxy/glm-5-2/v1/chat/completions", json={
                "model": "glm-5.2-proxy",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "echo"}]}],
                "tools": tools,
                "stream": True,
            })
            body = response.get_data(as_text=True)

    events = _sse_json(body)
    reasoning = "".join(
        choice["delta"].get("reasoning_content", "")
        for event in events for choice in event.get("choices", [])
    )
    tool_event = next(event for event in events if event["choices"][0]["delta"].get("tool_calls"))
    assert response.status_code == 200
    assert response.content_type.startswith("text/event-stream")
    assert reasoning == "need echo"
    assert tool_event["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "echo"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert body.endswith("data: [DONE]\n\n")
    assert "<think>" not in body and "<tool_call>" not in body and "fake" not in body

    cached = next(iter(cache.cache.values()))
    assert cached["compatibility"] == "glm_5_2_nvfp4"
