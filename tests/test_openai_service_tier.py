from unittest import mock

import pytest
from flask import Flask, Response

from ghc_api.routes import openai


@pytest.fixture
def client(monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(openai.openai_bp)
    monkeypatch.setattr(openai, "ensure_copilot_token", mock.Mock())
    monkeypatch.setattr(openai, "get_copilot_headers", mock.Mock(return_value={}))
    monkeypatch.setattr(
        openai, "get_copilot_base_url", lambda: "https://upstream.test"
    )
    monkeypatch.setattr(openai, "supports_responses_api", lambda model: True)
    monkeypatch.setattr(openai, "translate_model_name", lambda model: model)
    monkeypatch.setattr(
        openai, "_should_route_chat_completions_via_responses", lambda model: False
    )
    monkeypatch.setattr(openai.state, "max_connection_retries", 0)
    monkeypatch.setattr(openai.state, "sse_keepalive_interval", 0)
    monkeypatch.setattr(openai.cache, "add_request", mock.Mock())
    upstream = mock.Mock(
        status_code=200, ok=True, text='{"id":"test","usage":{}}'
    )
    upstream.json.return_value = {"id": "test", "usage": {}}
    monkeypatch.setattr(openai.requests, "post", mock.Mock(return_value=upstream))
    return app.test_client()


@pytest.mark.parametrize(
    "path", ["/v1/responses", "/responses", "/v1/chat/completions", "/chat/completions"]
)
@pytest.mark.parametrize("tier", ["priority", "auto", "default", "flex", None, "absent"])
def test_nonstream_omits_service_tier_and_preserves_original(client, path, tier):
    payload = {"model": "test-model", "stream": False, "temperature": 0.5}
    if "responses" in path:
        payload["input"] = "hello"
    else:
        payload["messages"] = [{"role": "user", "content": "hello"}]
    if tier != "absent":
        payload["service_tier"] = tier

    response = client.post(path, json=payload)

    assert response.status_code == 200, response.data
    expected = {key: value for key, value in payload.items() if key != "service_tier"}
    assert openai.requests.post.call_args.kwargs["json"] == expected
    cached = openai.cache.add_request.call_args.args[1]
    assert cached["original_request_body"] == payload
    assert cached["request_body"] == expected
    assert response.json == {"id": "test", "usage": {}}


@pytest.mark.parametrize("keepalive", [0, 15])
def test_streaming_responses_omits_service_tier_before_post(client, monkeypatch, keepalive):
    monkeypatch.setattr(openai.state, "sse_keepalive_interval", keepalive)
    monkeypatch.setattr(openai.state, "responses_pre_header_grace", 1)
    handler = mock.Mock()
    handler.stream.return_value = Response(
        "data: test\n\n", mimetype="text/event-stream"
    )
    factory = mock.Mock(return_value=handler)
    monkeypatch.setattr(openai, "_build_responses_stream_handler", factory)
    payload = {
        "model": "test-model",
        "input": "hello",
        "stream": True,
        "service_tier": "priority",
    }

    response = client.post("/v1/responses", json=payload)

    assert response.status_code == 200, response.data
    assert response.data == b"data: test\n\n"
    expected = {key: value for key, value in payload.items() if key != "service_tier"}
    assert openai.requests.post.call_args.kwargs["json"] == expected
    assert openai.requests.post.call_args.kwargs["stream"] is True
    assert factory.call_args.args[7] == payload


@pytest.mark.parametrize(
    "via_responses,stream",
    [(False, True), (True, False), (True, True)],
)
def test_chat_omits_service_tier_before_delegation(
    client, monkeypatch, via_responses, stream
):
    monkeypatch.setattr(
        openai, "_should_route_chat_completions_via_responses",
        lambda model: via_responses,
    )
    forward = mock.Mock(return_value=Response("test"))
    handler_name = (
        "chat_completions_via_responses" if via_responses else "stream_chat_completions"
    )
    monkeypatch.setattr(openai, handler_name, forward)
    payload = {
        "model": "test-model",
        "messages": [],
        "stream": stream,
        "service_tier": "priority",
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.data
    expected = {key: value for key, value in payload.items() if key != "service_tier"}
    assert forward.call_args.args[0] == expected
    assert forward.call_args.args[8] == payload
