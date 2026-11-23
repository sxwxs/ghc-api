"""A Copilot token rejected with 401/403 is refreshed before the next request."""

import json
import time
from unittest import mock

import pytest
import requests
from flask import Flask

from ghc_api import api_helpers
from ghc_api.api_helpers import copilot_post, reset_copilot_token_on_auth_error
from ghc_api.routes import openai
from ghc_api.state import state


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = body
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def cached_token(monkeypatch):
    expires_at = time.time() + 86400
    monkeypatch.setattr(state, "copilot_token", "cached")
    monkeypatch.setattr(state, "token_expires_at", expires_at)
    return expires_at


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_error_expires_the_token_that_was_sent(cached_token, status_code):
    reset_copilot_token_on_auth_error(
        FakeResponse(status_code, "forbidden\n"), {"Authorization": "Bearer cached"}
    )

    assert state.token_expires_at == 0
    assert state.copilot_token == "cached"


@pytest.mark.parametrize("status_code", [200, 400, 402, 429, 500])
def test_other_statuses_keep_the_token(cached_token, status_code):
    reset_copilot_token_on_auth_error(
        FakeResponse(status_code, "{}"), {"Authorization": "Bearer cached"}
    )

    assert state.token_expires_at == cached_token


def test_auth_error_for_a_replaced_token_keeps_the_new_token(cached_token):
    reset_copilot_token_on_auth_error(
        FakeResponse(403, "forbidden\n"), {"Authorization": "Bearer older"}
    )

    assert state.token_expires_at == cached_token


def test_copilot_post_passes_through_to_requests(cached_token, monkeypatch):
    upstream = FakeResponse(403, "forbidden\n")
    post = mock.Mock(return_value=upstream)
    monkeypatch.setattr(requests, "post", post)
    headers = {"Authorization": "Bearer cached"}

    response = copilot_post("https://upstream.test/chat/completions", headers=headers,
                            json={"a": 1}, stream=True, timeout=5)

    assert response is upstream
    post.assert_called_once_with("https://upstream.test/chat/completions", headers=headers,
                                 json={"a": 1}, stream=True, timeout=5)
    assert state.token_expires_at == 0


def test_fetch_models_auth_error_expires_the_token(cached_token, monkeypatch):
    monkeypatch.setattr(requests, "get", mock.Mock(return_value=FakeResponse(401, "{}")))

    assert api_helpers.fetch_models() is False
    assert state.token_expires_at == 0


def test_revoked_token_is_replaced_on_the_next_chat_request(monkeypatch):
    """Copilot revoked a token before its 24h expiry; only one request should fail."""
    monkeypatch.setattr(state, "github_token", "github-token")
    monkeypatch.setattr(state, "copilot_token", "revoked")
    monkeypatch.setattr(state, "token_expires_at", time.time() + 86400)
    monkeypatch.setattr(state, "max_connection_retries", 0)
    monkeypatch.setattr(openai, "translate_model_name", lambda model: model)
    monkeypatch.setattr(openai, "is_non_chat_model", lambda model: False)
    monkeypatch.setattr(openai, "_should_route_chat_completions_via_responses", lambda model: False)
    monkeypatch.setattr(openai.cache, "add_request", mock.Mock())
    monkeypatch.setattr(openai, "log_error_request", mock.Mock())

    token_endpoint = mock.Mock(return_value=FakeResponse(
        200, '{"token": "fresh", "refresh_in": 86340}'
    ))
    monkeypatch.setattr(requests, "get", token_endpoint)

    def upstream(url, headers, **kwargs):
        if headers["Authorization"] == "Bearer revoked":
            return FakeResponse(403, "forbidden\n")
        return FakeResponse(200, '{"id": "ok", "usage": {}}')

    post = mock.Mock(side_effect=upstream)
    monkeypatch.setattr(requests, "post", post)

    app = Flask(__name__)
    app.register_blueprint(openai.openai_bp)
    client = app.test_client()
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 403
    assert second.status_code == 200, second.data
    token_endpoint.assert_called_once()
    assert token_endpoint.call_args.args[0].endswith("/copilot_internal/v2/token")
    assert [call.kwargs["headers"]["Authorization"] for call in post.call_args_list] == [
        "Bearer revoked", "Bearer fresh",
    ]
    assert state.copilot_token == "fresh"
