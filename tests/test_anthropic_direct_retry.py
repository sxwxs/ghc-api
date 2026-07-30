"""Non-streaming direct Anthropic path: only repairable errors are retried."""

import json

import pytest

from ghc_api.routes import anthropic as anthropic_routes
from ghc_api.state import state


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    """Capture upstream POSTs and neutralise token refresh / disk logging."""
    calls = []
    responses = []

    def fake_post(url, **kwargs):
        calls.append(kwargs.get("json"))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(anthropic_routes.requests, "post", fake_post)
    monkeypatch.setattr(anthropic_routes, "ensure_copilot_token", lambda: None)
    monkeypatch.setattr(anthropic_routes, "get_anthropic_headers", lambda *a, **k: {})
    monkeypatch.setattr(anthropic_routes, "get_copilot_base_url", lambda: "https://upstream.test")
    monkeypatch.setattr(anthropic_routes, "log_error_request", lambda *a, **k: None)
    monkeypatch.setattr(anthropic_routes, "log_tool_result_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(anthropic_routes.cache, "add_request", lambda *a, **k: None)
    return calls, responses


def _call(payload=None):
    from ghc_api import create_app

    app = create_app()
    with app.test_request_context("/v1/messages", json={}):
        return anthropic_routes.handle_direct_anthropic_request(
            payload or {"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hi"}]},
            "req-1",
            0.0,
            "claude-sonnet-4",
            "claude-sonnet-4",
        )


@pytest.mark.parametrize("status", [429, 500, 400])
def test_unrepairable_errors_are_returned_without_resending(upstream, status):
    calls, responses = upstream
    responses.append(FakeResponse(status, {"type": "error", "error": {"message": "nope"}}))

    response = _call()

    assert len(calls) == 1, "upstream request must not be repeated"
    assert response.status_code == status


def test_orphaned_tool_result_error_is_repaired_and_retried(upstream):
    calls, responses = upstream
    responses.append(FakeResponse(400, {
        "error": {
            "message": "messages.0.content.0: unexpected `tool_use_id` found in "
                       "`tool_result` blocks: toolu_orphan. Each `tool_result` block "
                       "must have a corresponding `tool_use` block."
        }
    }))
    responses.append(FakeResponse(200, {"id": "msg_1", "usage": {}}))

    payload = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "x"}],
        }],
    }
    response = _call(payload)

    assert len(calls) == 2, "the repaired payload must be retried once"
    assert calls[1]["messages"] == []
    assert response.status_code == 200


def test_web_search_unsupported_error_is_repaired_and_retried(upstream, monkeypatch):
    calls, responses = upstream
    responses.append(FakeResponse(400, {"error": {"message": "web search is not supported"}}))
    responses.append(FakeResponse(200, {"id": "msg_1", "usage": {}}))
    monkeypatch.setattr(anthropic_routes, "call_search_proxy", lambda *a, **k: [], raising=False)
    monkeypatch.setattr("ghc_api.web_search.call_search_proxy", lambda *a, **k: [])

    original = state.enable_web_search_proxy
    state.enable_web_search_proxy = True
    try:
        payload = {
            "model": "claude-sonnet-4",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "find something"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        response = _call(payload)
    finally:
        state.enable_web_search_proxy = original

    assert len(calls) == 2
    assert "tools" not in calls[1]
    assert response.status_code == 200


def test_orphaned_tool_use_id_extraction_is_not_truncated():
    """IDs containing 'n' used to be cut short by a literal stop-char set."""
    from ghc_api.utils import extract_orphaned_tool_use_ids

    error = (
        '{"error":{"message":"messages.0.content.0: unexpected `tool_use_id` found in '
        '`tool_result` blocks: toolu_01Nan9xyz. Each `tool_result` block must have a '
        'corresponding `tool_use` block in the previous message."}}'
    )
    assert extract_orphaned_tool_use_ids(error) == ["toolu_01Nan9xyz"]
