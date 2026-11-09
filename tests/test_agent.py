"""The /agent page: the ghc-api provider built from the Copilot catalog, the pyagent bridge, and its routes."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from ghc_api import agent_bridge
from ghc_api.agent_bridge import AgentBridge, catalog_models, ghc_setup, self_url
from ghc_api.app import create_app
from ghc_api.auth import INTERNAL_TOKEN
from ghc_api.pyagent.ai.models import ProviderOverride, merge_providers
from ghc_api.pyagent.ai.providers.faux import FauxProvider, text_reply, tool_reply
from ghc_api.pyagent.coding.agent_session import SessionSetup
from ghc_api.pyagent.coding.config import Config
from ghc_api.routes import agent as agent_routes
from ghc_api.state import state

CATALOG = [
    {"id": "claude-x", "supported_endpoints": ["/v1/messages", "/chat/completions"],
     "capabilities": {"type": "chat", "limits": {"max_context_window_tokens": 200000, "max_output_tokens": 64000},
                      "supports": {"tool_calls": True, "vision": True, "max_thinking_budget": 32000}}},
    {"id": "gpt-x", "supported_endpoints": ["/responses", "/v1/chat/completions"],
     "capabilities": {"type": "chat", "limits": {"max_context_window_tokens": 400000},
                      "supports": {"tool_calls": True, "reasoning_effort": ["low", "high"]}}},
    {"id": "gemini-x", "supported_endpoints": ["/chat/completions"],
     "capabilities": {"type": "chat", "supports": {"tool_calls": True, "reasoning_effort": ["low"]}}},
    {"id": "old-x", "capabilities": {"type": "chat", "supports": {}}},
    {"id": "no-tools", "supported_endpoints": ["/chat/completions"],
     "capabilities": {"type": "chat", "supports": {"tool_calls": False}}},
    {"id": "embed-x", "capabilities": {"type": "embeddings"}},
]


def test_catalog_models_pick_the_best_endpoint_per_model():
    models = catalog_models(CATALOG)
    assert list(models) == ["claude-x", "gpt-x", "gemini-x", "old-x"]
    claude, gpt, gemini, old = models.values()
    assert (claude.api, claude.thinking_format, claude.reasoning, claude.images) == (
        "anthropic", "anthropic_budget", True, True)
    assert (claude.context_window, claude.max_tokens) == (200000, 64000)
    assert (gpt.api, gpt.reasoning, gpt.context_window, gpt.max_tokens) == ("openai_responses", True, 400000, 32000)
    assert (gemini.api, gemini.thinking_format) == ("openai_chat", "reasoning_effort")
    assert (old.api, old.reasoning, old.thinking_format) == ("openai_chat", False, "none")


def test_ghc_setup_points_back_at_this_server_with_the_internal_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    with mock.patch.object(state, "models", {"data": CATALOG}), \
            mock.patch.object(state, "server_host", "0.0.0.0"), mock.patch.object(state, "server_port", 9000):
        setup = ghc_setup()
    provider = setup.providers["ghc-api"]
    assert (provider.base_url, provider.api_key, provider.default_model) == (
        "http://127.0.0.1:9000", INTERNAL_TOKEN, "claude-x")
    assert (setup.config.default_provider, setup.config.default_model) == ("ghc-api", "claude-x")
    assert setup.home == tmp_path / "pyagent"


@pytest.mark.parametrize("host, url", [
    ("localhost", "http://localhost:8313"), ("0.0.0.0", "http://127.0.0.1:8313"),
    ("::", "http://[::1]:8313"), ("10.0.0.5", "http://10.0.0.5:8313"),
])
def test_self_url(host, url):
    with mock.patch.object(state, "server_host", host), mock.patch.object(state, "server_port", 8313):
        assert self_url() == url


@pytest.fixture
def bridge(tmp_path):
    provider = FauxProvider([])
    config = Config(default_provider="test", default_model="m1",
                    providers={"test": ProviderOverride(api="openai_chat", base_url="http://unused")})

    def make_setup():
        return SessionSetup(home=tmp_path / "home", config=config, providers=merge_providers(config.providers),
                            provider=provider)

    instance = AgentBridge(make_setup)
    instance.provider = provider
    yield instance
    instance.close()


def run_to_end(bridge, session_id, since):
    """Collect events after `since` until run_end."""
    events = []
    while not any(e["event"]["type"] == "run_end" for e in events):
        batch = bridge.wait_events(session_id, since, 5)
        assert batch, "no events within 5 seconds"
        events += [json.loads(text) for _, text in batch]
        since = batch[-1][0]
    return events


def test_bridge_runs_a_session_and_keeps_the_current_runs_events(bridge, tmp_path):
    bridge.provider.replies += [tool_reply(("c1", "read", {"path": "missing.txt"})), text_reply("done"),
                                text_reply("again")]
    state_ = bridge.call("session/new", {"cwd": str(tmp_path)})["result"]
    session_id = state_["sessionId"]
    assert bridge.call("run/prompt", {"sessionId": session_id, "content": [{"type": "text", "text": "hi"}]})[
        "result"]["disposition"] == "started"

    events = run_to_end(bridge, session_id, state_["lastSeq"])
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["event"]["type"] == "run_start"
    assert any(e["event"]["type"] == "tool_end" for e in events)

    messages = bridge.call("session/messages", {"sessionId": session_id})["result"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool_result", "assistant"]

    # A second run drops the first run's events: a client that missed them must reload.
    bridge.call("run/prompt", {"sessionId": session_id, "content": [{"type": "text", "text": "more"}]})
    second = run_to_end(bridge, session_id, events[-1]["seq"])
    assert second[0]["event"]["type"] == "run_start"
    assert bridge.wait_events(session_id, 0, 0) is None
    assert bridge.wait_events(session_id, second[-1]["seq"], 0) in ([], [(second[-1]["seq"] + 1, mock.ANY)])

    assert "result" in bridge.call("session/close", {"sessionId": session_id})
    assert bridge.wait_events(session_id, 0, 0) is None

    listed = bridge.call("session/list", {"all": True})["result"]["sessions"]
    assert [s["sessionId"] for s in listed] == [session_id]


def test_bridge_passes_protocol_errors_through(bridge):
    reply = bridge.call("session/state", {"sessionId": "nope"})
    assert reply["error"]["code"] == -32002  # SESSION_NOT_FOUND
    assert bridge.wait_events("nope", 0, 0) is None


@pytest.fixture
def client(bridge):
    with mock.patch.object(agent_bridge, "_bridge", bridge), mock.patch.object(agent_routes, "KEEPALIVE_SECONDS", 0.05):
        yield create_app().test_client()


def test_rpc_route_validates_requests(client, tmp_path):
    assert client.post("/api/agent/rpc", json={"params": {}}).status_code == 400
    assert client.post("/api/agent/rpc", json={"method": "initialize"}).status_code == 400
    missing = client.post("/api/agent/rpc", json={"method": "session/new", "params": {"cwd": str(tmp_path / "x")}})
    assert missing.status_code == 400 and "Not a directory" in missing.get_json()["error"]["message"]
    created = client.post("/api/agent/rpc", json={"method": "session/new", "params": {"cwd": str(tmp_path)}})
    assert created.status_code == 200 and created.get_json()["result"]["cwd"] == str(tmp_path)


def test_events_route_streams_and_asks_unknown_sessions_to_resync(client, bridge, tmp_path):
    response = client.get("/api/agent/sessions/nope/events")
    assert response.mimetype == "text/event-stream"
    assert response.get_data(as_text=True) == "event: resync\ndata: {}\n\n"

    bridge.provider.replies.append(text_reply("hello"))
    session_id = bridge.call("session/new", {"cwd": str(tmp_path)})["result"]["sessionId"]
    bridge.call("run/prompt", {"sessionId": session_id, "content": [{"type": "text", "text": "hi"}]})
    run_to_end(bridge, session_id, 0)

    response = client.get(f"/api/agent/sessions/{session_id}/events", headers={"Last-Event-ID": "1"})
    chunks = iter(response.response)
    first = next(chunks)
    assert first.startswith(b"id: 2\ndata: ")
    response.close()


def test_agent_page_script_is_valid_javascript(client):
    page = client.get("/agent")
    assert page.status_code == 200
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the script check")
    script = page.get_data(as_text=True).rsplit("<script>", 1)[-1].split("</script>", 1)[0]
    result = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
