"""AnySearch native contract, key custody, and sensitive-response boundary."""

import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
import requests

from ghc_api import anysearch
from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.auth import AuthResult
from ghc_api.cache import cache
from ghc_api.counters import counters
from ghc_api.state import state

UPSTREAM_ID = "7d6f4e91-2a83-4c5b-9f10-6e8a3d27b541"
LOCAL_ID = "5a3f8c27-1e64-4b90-a752-6d9e2f41c083"
SECRET = "GENERATED-SECRET-NOT-FOR-LOGS"


def upstream(status=200, body=None, headers=None):
    if body is None:
        body = {"code": 0, "message": "success", "request_id": UPSTREAM_ID,
                "data": {"results": [{"title": "", "url": "https://example.com", "snippet": "summary"}],
                         "metadata": {"total_results": 1, "search_time_ms": 312}}}
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return Mock(status_code=status, content=raw, headers=headers or {
        "Content-Type": "application/json", "X-Request-ID": UPSTREAM_ID,
    })


def settings(**kwargs):
    return SimpleNamespace(**{
        "enable_anysearch": True, "anysearch_api_key": "SERVER-KEY",
        "anysearch_base_url": anysearch.API_BASE_URL, "anysearch_timeout": 30,
        "log_anysearch_requests": True, **kwargs,
    })


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    for key, value in vars(settings()).items():
        monkeypatch.setattr(state, key, value)
    monkeypatch.setattr(state, "enable_auth", False)
    monkeypatch.setattr(state, "save_request_to_file", True)
    post = Mock(return_value=upstream())
    monkeypatch.setattr(anysearch.requests, "post", post)
    cache.cache.clear()
    counters.reset()
    yield create_app().test_client(), post, tmp_path
    cache.cache.clear()
    counters.reset()


def audit(directory):
    return [json.loads(line) for file in (directory / "anysearch").glob("*.jl")
            for line in file.read_text().splitlines()]


def test_native_body_and_response_are_unchanged(env):
    client, post, directory = env
    raw = b'{ "query":"q", "max_results":10, "tag":"code.doc", "params":{"library":"go"}, "future":true }'
    response = client.post("/v1/search", data=raw, content_type="application/json")
    assert response.status_code == 200
    assert response.data == post.return_value.content
    assert response.headers["X-Request-ID"] == UPSTREAM_ID
    assert post.call_args.args == (anysearch.API_BASE_URL + "/v1/search",)
    assert post.call_args.kwargs == {
        "data": raw, "headers": {"Authorization": "Bearer SERVER-KEY", "Content-Type": "application/json"},
        "timeout": 30, "allow_redirects": False,
    }
    post.return_value.close.assert_called_once()
    entry = audit(directory)[0]
    assert entry["result_count"] == 1
    assert entry["upstream"] == {"status_code": 200, "request_id": UPSTREAM_ID}
    assert entry["state"] == "completed"
    cached = cache.get_recent_requests(1)[0]
    assert cached["model"] == "anysearch_search"
    assert cached["input_tokens"] == cached["output_tokens"] == 0
    assert counters.snapshot()["anysearch.search"] == 1


@pytest.mark.parametrize("status", [400, 415, 429, 502])
def test_ordinary_errors_and_retry_header_survive(env, status):
    client, post, _ = env
    post.return_value = upstream(status, {"code": -1, "message": "upstream error"}, {
        "Content-Type": "application/json", "Retry-After": "30",
        "Connection": "keep-alive", "Content-Encoding": "gzip", "Content-Length": "9999",
    })
    response = client.post("/v1/search", data=b"malformed", content_type="application/json")
    assert response.status_code == status
    assert response.data == post.return_value.content
    assert response.headers["Retry-After"] == "30"
    assert "Connection" not in response.headers
    assert "Content-Encoding" not in response.headers
    assert response.headers["Content-Length"] != "9999"
    assert post.call_count == 1
    assert counters.snapshot()["anysearch.search_error"] == 1


def test_missing_content_type_is_not_silently_fixed(env):
    client, post, _ = env
    post.return_value = upstream(415)
    assert client.post("/v1/search", data=b'{}').status_code == 415
    assert "Content-Type" not in post.call_args.kwargs["headers"]


@pytest.mark.parametrize("raw", [
    json.dumps({"code": -1, "message": "username=account\npassword=" + SECRET + "\napi_key=" + SECRET,
                "request_id": UPSTREAM_ID, "extra": SECRET}).encode(),
    ("<html>password=" + SECRET + "</html>").encode(),
    ("{invalid-json:" + SECRET).encode(),
    b"[" * 1200 + SECRET.encode() + b"]" * 1200,
])
def test_402_never_leaks_to_any_sink(env, capsys, raw):
    client, post, directory = env
    post.return_value = upstream(402, raw, {
        "X-Request-ID": UPSTREAM_ID, "Content-Type": "text/plain", "X-Secret": SECRET,
        "Set-Cookie": SECRET, "Location": "https://example.com/" + SECRET,
    })
    response = client.post("/v1/search", json={"query": "q"})
    assert response.status_code == 402
    assert response.get_json()["code"] == -1
    assert response.get_json()["request_id"] == UPSTREAM_ID
    assert response.headers["Cache-Control"] == "no-store"
    assert post.call_count == 1
    sinks = [response.get_data(as_text=True), str(response.headers), str(cache.get_recent_requests(10)),
             client.get("/api/requests").get_data(as_text=True), capsys.readouterr().out]
    sinks += [p.read_text() for p in directory.rglob("*.jl")]
    assert all(SECRET not in value for value in sinks)
    assert audit(directory)[0]["upstream"]["status_code"] == 402
    assert list((directory / "requests").glob("*.jl")), "exercise shared persisted cache too"


@pytest.mark.parametrize("status,expected", [(401, 503), (403, 503), (302, 502), (307, 502)])
def test_credential_errors_and_redirects_are_safe(env, status, expected):
    client, post, directory = env
    post.return_value = upstream(status, {"message": SECRET, "request_id": UPSTREAM_ID}, {
        "X-Request-ID": UPSTREAM_ID, "Location": "https://example.com/" + SECRET,
    })
    response = client.post("/v1/search", json={"query": "q"})
    assert response.status_code == expected
    assert SECRET not in response.get_data(as_text=True)
    assert "Location" not in response.headers
    assert audit(directory)[0]["upstream"]["status_code"] == status
    assert post.call_count == 1


def test_invalid_upstream_request_id_is_not_a_secret_escape_hatch(env):
    client, post, _ = env
    post.return_value = upstream(402, {"request_id": SECRET}, {"X-Request-ID": SECRET})
    response = client.post("/v1/search", json={"query": "q"})
    assert UUID(response.get_json()["request_id"]).version == 4
    assert SECRET not in response.get_data(as_text=True)


@pytest.mark.parametrize("disabled", ["enable_anysearch", "anysearch_api_key"])
def test_key_is_required_no_anonymous_fallback(env, monkeypatch, disabled):
    client, post, _ = env
    monkeypatch.setattr(state, disabled, False if disabled == "enable_anysearch" else "")
    assert client.post("/v1/search", json={"query": "q"}).status_code == 503
    post.assert_not_called()


@pytest.mark.parametrize("exc,expected", [
    (requests.Timeout(SECRET), 504), (requests.ConnectionError(SECRET), 502),
    (requests.RequestException(SECRET), 502),
])
def test_network_errors_never_echo_exceptions_or_retry(env, capsys, exc, expected):
    client, post, directory = env
    post.side_effect = exc
    response = client.post("/v1/search", json={"query": "q"})
    assert response.status_code == expected
    assert post.call_count == 1
    assert SECRET not in response.get_data(as_text=True) + str(audit(directory)) + capsys.readouterr().out


def test_local_auth_and_server_key_are_separate(env, monkeypatch):
    client, post, directory = env
    monkeypatch.setattr(state, "enable_auth", True)
    assert "/v1/search" in PROTECTED_PATHS
    assert client.post("/v1/search", json={"query": "q"}).status_code == 401
    post.assert_not_called()
    monkeypatch.setattr("ghc_api.app.require_auth", lambda req: AuthResult(user_id="test-user"))
    response = client.post("/v1/search", json={"query": "q"}, headers={
        "Authorization": "Bearer CLIENT-TOKEN", "x-api-key": "CLIENT-KEY",
        "x-apikey": "OTHER-CLIENT-KEY", "Cookie": "secret-cookie",
    })
    assert response.status_code == 200
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer SERVER-KEY"
    assert "x-api-key" not in post.call_args.kwargs["headers"]
    assert audit(directory)[0]["user_id"] == "test-user"
    recorded = str(cache.get_recent_requests(1)) + str(audit(directory))
    for secret in ("SERVER-KEY", "CLIENT-TOKEN", "CLIENT-KEY", "OTHER-CLIENT-KEY"):
        assert secret not in recorded


@pytest.mark.parametrize("body,success", [
    ({"code": 0, "data": {"results": []}}, True),
    ({"code": -1, "data": {"results": []}}, False),
    ({"code": False, "data": {"results": []}}, False),
    ({"code": 0, "data": {"results": [None]}}, False),
    ({"code": 0, "data": {"results": [{"title": "", "url": "u", "content": None}]}}, False),
    ({"code": 0, "data": None}, False),
    (b"not JSON", False),
])
def test_business_and_malformed_responses_are_not_counted_as_success(env, body, success):
    client, post, directory = env
    post.return_value = upstream(200, body)
    response = client.post("/v1/search", json={"query": "q"})
    assert response.data == post.return_value.content, "native API remains transparent"
    assert audit(directory)[0]["state"] == ("completed" if success else "error")
    assert cache.get_recent_requests(1)[0]["state"] == ("completed" if success else "error")
    assert cache.get_recent_requests(1)[0]["upstream_request_id"] == UPSTREAM_ID
    assert counters.snapshot()["anysearch.search" if success else "anysearch.search_error"] == 1


def test_runtime_log_switch_and_no_key_exposure(env):
    client, _, directory = env
    before = client.get("/api/runtime-config").get_json()
    assert before["log_anysearch_requests"] is True
    assert "anysearch_api_key" not in before
    assert "SERVER-KEY" not in client.get("/chat").get_data(as_text=True)
    assert client.post("/api/runtime-config", json={"log_anysearch_requests": "yes"}).status_code == 400
    assert client.post("/api/runtime-config", json={"log_anysearch_requests": False}).status_code == 200
    assert client.post("/v1/search", json={"query": "q"}).status_code == 200
    assert not audit(directory)
    assert cache.get_recent_requests(1)[0]["model"] == "anysearch_search"


def test_log_failure_does_not_break_search(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setattr("ghc_api.routes.anysearch.record_search_to_file", Mock(side_effect=OSError(SECRET)))
    assert client.post("/v1/search", json={"query": "q"}).status_code == 200


def test_config_private_file_and_base_prefix(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-key\n")
    target = settings()
    anysearch.apply_config({"anysearch_api_key_file": str(key_file),
                           "anysearch_base_url": "http://127.0.0.1:9000/root/"}, target)
    assert target.anysearch_api_key == "test-key"
    assert anysearch.endpoint_for(target) == "http://127.0.0.1:9000/root/v1/search"


@pytest.mark.parametrize("config", [
    {"enable_anysearch": "false"}, {"log_anysearch_requests": 1},
    {"anysearch_timeout": 0}, {"anysearch_timeout": True}, {"anysearch_timeout": 2.5},
    {"anysearch_base_url": "https://user:secret@example.com"},
    {"anysearch_base_url": "https://example.com?api_key=secret"},
    {"anysearch_base_url": "file:///tmp/search"}, {"anysearch_base_url": 123},
    {"anysearch_api_key": "has\na-newline"},
    {"anysearch_api_key": False}, {"anysearch_api_key_file": 0},
    {"anysearch_api_key": "key", "anysearch_api_key_file": "somewhere"},
    {"anysearch_api_key_file": "/no-such-anysearch-key-file"},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        anysearch.apply_config(config, settings())


def test_generated_config_has_safe_defaults(tmp_path, monkeypatch):
    import yaml
    from ghc_api.generate_config import generate_config_file

    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    generate_config_file()
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["enable_anysearch"] is False
    assert config["anysearch_api_key"] == config["anysearch_api_key_file"] == ""
    target = settings()
    anysearch.apply_config(config, target)
    assert not anysearch.is_configured(target)


def test_search_schema_supports_discovered_capability_parameters():
    chat = anysearch.tool_definition()
    responses = anysearch.tool_definition("responses")
    assert chat["function"]["name"] == responses["name"] == "anysearch_search"
    assert chat["function"]["strict"] is False
    assert set(responses["parameters"]["properties"]) == {
        "query", "max_results", "tag", "params", "zone", "language", "format",
    }
    assert responses["parameters"]["required"] == ["query"]
    assert responses["parameters"]["properties"]["params"].get("additionalProperties", True) is True
    tools = anysearch.tool_definitions("responses")
    assert all(tool["strict"] is False for tool in tools)
    assert [t["name"] for t in tools] == ["anysearch_search", "anysearch_extract", "anysearch_sub_domains"]
    assert tools[1]["parameters"]["required"] == ["url"]
    assert tools[2]["parameters"]["properties"]["domain"]["type"] == "array"
    assert "NOT DNS" in tools[2]["description"]
