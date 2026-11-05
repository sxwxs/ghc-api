"""All three native AnySearch contracts; no real credentials or upstream calls."""

import json
from unittest.mock import Mock

import pytest
import requests

from ghc_api import anysearch
from ghc_api.app import create_app, PROTECTED_PATHS
from ghc_api.auth import AuthResult, extract_token
from ghc_api.cache import cache
from ghc_api.counters import counters
from ghc_api.state import state

ID = "7d6f4e91-2a83-4c5b-9f10-6e8a3d27b541"
SECRET = "PRIVATE-CREDENTIAL-NEVER-RETAIN"
EXTRACT = {"code": 0, "message": "success", "request_id": ID,
           "data": {"url": "https://example.com", "title": "", "content": "<p>Clean content</p>"}}
CATALOG = {"code": 0, "message": "success", "request_id": ID, "data": {"domains": [
    {"domain": "finance", "description": "Financial data", "sub_domains": [
        {"sub_domain": "finance.quote", "params": {"type": {
            "description": "Asset type", "required": True, "sort_order": 0}}}]},
    {"domain": "code", "sub_domains": [{"sub_domain": "code.doc"}]},
]}}
PATHS = (anysearch.SEARCH_PATH, anysearch.EXTRACT_PATH, anysearch.SUB_DOMAINS_PATH)


def upstream(body, status=200, headers=None):
    return Mock(status_code=status, content=body if isinstance(body, bytes) else json.dumps(body).encode(),
                headers=headers or {"Content-Type": "application/json", "X-Request-ID": ID})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    for key, value in {"enable_anysearch": True, "anysearch_api_key": "SERVER-KEY",
                       "anysearch_base_url": "https://api.anysearch.com", "anysearch_timeout": 30,
                       "anysearch_extract_timeout": 60, "enable_auth": False,
                       "log_anysearch_requests": True, "save_request_to_file": True}.items():
        monkeypatch.setattr(state, key, value)
    post, get = Mock(return_value=upstream(EXTRACT)), Mock(return_value=upstream(CATALOG))
    monkeypatch.setattr(anysearch.requests, "post", post)
    monkeypatch.setattr(anysearch.requests, "get", get)
    cache.cache.clear()
    counters.reset()
    yield create_app().test_client(), post, get, tmp_path
    cache.cache.clear()
    counters.reset()


def call(client, path, **kwargs):
    if path == anysearch.SUB_DOMAINS_PATH:
        return client.get(path + "?domain=code&domain=finance", **kwargs)
    return client.post(path, data=b'{"url":"https://example.com"}', content_type="application/json", **kwargs)


def audit(root):
    return [json.loads(line) for p in (root / "anysearch").glob("*.jl") for line in p.read_text().splitlines()]


def test_extract_preserves_body_content_and_independent_timeout(env):
    client, post, get, root = env
    raw = b'{ "url": "https://example.com" }'
    response = client.post("/v1/extract", data=raw, content_type="application/json; charset=utf-8",
                           headers={"Content-Encoding": "identity"})
    assert response.status_code == 200 and response.data == post.return_value.content
    assert post.call_args.args == ("https://api.anysearch.com/v1/extract",)
    assert post.call_args.kwargs == {"data": raw, "headers": {
        "Authorization": "Bearer SERVER-KEY", "Content-Type": "application/json; charset=utf-8",
        "Content-Encoding": "identity"}, "timeout": 60, "allow_redirects": False}
    get.assert_not_called()  # Never fetch the user's URL locally.
    post.return_value.close.assert_called_once()
    item = audit(root)[0]
    assert item["url"] == "https://example.com"
    assert item["type"] == "anysearch_extract"
    assert item["result_count"] is None
    assert counters.snapshot() == {"anysearch.extract": 1}


@pytest.mark.parametrize("body,content_type,encoding", [
    (b'{"url":"https://a","url":"https://b"}', "application/json", None),
    (b'{"url":"https://a","extra":true}', "application/json", None),
    (b'{}{}', "application/json", None),
    (b'\xff', "application/json", None),
    (b'{"url":"' + b'x' * 16384 + b'"}', "application/json", None),
    (b'{"url":"http://127.0.0.1/private"}', "application/json", None),
    (b'{"url":"https://a"}', "application/json; charset=iso-8859-1", None),
    (b'compressed bytes', "application/json", "gzip"),
    (b'{}', None, None),
])
def test_extract_strict_validation_belongs_to_upstream(env, body, content_type, encoding):
    client, post, get, _ = env
    post.return_value = upstream({"code": -1, "error_code": "invalid_extract_url"}, 400)
    headers = {"Content-Encoding": encoding} if encoding else {}
    response = client.post("/v1/extract", data=body, content_type=content_type, headers=headers)
    assert response.status_code == 400
    assert response.data == post.return_value.content
    assert post.call_args.kwargs["data"] == body
    sent = post.call_args.kwargs["headers"]
    assert sent.get("Content-Type") == content_type
    assert sent.get("Content-Encoding") == encoding
    assert post.call_args.args[0] == "https://api.anysearch.com/v1/extract"
    get.assert_not_called()


def test_catalog_repeated_domains_and_parameters_are_not_rewritten(env):
    client, post, get, root = env
    query = 'domain=code&domain=finance&domain=code&future=a%2Bb&domains=not-an-alias'
    response = client.get('/v1/sub-domains?' + query)
    assert response.status_code == 200
    assert response.data == get.return_value.content
    assert get.call_args.args == ('https://api.anysearch.com/v1/sub-domains',)
    assert get.call_args.kwargs == {"params": query.encode(), "headers": {"Authorization": "Bearer SERVER-KEY"},
                                    "timeout": 30, "allow_redirects": False}
    post.assert_not_called()
    entry = audit(root)[0]
    assert entry["request_query"] == query
    assert entry["request_body"] is None
    assert entry["uses_search_quota"] is False
    assert entry["type"] == "anysearch_sub_domains"
    assert entry["result_count"] == 2
    assert counters.snapshot() == {"anysearch.sub_domains": 1}
    cached = cache.get_recent_requests(1)[0]
    assert cached["model"] == "anysearch_sub_domains"
    assert cached["uses_search_quota"] is False
    assert cached["input_tokens"] == cached["output_tokens"] == 0
    # requests preserves duplicate parameters when preparing these raw bytes.
    prepared = requests.Request('GET', get.call_args.args[0], params=get.call_args.kwargs['params']).prepare()
    assert prepared.url.count('domain=code') == 2


@pytest.mark.parametrize('query,status,data', [
    ('', 400, {"code": -1, "message": "At least one domain parameter is required."}),
    ('domains=code', 400, {"code": -1, "message": "At least one domain parameter is required."}),
    ('domain=unknown', 200, {"code": 0, "data": {"domains": []}}),
])
def test_catalog_missing_and_unknown_domains_follow_upstream(env, query, status, data):
    client, _, get, _ = env
    get.return_value = upstream(data, status)
    response = client.get('/v1/sub-domains?' + query)
    assert response.status_code == status and response.data == get.return_value.content
    assert get.call_args.kwargs['params'] == query.encode()


@pytest.mark.parametrize('path', PATHS)
def test_all_endpoints_use_local_auth_and_server_key(env, monkeypatch, path):
    client, post, get, _ = env
    assert path in PROTECTED_PATHS
    monkeypatch.setattr(state, 'enable_auth', True)
    assert call(client, path).status_code == 401
    post.assert_not_called(); get.assert_not_called()
    monkeypatch.setattr('ghc_api.app.require_auth', lambda req: AuthResult(user_id='test-user'))
    assert call(client, path, headers={'Authorization': 'Bearer CLIENT-KEY', 'x-api-key': 'OTHER-KEY'}).status_code == 200
    sent = (get if path == anysearch.SUB_DOMAINS_PATH else post).call_args.kwargs['headers']
    assert sent['Authorization'] == 'Bearer SERVER-KEY'
    assert 'x-api-key' not in sent
    assert 'CLIENT-KEY' not in str(cache.get_recent_requests(1))
    assert cache.get_recent_requests(1)[0]['user_id'] == 'test-user'


def test_catalog_never_forwards_or_records_client_query_token(env, capsys, monkeypatch):
    client, _, get, root = env
    monkeypatch.setattr(state, 'enable_auth', True)
    def approved(req):
        assert extract_token(req) == SECRET
        return AuthResult(user_id='query-token-user')
    monkeypatch.setattr('ghc_api.app.require_auth', approved)
    query = f'domain=code&api_key={SECRET}&domain=finance&api%5Fkey={SECRET}&API_KEY={SECRET}'
    response = client.get('/v1/sub-domains?' + query)
    assert response.status_code == 200
    assert get.call_args.kwargs['params'] == b'domain=code&domain=finance'
    sinks = str(audit(root)) + str(cache.get_recent_requests(1)) + capsys.readouterr().out
    sinks += ''.join(p.read_text() for p in root.rglob('*.jl'))
    assert SECRET not in sinks


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('raw', [
    {"code": -1, "message": "username=u\npassword=" + SECRET + "\napi_key=" + SECRET, "request_id": ID},
    b'not JSON: ' + SECRET.encode(),
])
def test_sensitive_402_is_removed_from_all_endpoints_and_sinks(env, capsys, path, raw):
    client, post, get, root = env
    sender = get if path == anysearch.SUB_DOMAINS_PATH else post
    sender.return_value = upstream(raw, 402, {'X-Request-ID': ID, 'Set-Cookie': SECRET, 'X-Sensitive': SECRET})
    response = call(client, path)
    assert response.status_code == 402
    assert response.get_json()['request_id'] == ID
    assert response.headers['Cache-Control'] == 'no-store'
    assert sender.call_count == 1
    sinks = str(response.headers) + response.get_data(as_text=True) + str(cache.get_recent_requests(1))
    sinks += capsys.readouterr().out + ''.join(p.read_text() for p in root.rglob('*.jl'))
    assert SECRET not in sinks
    assert audit(root)[0]['upstream']['status_code'] == 402


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('status,expected', [(401, 503), (403, 503), (307, 502), (422, 422), (429, 429), (502, 502)])
def test_error_mapping_is_shared_without_losing_native_errors(env, path, status, expected):
    client, post, get, _ = env
    sender = get if path == anysearch.SUB_DOMAINS_PATH else post
    sender.return_value = upstream({'code': -1, 'error_code': 'native-error'}, status,
                                   {'Content-Type': 'application/json', 'Retry-After': '30', 'X-Request-ID': ID})
    response = call(client, path)
    assert response.status_code == expected
    if status in (422, 429, 502):
        assert response.data == sender.return_value.content
        assert response.headers['Retry-After'] == '30'
    else:
        assert response.get_json()['request_id'] == ID
    assert sender.call_count == 1


@pytest.mark.parametrize('path', PATHS)
def test_no_key_never_makes_anonymous_calls(env, monkeypatch, path):
    client, post, get, _ = env
    monkeypatch.setattr(state, 'anysearch_api_key', '')
    assert call(client, path).status_code == 503
    post.assert_not_called(); get.assert_not_called()


@pytest.mark.parametrize('path', (anysearch.EXTRACT_PATH, anysearch.SUB_DOMAINS_PATH))
@pytest.mark.parametrize('error,expected', [(requests.Timeout(SECRET), 504), (requests.ConnectionError(SECRET), 502)])
def test_new_endpoints_do_not_retry_or_leak_network_exceptions(env, path, error, expected):
    client, post, get, _ = env
    sender = get if path == anysearch.SUB_DOMAINS_PATH else post
    sender.side_effect = error
    response = call(client, path)
    assert response.status_code == expected
    assert SECRET not in response.get_data(as_text=True)
    assert sender.call_count == 1


@pytest.mark.parametrize('path,body', [
    (anysearch.EXTRACT_PATH, {'code': 0, 'data': {'url': 'u', 'title': ''}}),
    (anysearch.EXTRACT_PATH, {'code': 0, 'data': {'url': 'u', 'title': '', 'content': None}}),
    (anysearch.SUB_DOMAINS_PATH, {'code': 0, 'data': {'domains': {}}}),
    (anysearch.SUB_DOMAINS_PATH, {'code': 0, 'data': {'domains': [None]}}),
    (anysearch.SUB_DOMAINS_PATH, {'code': 0, 'data': {'domains': [{'domain': 'code', 'sub_domains': [None]}]}}),
])
def test_malformed_success_is_recorded_as_error_without_rewriting_native_response(env, path, body):
    client, post, get, root = env
    sender = get if path == anysearch.SUB_DOMAINS_PATH else post
    sender.return_value = upstream(body)
    response = call(client, path)
    assert response.status_code == 200 and response.data == sender.return_value.content
    assert audit(root)[0]['state'] == 'error'
    assert cache.get_recent_requests(1)[0]['state'] == 'error'


@pytest.mark.parametrize('method,path,status', [
    ('GET', '/v1/search', 405), ('GET', '/v1/extract', 405),
    ('POST', '/v1/sub-domains', 405), ('DELETE', '/v1/extract', 405),
    ('GET', '/v1/sub_domains', 404),
])
def test_wrong_paths_and_methods_never_spend_the_key(env, method, path, status):
    client, post, get, _ = env
    assert client.open(path, method=method).status_code == status
    post.assert_not_called()
    get.assert_not_called()


def test_endpoint_allowlist_and_extract_timeout(env):
    _, post, get, _ = env
    with pytest.raises(ValueError): anysearch.endpoint_for(state, '/v1/arbitrary')
    result = anysearch.request_api(b'{}', state, api_path='/v1/arbitrary', local_request_id=ID)
    assert result.status_code == 503
    post.assert_not_called(); get.assert_not_called()
    anysearch.apply_config({'anysearch_extract_timeout': 90}, state)
    anysearch.request_api(b'{}', state, api_path=anysearch.EXTRACT_PATH, local_request_id=ID)
    assert post.call_args.kwargs['timeout'] == 90
    for value in (0, True, '60'):
        with pytest.raises(ValueError): anysearch.apply_config({'anysearch_extract_timeout': value}, state)
