"""Images contract tests; no model service or real config directory is used."""
import io
import json
from unittest import mock

import pytest
import requests
from werkzeug.datastructures import MultiDict

from ghc_api.api_helpers import apply_configured_chat_completions_support
from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.config import chat_completions_model_support
from ghc_api.routes.images import MAX_IMAGE_REQUEST_BYTES
from ghc_api.state import state

MODEL = "gpt-image-2.5-flare"
PATHS = ("/v1/images/generations", "/images/generations", "/v1/images/edits", "/images/edits")
CATALOG = {"data": [
    {"id": MODEL, "capabilities": {"type": "image"},
     "supported_endpoints": ["/v1/images/generations", "/v1/images/edits"]},
    {"id": "image-bare", "supported_endpoints": ["/images/generations"]},
    {"id": "gpt-text", "capabilities": {"type": "chat"}, "supported_endpoints": ["/responses"]},
]}
BODY = {"model": MODEL, "prompt": "A blue circle", "quality": "low", "n": 1}
RESULT = {
    "created": 123, "data": [{"b64_json": "aW1hZ2U="}], "output_format": "png",
    "usage": {"input_tokens": 30, "output_tokens": 196,
              "input_tokens_details": {"text_tokens": 20, "image_tokens": 10, "cached_tokens": 5}},
    "copilot_usage": {"total_nano_aiu": 123},
}


def upstream(body=RESULT, status=200, content_type="application/json"):
    response = requests.Response()
    response.status_code = status
    response._content = (json.dumps(body) if isinstance(body, (dict, list)) else body).encode()
    response.headers["Content-Type"] = content_type
    response.close = mock.Mock()
    return response


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(state, "models", CATALOG)
    monkeypatch.setattr(state, "enable_auth", False)
    with mock.patch("ghc_api.routes.images.ensure_copilot_token"), \
         mock.patch("ghc_api.routes.images.get_copilot_headers", side_effect=lambda: {"Content-Type": "application/json"}), \
         mock.patch("ghc_api.routes.images.cache") as cache, \
         mock.patch("ghc_api.routes.images.requests.post") as post:
        post.return_value = upstream()
        yield create_app().test_client(), post, cache


@pytest.mark.parametrize("path", PATHS[:2])
def test_generate_passthrough_and_usage(api, path):
    client, post, cache = api
    response = client.post(path, json=BODY, headers={"Authorization": "Bearer private"})
    assert response.status_code == 200
    assert response.data == post.return_value.content
    assert response.get_json() == RESULT
    assert post.call_args.args[0].endswith("/v1/images/generations")
    assert post.call_args.kwargs["json"] == BODY
    assert post.call_args.kwargs["headers"]["X-Initiator"] == "user"
    record = cache.complete_request.call_args.args[1]
    # input_tokens is the uncached slice: 30 total - 5 cached reads.
    assert record["input_tokens"] == 25
    assert record["output_tokens"] == 196
    assert record["cache_read_input_tokens"] == 5
    assert record["request_headers"]["Authorization"] == "***REDACTED***"
    assert record["response_body"]["data"][0]["b64_json"]["_omitted"]
    assert record["response_body"]["copilot_usage"] == RESULT["copilot_usage"]
    post.return_value.close.assert_called_once()


@pytest.mark.parametrize("path", PATHS[2:])
def test_multipart_multiple_images_mask_and_repeated_fields(api, path):
    client, post, cache = api
    captured = {}
    def send(*args, **kwargs):
        captured.update(kwargs)
        captured["file_contents"] = [f[1][1].read() for f in kwargs["files"]]
        return upstream()
    post.side_effect = send
    form = MultiDict([
        ("model", MODEL), ("prompt", "Make it red"), ("quality", "low"),
        ("image[]", (io.BytesIO(b"image-one"), "one.png", "image/png")),
        ("image[]", (io.BytesIO(b"image-two"), "two.webp", "image/webp")),
        ("mask", (io.BytesIO(b"mask"), "mask.png", "image/png")),
        ("extra", "one"), ("extra", "two"),
    ])
    response = client.post(path, data=form)
    assert response.status_code == 200
    assert "Content-Type" not in captured["headers"]
    assert captured["file_contents"] == [b"image-one", b"image-two", b"mask"]
    assert [f[0] for f in captured["files"]] == ["image[]", "image[]", "mask"]
    assert ("extra", "one") in captured["data"] and ("extra", "two") in captured["data"]
    record = cache.complete_request.call_args.args[1]
    assert record["endpoint"] == "/v1/images/edits"
    assert record["request_body"]["_files"][0]["filename"] == "one.png"
    assert "image-one" not in json.dumps(record)


def test_json_edits_pass_through_without_logging_data_urls(api):
    client, post, cache = api
    body = {**BODY, "images": [{"image_url": "data:image/png;base64,c2VjcmV0"}]}
    response = client.post("/v1/images/edits", json=body)
    assert response.status_code == 200
    assert post.call_args.kwargs["json"] == body
    assert "c2VjcmV0" not in json.dumps(cache.start_request.call_args.args[1])


def test_model_mapping_and_advertised_bare_endpoint(api):
    client, post, _ = api
    with mock.patch("ghc_api.routes.images.translate_model_name", return_value="image-bare"):
        response = client.post(PATHS[0], json=BODY)
    assert response.status_code == 200
    assert post.call_args.args[0].endswith("/images/generations")
    assert not post.call_args.args[0].endswith("/v1/images/generations")
    assert post.call_args.kwargs["json"]["model"] == "image-bare"


@pytest.mark.parametrize("body,code", [
    ([], "invalid_json"), ({"prompt": "x"}, "missing_model"),
    ({"model": MODEL}, "invalid_prompt"), ({**BODY, "prompt": "  "}, "invalid_prompt"),
    ({**BODY, "stream": True}, "unsupported_stream"),
    ({**BODY, "model": "gpt-text"}, "unsupported_model"),
    ({**BODY, "model": "unknown"}, "unsupported_model"),
])
def test_invalid_requests_do_not_spend_quota(api, body, code):
    client, post, _ = api
    response = client.post(PATHS[0], json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == code
    post.assert_not_called()


@pytest.mark.parametrize("stream", [False, None, "false", "0"])
def test_false_stream_values_return_json_and_are_not_forwarded(api, stream):
    client, post, _ = api
    response = client.post(PATHS[0], json={**BODY, "stream": stream})
    assert response.status_code == 200
    assert response.get_json() == RESULT
    assert "stream" not in post.call_args.kwargs["json"]


def test_missing_edit_file_and_duplicate_model_are_rejected(api):
    client, post, _ = api
    response = client.post(PATHS[2], data={"model": MODEL, "prompt": "x"}, content_type="multipart/form-data")
    assert response.get_json()["error"]["code"] == "missing_image"
    form = MultiDict([("model", MODEL), ("model", "other"), ("prompt", "x")])
    response = client.post(PATHS[2], data=form, content_type="multipart/form-data")
    assert response.get_json()["error"]["code"] == "invalid_form"
    post.assert_not_called()


@pytest.mark.parametrize("status,body,content_type", [
    (403, {"error": {"message": "Disabled by organization"}}, "application/json"),
    (422, {"detail": "Invalid image size"}, "application/json"),
    (502, "upstream unavailable", "text/plain"),
])
def test_upstream_errors_are_preserved_and_recorded(api, status, body, content_type):
    client, post, cache = api
    post.return_value = upstream(body, status, content_type)
    response = client.post(PATHS[0], json=BODY)
    assert response.status_code == status
    assert response.data == post.return_value.content
    assert response.content_type == content_type
    assert cache.complete_request.call_args.args[1]["status_code"] == status


@pytest.mark.parametrize("failure,status", [(requests.ReadTimeout(), 504), (requests.ConnectionError(), 502)])
def test_no_retries_after_ambiguous_network_failure(api, monkeypatch, failure, status):
    client, post, cache = api
    monkeypatch.setattr(state, "max_connection_retries", 5)
    post.side_effect = failure
    response = client.post(PATHS[0], json=BODY)
    assert response.status_code == status
    assert "not retried" in response.get_json()["error"]["message"]
    post.assert_called_once()
    assert cache.complete_request.call_args.args[1]["status_code"] == status


@pytest.mark.parametrize("path", PATHS)
def test_auth_precedes_parsing_and_upstream_calls(api, monkeypatch, path):
    client, post, _ = api
    monkeypatch.setattr(state, "enable_auth", True)
    assert path in PROTECTED_PATHS
    response = client.post(path, data="not JSON", content_type="application/json")
    assert response.status_code == 401
    post.assert_not_called()


def test_upload_limit_checked_before_parsing(api):
    client, post, _ = api
    response = client.post(PATHS[0], data="{}", content_type="application/json",
                           environ_overrides={"CONTENT_LENGTH": str(MAX_IMAGE_REQUEST_BYTES + 1)})
    assert response.status_code == 413
    post.assert_not_called()


def test_chunked_upload_limit_is_enforced_without_content_length(api, monkeypatch):
    client, post, _ = api
    monkeypatch.setattr("ghc_api.app.MAX_IMAGE_REQUEST_BYTES", 32)
    response = client.post(PATHS[0], data=json.dumps(BODY), content_type="application/json",
                           environ_overrides={"CONTENT_LENGTH": "", "wsgi.input_terminated": True})
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "request_too_large"
    post.assert_not_called()


def test_approved_user_is_attributed_to_request(api, monkeypatch):
    from types import SimpleNamespace
    client, _, cache = api
    monkeypatch.setattr(state, "enable_auth", True)
    monkeypatch.setattr("ghc_api.app.require_auth", lambda request: SimpleNamespace(user_id="alice"))
    assert client.post(PATHS[0], json=BODY).status_code == 200
    assert cache.complete_request.call_args.args[1]["user_id"] == "alice"


def test_catalog_fetch_failure_is_502(api, monkeypatch):
    client, post, _ = api
    monkeypatch.setattr(state, "models", None)
    with mock.patch("ghc_api.routes.images.fetch_models", return_value=False):
        response = client.post(PATHS[0], json=BODY)
    assert response.status_code == 502
    post.assert_not_called()


def test_preflight_upstream_failure_is_502(api):
    client, post, _ = api
    with mock.patch("ghc_api.routes.images.ensure_copilot_token", side_effect=RuntimeError("token unavailable")):
        response = client.post(PATHS[0], json=BODY)
    assert response.status_code == 502
    assert response.get_json()["error"]["code"] == "upstream_auth_unavailable"
    post.assert_not_called()


def test_non_chat_models_are_not_given_chat_compatibility(monkeypatch):
    monkeypatch.setattr(chat_completions_model_support, "prefix_model_names", ["gpt-"])
    monkeypatch.setattr(chat_completions_model_support, "exact_model_names", [])
    catalog = json.loads(json.dumps(CATALOG))
    catalog["data"].extend([
        {"id": "gpt-embedding", "capabilities": {"type": "embeddings"}},
        {"id": "gpt-decision", "capabilities": {"type": "decisions"}},
        {"id": "gpt-image-untyped", "supported_endpoints": ["/v1/images/generations"]},
    ])
    assert apply_configured_chat_completions_support(catalog) == 1
    for model in catalog["data"]:
        assert ("/v1/chat/completions" in model.get("supported_endpoints", [])) == (model["id"] == "gpt-text")
    # Runtime reapplication removes only an override we previously introduced.
    catalog["data"][2]["capabilities"]["type"] = "image"
    assert apply_configured_chat_completions_support(catalog) == 1
    assert "/v1/chat/completions" not in catalog["data"][2]["supported_endpoints"]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/chat/completions", "/v1/messages", "/v1/responses"])
def test_image_model_is_rejected_by_text_apis(api, path):
    client, post, _ = api
    with mock.patch("ghc_api.routes.openai.ensure_copilot_token"), \
         mock.patch("ghc_api.routes.anthropic.ensure_copilot_token"):
        response = client.post(path, json={"model": MODEL, "messages": [{"role": "user", "content": "draw"}]})
    assert response.status_code == 400
    post.assert_not_called()
