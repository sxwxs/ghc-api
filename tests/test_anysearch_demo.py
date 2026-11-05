"""The CLI uses the local proxy token and never dumps sensitive error bodies."""

from pathlib import Path
import runpy
from unittest.mock import MagicMock, Mock

import pytest
import requests


@pytest.mark.parametrize("status", [200, 402, 429])
def test_demo_response_handling(monkeypatch, capsys, status):
    demo = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "anysearch_search_demo.py"))
    monkeypatch.setattr("sys.argv", ["anysearch_search_demo.py", "query", "--tag", "code.doc"])
    monkeypatch.setenv("GHC_API_TOKEN", "LOCAL-USER-TOKEN")
    response = Mock(status_code=status)
    response.json.return_value = {"code": 0, "data": {"results": []}}
    if status != 200:
        response.json.side_effect = AssertionError("must not inspect sensitive error body")
    context = MagicMock()
    context.__enter__.return_value = response
    post = Mock(return_value=context)
    monkeypatch.setattr(requests, "post", post)
    assert demo["main"]() == (0 if status == 200 else 1)
    assert post.call_args.args == ("http://127.0.0.1:8313/v1/search",)
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer LOCAL-USER-TOKEN"
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"]["tag"] == "code.doc"
    assert "LOCAL-USER-TOKEN" not in capsys.readouterr().out
    if status != 200:
        response.json.assert_not_called()


@pytest.mark.parametrize("mode,endpoint,method,request_args,timeout", [
    (["--extract", "https://example.com/article"], "/v1/extract", "post",
     {"json": {"url": "https://example.com/article"}}, 60),
    (["--domain", "code", "--domain", "finance"], "/v1/sub-domains", "get",
     {"params": [("domain", "code"), ("domain", "finance")]}, 30),
    (["q", "--tag", "code.doc", "--params", '{"library":"golang"}'], "/v1/search", "post",
     {"json": {"query": "q", "tag": "code.doc", "params": {"library": "golang"},
               "max_results": 5, "format": "json"}}, 30),
])
@pytest.mark.parametrize("status", [200, 402])
def test_demo_all_endpoints(monkeypatch, capsys, mode, endpoint, method, request_args, timeout, status):
    demo = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "anysearch_search_demo.py"))
    monkeypatch.setattr("sys.argv", ["anysearch_search_demo.py", *mode])
    monkeypatch.setenv("GHC_API_TOKEN", "LOCAL-USER-TOKEN")
    response = Mock(status_code=status)
    response.json.return_value = {"code": 0, "data": {}}
    if status == 402:
        response.json.side_effect = AssertionError("must not inspect sensitive error body")
    context = MagicMock()
    context.__enter__.return_value = response
    sender, other = Mock(return_value=context), Mock(side_effect=AssertionError("wrong HTTP method"))
    monkeypatch.setattr(requests, method, sender)
    monkeypatch.setattr(requests, "get" if method == "post" else "post", other)
    assert demo["main"]() == (0 if status == 200 else 1)
    assert sender.call_args.args == ("http://127.0.0.1:8313" + endpoint,)
    expected_headers = {"Authorization": "Bearer LOCAL-USER-TOKEN"}
    if method == "post": expected_headers["Content-Type"] = "application/json"
    assert sender.call_args.kwargs == {**request_args, "headers": expected_headers,
                                       "timeout": timeout, "allow_redirects": False}
    other.assert_not_called()
    assert "LOCAL-USER-TOKEN" not in capsys.readouterr().out
    if status == 402: response.json.assert_not_called()


@pytest.mark.parametrize("args", [[], ["q", "--extract", "https://example.com"],
                                  ["q", "--params", "[]"], ["q", "--params", "not-json"],
                                  ["--domain", "code", "--params", "{}"]])
def test_invalid_demo_usage_never_calls_upstream(monkeypatch, args):
    demo = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "anysearch_search_demo.py"))
    monkeypatch.setattr("sys.argv", ["anysearch_search_demo.py", *args])
    post, get = Mock(), Mock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(SystemExit) as exc:
        demo["main"]()
    assert exc.value.code == 2
    post.assert_not_called()
    get.assert_not_called()
