import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.auth import AuthResult
from ghc_api.cache import cache
from ghc_api.proxy.affinity import ProxyAffinityStore
from ghc_api.proxy.auth import ProxyAuthProvider
from ghc_api.proxy.client import ProxyRuntime
from ghc_api.proxy.config import ProxyAuthConfig, ProxyConfigError, ProxyRegistry, parse_proxy_config
from ghc_api.routes import proxy as proxy_routes
from ghc_api.state import state


CONFIG = """
proxies:
  demo-profile:
    auth:
      type: none
    headers:
      X-Profile: profile
    affinity:
      enabled: true
      response_header: X-Route-Token
      request_header: X-Route-Token
      scope: model
      persist: true
    apis:
      responses:
        upstream_url: https://gateway.example.test/responses
        request_model: omit
        response_model: public
        headers:
          X-API: responses
      chat_completions:
        upstream_url: https://gateway.example.test/chat/completions
        request_model: upstream
        response_model: public
        headers:
          X-API: chat
    models:
      demo-model:
        display_name: Demo Model
        reasoning: true
        input: [text]
        context_window: 64000
        max_output_tokens: 4096
        headers:
          X-Upstream-Model: route-name
        apis:
          responses:
            upstream_model: null
          chat_completions:
            upstream_model: chat-deployment
"""


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, lines=None, content=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = headers or {"Content-Type": "application/json"}
        self._payload = payload
        self._lines = lines or []
        self.closed = False
        if content is not None:
            self.content = content
        elif payload is None:
            self.content = b""
        else:
            self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class ConfiguredProxyConfigTest(unittest.TestCase):
    def test_parses_responses_and_chat_completions(self):
        snapshot = parse_proxy_config(__import__("yaml").safe_load(CONFIG))
        profile = snapshot.profiles["demo-profile"]

        self.assertEqual(set(profile.apis), {"responses", "chat_completions"})
        self.assertEqual(profile.apis["responses"].request_model, "omit")
        self.assertEqual(profile.apis["chat_completions"].request_model, "upstream")
        self.assertEqual(
            profile.models["demo-model"].apis["chat_completions"].upstream_model,
            "chat-deployment",
        )

    def test_rejects_upstream_mode_without_upstream_model(self):
        config = __import__("yaml").safe_load(CONFIG)
        config["proxies"]["demo-profile"]["models"]["demo-model"]["apis"]["chat_completions"]["upstream_model"] = None

        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)

    def test_registry_keeps_last_known_good_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proxies.yaml"
            path.write_text(CONFIG, encoding="utf-8")
            registry = ProxyRegistry(path)
            self.assertIsNotNone(registry.get_profile("demo-profile"))

            path.write_text("proxies: [invalid", encoding="utf-8")
            self.assertIsNotNone(registry.get_profile("demo-profile"))
            self.assertIsNotNone(registry.last_error)


class ConfiguredProxyAffinityTest(unittest.TestCase):
    def test_persisted_token_is_available_after_store_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "affinity.json"
            first = ProxyAffinityStore(path)
            first.set("route-key", "route-token", persist=True)

            second = ProxyAffinityStore(path)
            self.assertEqual(second.get("route-key"), "route-token")


class ConfiguredProxyAuthTest(unittest.TestCase):
    def test_command_token_is_cached(self):
        provider = ProxyAuthProvider(ProxyAuthConfig(
            type="bearer_command",
            command=("credential-helper",),
            cache_ttl_seconds=300,
        ))
        completed = mock.Mock(returncode=0, stdout="token-value\n", stderr="")

        with mock.patch("ghc_api.proxy.auth.subprocess.run", return_value=completed) as run:
            self.assertEqual(provider.get_token(), "token-value")
            self.assertEqual(provider.get_token(), "token-value")

        self.assertEqual(run.call_count, 1)

    def test_command_token_is_refreshed_after_upstream_401(self):
        config = __import__("yaml").safe_load(CONFIG)
        config["proxies"]["demo-profile"]["auth"] = {
            "type": "bearer_command",
            "command": ["credential-helper"],
            "cache_ttl_seconds": 300,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proxies.yaml"
            path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
            runtime = ProxyRuntime(
                registry=ProxyRegistry(path),
                affinity_store=ProxyAffinityStore(Path(tmp) / "affinity.json"),
            )
            profile = runtime.registry.get_profile("demo-profile")
            api, model, model_api = profile.resolve("responses", "demo-model")
            unauthorized = FakeResponse(status_code=401, payload={"error": "expired"})
            success = FakeResponse(payload={"id": "resp-1", "output": []})
            command_results = [
                mock.Mock(returncode=0, stdout="token-one\n", stderr=""),
                mock.Mock(returncode=0, stdout="token-two\n", stderr=""),
            ]

            with mock.patch("ghc_api.proxy.auth.subprocess.run", side_effect=command_results) as run, \
                    mock.patch("ghc_api.proxy.client.requests.post", side_effect=[unauthorized, success]) as post:
                result = runtime.post(profile, api, model, model_api, {"model": "demo-model"}, False)

        self.assertIs(result.response, success)
        self.assertTrue(unauthorized.closed)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["headers"]["Authorization"], "Bearer token-one")
        self.assertEqual(post.call_args_list[1].kwargs["headers"]["Authorization"], "Bearer token-two")


class ConfiguredProxyRouteTest(unittest.TestCase):
    def setUp(self):
        self.saved_enable_auth = state.enable_auth
        state.enable_auth = False
        cache.cache.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        config_path = root / "proxies.yaml"
        config_path.write_text(CONFIG, encoding="utf-8")
        self.runtime = ProxyRuntime(
            registry=ProxyRegistry(config_path),
            affinity_store=ProxyAffinityStore(root / "affinity.json"),
        )
        self.runtime_patch = mock.patch.object(proxy_routes, "proxy_runtime", self.runtime)
        self.runtime_patch.start()
        self.app = create_app()

    def tearDown(self):
        self.runtime_patch.stop()
        self.temp_dir.cleanup()
        state.enable_auth = self.saved_enable_auth
        cache.cache.clear()

    def test_responses_proxy_omits_model_rewrites_response_and_reuses_affinity(self):
        first = FakeResponse(
            payload={
                "id": "resp-1",
                "object": "response",
                "model": "private-deployment",
                "output": [],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
            headers={
                "Content-Type": "application/json",
                "X-Route-Token": "route-token",
            },
        )
        second = FakeResponse(
            payload={
                "id": "resp-2",
                "object": "response",
                "model": "private-deployment",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        with mock.patch("ghc_api.proxy.client.requests.post", side_effect=[first, second]) as post, \
                mock.patch("ghc_api.routes.openai.ensure_copilot_token") as copilot_token:
            with self.app.test_client() as client:
                response1 = client.post("/proxy/demo-profile/v1/responses", json={
                    "model": "demo-model",
                    "input": "hello",
                })
                response2 = client.post("/proxy/demo-profile/v1/responses", json={
                    "model": "demo-model",
                    "input": "again",
                })

        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response1.get_json()["model"], "demo-model")
        self.assertEqual(response2.status_code, 200)
        self.assertNotIn("model", post.call_args_list[0].kwargs["json"])
        self.assertEqual(post.call_args_list[0].kwargs["headers"]["X-Upstream-Model"], "route-name")
        self.assertEqual(post.call_args_list[1].kwargs["headers"]["X-Route-Token"], "route-token")
        copilot_token.assert_not_called()

    def test_chat_completions_stream_rewrites_model_and_body_model(self):
        lines = [
            b'data: {"id":"chat-1","model":"private-deployment","choices":[{"index":0,"delta":{"content":"OK"}}]}',
            b'data: {"id":"chat-1","model":"private-deployment","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}',
            b"data: [DONE]",
        ]
        upstream = FakeResponse(
            headers={"Content-Type": "text/event-stream"},
            lines=lines,
            content=b"unused",
        )

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/chat/completions", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                })

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"model":"demo-model"', body)
        self.assertIn("data: [DONE]", body)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "chat-deployment")

    def test_models_are_isolated_to_profile_endpoint(self):
        with self.app.test_client() as client:
            response = client.get("/proxy/demo-profile/v1/models")

        self.assertEqual(response.status_code, 200)
        model = response.get_json()["data"][0]
        self.assertEqual(model["id"], "demo-model")
        self.assertEqual(
            model["supported_endpoints"],
            ["/responses", "/chat/completions"],
        )
        self.assertNotIn("headers", model)
        self.assertNotIn("upstream_url", model)

    def test_empty_success_response_becomes_bad_gateway(self):
        upstream = FakeResponse(payload=None, content=b"")

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/responses", json={
                    "model": "demo-model",
                    "input": "hello",
                })

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["error"]["code"], "empty_upstream_response")

    def test_unknown_profile_does_not_touch_upstream(self):
        with mock.patch("ghc_api.proxy.client.requests.post") as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/missing/v1/responses", json={
                    "model": "demo-model",
                    "input": "hello",
                })

        self.assertEqual(response.status_code, 404)
        post.assert_not_called()

    def test_proxy_auth_is_blueprint_local(self):
        state.enable_auth = True
        denied = AuthResult(
            user_id=None,
            error_code="missing_token",
            error_message="missing",
            http_status=401,
        )
        with mock.patch.object(proxy_routes, "require_auth", return_value=denied) as require:
            with self.app.test_client() as client:
                response = client.get("/proxy/demo-profile/v1/models")

        self.assertEqual(response.status_code, 401)
        require.assert_called_once()
        self.assertNotIn("/proxy/demo-profile/v1/models", PROTECTED_PATHS)


if __name__ == "__main__":
    unittest.main()
