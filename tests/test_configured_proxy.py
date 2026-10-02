import json
import tempfile
import unittest
import uuid
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
      messages:
        upstream_url: https://gateway.example.test/messages
        request_model: upstream
        response_model: public
        headers:
          X-API: messages
          anthropic-version: "2023-06-01"
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
          messages:
            upstream_model: claude-deployment
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
    def test_parses_responses_chat_completions_and_messages(self):
        snapshot = parse_proxy_config(__import__("yaml").safe_load(CONFIG))
        profile = snapshot.profiles["demo-profile"]

        self.assertEqual(set(profile.apis), {"responses", "chat_completions", "messages"})
        self.assertEqual(profile.apis["responses"].request_model, "omit")
        self.assertEqual(profile.apis["chat_completions"].request_model, "upstream")
        self.assertEqual(
            profile.models["demo-model"].apis["chat_completions"].upstream_model,
            "chat-deployment",
        )
        self.assertEqual(
            profile.models["demo-model"].apis["messages"].upstream_model,
            "claude-deployment",
        )

    def test_header_bindings_are_opt_in_and_reject_credential_names(self):
        config = __import__("yaml").safe_load(CONFIG)
        api = config["proxies"]["demo-profile"]["apis"]["messages"]
        self.assertEqual(parse_proxy_config(config).profiles["demo-profile"].apis["messages"].header_bindings, ())
        binding = {"client_name": "X-Test-Client-Context", "upstream_name": "X-Test-Upstream-Context",
                   "value_type": "uuid_v4"}
        api["header_bindings"] = [binding]
        parsed = parse_proxy_config(config).profiles["demo-profile"].apis["messages"].header_bindings[0]
        self.assertEqual((parsed.client_name, parsed.upstream_name, parsed.value_type), tuple(binding.values()))
        for field in ("client_name", "upstream_name"):
            for invalid in ("Authorization", "x-api-key", "api-key", "X-ApiKey", "x-auth-token",
                            "x-access-token", "Tenant-Api-Key", "Vendor-Secret", "cookie",
                            "Ocp-Apim-Subscription-Key", "Content-Type", "Connection",
                            "bad header", "X-Bad\nHeader", ""):
                with self.subTest(field=field, invalid=invalid):
                    binding[field] = invalid
                    with self.assertRaises(ProxyConfigError):
                        parse_proxy_config(config)
            binding[field] = getattr(parsed, field)
        binding["client_name"] = parsed.upstream_name.lower()
        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)
        binding["client_name"] = parsed.client_name
        binding["value_type"] = "arbitrary"
        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)
        binding["value_type"] = "uuid_v4"
        api["header_bindings"].append(dict(binding))
        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)

    def test_rejects_header_binding_that_shadows_the_affinity_header(self):
        for field in ("client_name", "upstream_name"):
            for affinity_header in ("X-Route-Token", "x-route-token"):
                with self.subTest(field=field, affinity_header=affinity_header):
                    config = __import__("yaml").safe_load(CONFIG)
                    binding = {"client_name": "X-Test-Client-Context",
                               "upstream_name": "X-Test-Upstream-Context",
                               "value_type": "uuid_v4"}
                    binding[field] = affinity_header
                    config["proxies"]["demo-profile"]["apis"]["messages"]["header_bindings"] = [binding]
                    with self.assertRaises(ProxyConfigError):
                        parse_proxy_config(config)

    def test_allows_affinity_header_name_when_affinity_is_disabled(self):
        config = __import__("yaml").safe_load(CONFIG)
        config["proxies"]["demo-profile"]["affinity"]["enabled"] = False
        config["proxies"]["demo-profile"]["apis"]["messages"]["header_bindings"] = [{
            "client_name": "X-Client-Route", "upstream_name": "X-Route-Token", "value_type": "uuid_v4",
        }]
        parsed = parse_proxy_config(config).profiles["demo-profile"].apis["messages"].header_bindings[0]
        self.assertEqual(parsed.upstream_name, "X-Route-Token")

    def test_rejects_upstream_mode_without_upstream_model(self):
        config = __import__("yaml").safe_load(CONFIG)
        config["proxies"]["demo-profile"]["models"]["demo-model"]["apis"]["chat_completions"]["upstream_model"] = None

        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)

    def test_rejects_non_boolean_flags(self):
        paths = [
            ("enabled",),
            ("affinity", "enabled"),
            ("affinity", "persist"),
            ("apis", "responses", "enabled"),
            ("apis", "messages", "enabled"),
            ("models", "demo-model", "reasoning"),
            ("models", "demo-model", "apis", "responses", "enabled"),
            ("models", "demo-model", "apis", "messages", "enabled"),
        ]
        for path in paths:
            with self.subTest(path=path):
                config = __import__("yaml").safe_load(CONFIG)
                target = config["proxies"]["demo-profile"]
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = "false"
                with self.assertRaises(ProxyConfigError):
                    parse_proxy_config(config)

    def test_header_names_are_trimmed_and_duplicates_rejected(self):
        config = __import__("yaml").safe_load(CONFIG)
        config["proxies"]["demo-profile"]["headers"] = {" X-Profile ": "profile"}
        profile = parse_proxy_config(config).profiles["demo-profile"]
        self.assertEqual(profile.headers, {"X-Profile": "profile"})

        config["proxies"]["demo-profile"]["headers"] = {
            "X-Profile": "one",
            " X-Profile ": "two",
        }
        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(config)

    def test_unspecified_model_apis_remain_enabled_by_default(self):
        config = __import__("yaml").safe_load(CONFIG)
        profile = config["proxies"]["demo-profile"]
        profile["apis"]["chat_completions"]["request_model"] = "preserve"
        profile["apis"]["messages"]["request_model"] = "preserve"
        profile["models"]["demo-model"]["apis"] = {
            "responses": {"upstream_model": None},
        }

        parsed = parse_proxy_config(config).profiles["demo-profile"]
        self.assertEqual(
            set(parsed.models["demo-model"].apis),
            {"responses", "chat_completions", "messages"},
        )

    def test_registry_keeps_last_known_good_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proxies.yaml"
            path.write_text(CONFIG, encoding="utf-8")
            registry = ProxyRegistry(path)
            self.assertIsNotNone(registry.get_profile("demo-profile"))

            path.write_text("proxies: [invalid", encoding="utf-8")
            self.assertIsNotNone(registry.get_profile("demo-profile"))
            self.assertIsNotNone(registry.last_error)

    def test_registry_disables_profiles_when_config_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proxies.yaml"
            path.write_text(CONFIG, encoding="utf-8")
            registry = ProxyRegistry(path)
            self.assertIsNotNone(registry.get_profile("demo-profile"))

            path.unlink()
            self.assertIsNone(registry.get_profile("demo-profile"))
            self.assertIsNone(registry.last_error)


class ConfiguredProxyAffinityTest(unittest.TestCase):
    def test_persisted_token_is_available_after_store_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "affinity.json"
            first = ProxyAffinityStore(path)
            first.set("route-key", "route-token", persist=True)

            second = ProxyAffinityStore(path)
            self.assertEqual(second.get("route-key"), "route-token")

    def test_affinity_header_survives_a_conflicting_dynamic_binding(self):
        # Config validation rejects this overlap; a stale last-known-good config
        # must still send the affinity token rather than a per-request UUID.
        profile = parse_proxy_config(__import__("yaml").safe_load(CONFIG)).profiles["demo-profile"]
        api, model, model_api = profile.resolve("messages", "demo-model")
        runtime = ProxyRuntime(registry=ProxyRegistry(Path("missing-config.yaml")))

        headers = runtime._build_headers(
            profile, api, model, model_api, ProxyAuthProvider(profile.auth),
            "route-token", {"x-route-token": str(uuid.uuid4())},
        )

        self.assertEqual([name for name in headers if name.lower() == "x-route-token"], ["X-Route-Token"])
        self.assertEqual(headers["X-Route-Token"], "route-token")


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
        config["proxies"]["demo-profile"]["apis"]["responses"]["header_bindings"] = [{
            "client_name": "X-Test-Client-Context", "upstream_name": "X-Test-Upstream-Context",
            "value_type": "uuid_v4",
        }]
        session_id = str(uuid.uuid4())
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
                result = runtime.post(profile, api, model, model_api, {"model": "demo-model"}, False,
                                      dynamic_headers={"X-Test-Upstream-Context": session_id})

        self.assertIs(result.response, success)
        self.assertTrue(unauthorized.closed)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["headers"]["Authorization"], "Bearer token-one")
        self.assertEqual(post.call_args_list[1].kwargs["headers"]["Authorization"], "Bearer token-two")
        self.assertEqual(
            [call.kwargs["headers"]["X-Test-Upstream-Context"] for call in post.call_args_list],
            [session_id, session_id],
        )


class ConfiguredProxyRouteTest(unittest.TestCase):
    @staticmethod
    def _reset_request_cache():
        with cache.lock:
            cache.cache.clear()
            cache.request_count = 0
            cache.bytes_sent = 0
            cache.bytes_received = 0
            cache.model_stats.clear()
            cache.endpoint_stats.clear()
            cache.user_model_stats.clear()
            cache.user_endpoint_stats.clear()
            cache.user_totals.clear()

    def setUp(self):
        self.saved_enable_auth = state.enable_auth
        state.enable_auth = False
        self._reset_request_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.root = root
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
        self._reset_request_cache()

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
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["translated_model"], "chat-deployment")
        self.assertEqual(cached["input_tokens"], 2)
        self.assertEqual(cached["output_tokens"], 1)
        stats = cache.get_stats()
        self.assertEqual(stats["model_stats"]["chat-deployment"]["input_tokens"], 2)
        self.assertEqual(stats["model_stats"]["chat-deployment"]["output_tokens"], 1)
        self.assertEqual(
            stats["endpoint_stats"]["/proxy/demo-profile/v1/chat/completions"]["request_count"],
            1,
        )
        self.assertEqual(cached["upstream_provider"], "configured_proxy")
        self.assertEqual(cached["upstream_profile"], "demo-profile")
        self.assertEqual(cached["upstream_api"], "chat_completions")
        self.assertEqual(
            cache.get_user_model_token_snapshot()[("anonymous", "chat-deployment")]["request_count"],
            1,
        )

    def test_messages_proxy_passes_anthropic_request_and_non_stream_response(self):
        upstream_payload = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-deployment",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 1,
            },
        }
        upstream = FakeResponse(payload=upstream_payload)

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/messages", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 128,
                })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model"], "demo-model")
        self.assertNotIn("X-Test-Client-Context", response.headers)  # Disabled unless configured.
        self.assertEqual(post.call_args.kwargs["json"]["model"], "claude-deployment")
        self.assertEqual(post.call_args.kwargs["headers"]["anthropic-version"], "2023-06-01")
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["input_tokens"], 5)
        self.assertEqual(cached["output_tokens"], 2)
        self.assertEqual(cached["cache_creation_input_tokens"], 3)
        self.assertEqual(cached["cache_read_input_tokens"], 1)
        self.assertEqual(cached["upstream_api"], "messages")

    def _enable_header_binding(self, additional=()):
        config = __import__("yaml").safe_load(CONFIG)
        api = config["proxies"]["demo-profile"]["apis"]["messages"]
        api["header_bindings"] = [{
            "client_name": "X-Test-Client-Context", "upstream_name": "X-Test-Upstream-Context",
            "value_type": "uuid_v4",
        }]
        api["header_bindings"].extend(additional)
        api["headers"]["x-test-upstream-context"] = "${GHC_API_TEST_UNSET_DYNAMIC_HEADER}"
        path = self.root / "session-proxies.yaml"
        path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
        self.runtime.registry = ProxyRegistry(path)

    def test_messages_header_value_is_generated_or_reused_without_logging_client_header(self):
        self._enable_header_binding()
        upstream = FakeResponse(payload={"type": "message", "content": []})
        body = {"model": "demo-model", "messages": [], "max_tokens": 16}
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                first = client.post("/proxy/demo-profile/v1/messages", json=body)
                second = client.post("/proxy/demo-profile/v1/messages", json=body)
                first_id = first.headers["X-Test-Client-Context"]
                reused = client.post("/proxy/demo-profile/v1/messages", json=body,
                                     headers={"X-Test-Client-Context": first_id.upper()})

        self.assertEqual([first.status_code, second.status_code, reused.status_code], [200] * 3)
        self.assertEqual(uuid.UUID(first_id).version, 4)
        self.assertNotEqual(first_id, second.headers["X-Test-Client-Context"])
        self.assertEqual(reused.headers["X-Test-Client-Context"], first_id)
        self.assertEqual(
            [call.kwargs["headers"]["X-Test-Upstream-Context"] for call in post.call_args_list],
            [first_id, second.headers["X-Test-Client-Context"], first_id],
        )
        self.assertNotIn("X-Test-Client-Context", post.call_args.kwargs["headers"])
        self.assertNotIn("x-test-upstream-context", post.call_args.kwargs["headers"])
        for entry in cache.cache.values():
            self.assertNotIn("X-Test-Client-Context", entry["request_headers"])

    def test_multiple_header_bindings_are_independent(self):
        self._enable_header_binding([{
            "client_name": "X-Test-Client-Trace", "upstream_name": "X-Test-Upstream-Trace",
            "value_type": "uuid_v4",
        }])
        supplied = str(uuid.uuid4())
        upstream = FakeResponse(payload={"type": "message", "content": []})
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/messages",
                                       json={"model": "demo-model", "messages": [], "max_tokens": 16},
                                       headers={"X-Test-Client-Trace": supplied})
                catalog = client.get("/proxy/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Test-Client-Trace"], supplied)
        self.assertEqual(post.call_args.kwargs["headers"]["X-Test-Upstream-Trace"], supplied)
        self.assertEqual(post.call_args.kwargs["headers"]["X-Test-Upstream-Context"],
                         response.headers["X-Test-Client-Context"])
        self.assertNotIn("X-Test-Client-Trace", next(iter(cache.cache.values()))["request_headers"])
        model = catalog.get_json()["data"][0]
        self.assertEqual(model["request_headers"]["/messages"], [
            {"name": "X-Test-Client-Context", "value_type": "uuid_v4"},
            {"name": "X-Test-Client-Trace", "value_type": "uuid_v4"},
        ])
        self.assertNotIn("X-Test-Upstream-Trace", catalog.get_data(as_text=True))

    def test_messages_invalid_header_value_never_contacts_upstream(self):
        self._enable_header_binding()
        body = {"model": "demo-model", "messages": [], "max_tokens": 16}
        with mock.patch("ghc_api.proxy.client.requests.post") as post:
            with self.app.test_client() as client:
                for bad in ("not-a-uuid", str(uuid.uuid1()), ""):
                    with self.subTest(bad=bad):
                        response = client.post("/proxy/demo-profile/v1/messages", json=body,
                                               headers={"X-Test-Client-Context": bad})
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(response.get_json()["type"], "error")
        post.assert_not_called()

    def test_messages_stream_echoes_the_configured_client_header(self):
        self._enable_header_binding()
        session_id = str(uuid.uuid4())
        upstream = FakeResponse(
            headers={"Content-Type": "text/event-stream"},
            lines=[b'event: message_stop', b'data: {"type":"message_stop"}'],
            content=b"unused",
        )
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/messages",
                    json={"model": "demo-model", "messages": [], "max_tokens": 16, "stream": True},
                    headers={"X-Test-Client-Context": session_id})
                self.assertIn("event: message_stop", response.get_data(as_text=True))
        self.assertEqual(response.headers["X-Test-Client-Context"], session_id)
        self.assertEqual(post.call_args.kwargs["headers"]["X-Test-Upstream-Context"], session_id)

    def test_messages_stream_preserves_anthropic_events_and_rewrites_model(self):
        lines = [
            b'event: message_start',
            b'data: {"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude-deployment","content":[],"usage":{"input_tokens":4,"cache_creation_input_tokens":2,"cache_read_input_tokens":1,"output_tokens":0}}}',
            b'event: content_block_start',
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            b'event: content_block_delta',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}',
            b'event: message_delta',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
            b'event: message_stop',
            b'data: {"type":"message_stop"}',
        ]
        upstream = FakeResponse(
            headers={"Content-Type": "text/event-stream"},
            lines=lines,
            content=b"unused",
        )

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/messages", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 128,
                    "stream": True,
                })
                body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message_start", body)
        self.assertIn('"model":"demo-model"', body)
        self.assertNotIn('"model":"claude-deployment"', body)
        self.assertIn("event: message_stop", body)
        self.assertNotIn("[DONE]", body)
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["input_tokens"], 4)
        self.assertEqual(cached["output_tokens"], 2)
        self.assertEqual(cached["cache_creation_input_tokens"], 2)
        self.assertEqual(cached["cache_read_input_tokens"], 1)
        self.assertEqual(cached["upstream_api"], "messages")
        self.assertIn('"model":"claude-deployment"', cached["raw_events"][0])

    def test_messages_proxy_errors_use_anthropic_shape(self):
        with self.app.test_client() as client:
            response = client.post("/proxy/missing/v1/messages", json={
                "model": "demo-model",
                "messages": [],
                "max_tokens": 16,
            })

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["type"], "error")
        self.assertEqual(response.get_json()["error"]["type"], "not_found_error")

    def test_responses_stream_rewrites_nested_model_and_extracts_usage(self):
        lines = [
            b'data: {"type":"response.created","response":{"id":"resp-1","model":"private-deployment"}}',
            b'data: {"type":"response.completed","response":{"id":"resp-1","model":"private-deployment","usage":{"input_tokens":4,"output_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}',
            b"data: [DONE]",
        ]
        upstream = FakeResponse(
            headers={"Content-Type": "text/event-stream"},
            lines=lines,
            content=b"unused",
        )

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/responses", json={
                    "model": "demo-model",
                    "input": "hello",
                    "stream": True,
                })
                body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('"model":"demo-model"', body)
        self.assertNotIn('"model":"private-deployment"', body)
        self.assertIn("data: [DONE]", body)
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["input_tokens"], 4)
        self.assertEqual(cached["output_tokens"], 2)
        self.assertEqual(cached["cache_creation_input_tokens"], 1)
        self.assertIn('"model":"private-deployment"', cached["raw_events"][-1])
        stats = cache.get_stats()
        self.assertEqual(stats["model_stats"]["demo-model"]["input_tokens"], 4)
        self.assertEqual(stats["model_stats"]["demo-model"]["output_tokens"], 2)
        self.assertEqual(stats["model_stats"]["demo-model"]["cache_creation_input_tokens"], 1)

    def test_non_stream_cache_records_effective_upstream_model(self):
        upstream_payload = {
            "id": "chat-1",
            "model": "private-deployment",
            "choices": [{"message": {"role": "assistant", "content": "你好"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        upstream_content = json.dumps(upstream_payload, ensure_ascii=False).encode("utf-8")
        upstream = FakeResponse(payload=upstream_payload, content=upstream_content)

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/chat/completions", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "你好"}],
                })

        self.assertEqual(response.status_code, 200)
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["model"], "demo-model")
        self.assertEqual(cached["translated_model"], "chat-deployment")
        self.assertEqual(
            cached["request_size"],
            len(json.dumps(post.call_args.kwargs["json"]).encode("utf-8")),
        )
        self.assertEqual(cached["response_size"], len(upstream_content))

    def test_error_responses_do_not_add_usage_to_token_stats(self):
        upstream = FakeResponse(
            status_code=429,
            payload={
                "error": {"message": "rate limited"},
                "usage": {
                    "prompt_tokens": 99,
                    "completion_tokens": 88,
                    "prompt_tokens_details": {"cached_tokens": 77},
                },
            },
        )

        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/chat/completions", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hello"}],
                })

        self.assertEqual(response.status_code, 429)
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["input_tokens"], 0)
        self.assertEqual(cached["output_tokens"], 0)
        self.assertEqual(cached["cache_read_input_tokens"], 0)

    def test_invalid_model_catalog_returns_configuration_error(self):
        config_path = self.root / "invalid-proxies.yaml"
        config_path.write_text("proxies: [invalid", encoding="utf-8")
        invalid_runtime = ProxyRuntime(
            registry=ProxyRegistry(config_path),
            affinity_store=ProxyAffinityStore(self.root / "invalid-affinity.json"),
        )

        with mock.patch.object(proxy_routes, "proxy_runtime", invalid_runtime):
            with self.app.test_client() as client:
                response = client.get("/proxy/models")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["code"], "proxy_config_error")

    def test_model_catalog_exposes_profile_routing_without_private_config(self):
        with self.app.test_client() as client:
            response = client.get("/proxy/models")

        self.assertEqual(response.status_code, 200)
        model = response.get_json()["data"][0]
        self.assertEqual(model["id"], "demo-model")
        self.assertEqual(model["profile"], "demo-profile")
        self.assertEqual(model["base_url"], "/proxy/demo-profile/v1")
        self.assertEqual(
            model["supported_endpoints"],
            ["/responses", "/chat/completions", "/messages"],
        )
        self.assertNotIn("headers", model)
        self.assertNotIn("upstream_url", model)
        self.assertNotIn("request_headers", model)

    def test_catalog_exposes_only_the_configured_client_header(self):
        self._enable_header_binding()
        with self.app.test_client() as client:
            response = client.get("/proxy/models")
            per_profile = client.get("/proxy/demo-profile/v1/models")
        self.assertEqual(response.status_code, 200)
        model = response.get_json()["data"][0]
        self.assertEqual(model["request_headers"], {"/messages": [{
            "name": "X-Test-Client-Context", "value_type": "uuid_v4",
        }]})
        self.assertNotIn("X-Test-Upstream-Context", response.get_data(as_text=True))
        self.assertNotIn("X-Test-Upstream-Context", per_profile.get_data(as_text=True))

    def test_chat_ui_discovers_and_routes_configured_proxy_models(self):
        with self.app.test_client() as client:
            response = client.get("/chat")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("fetch('/proxy/models')", html)
        self.assertIn("Configured proxy unavailable: ", html)
        self.assertIn("Configured proxy: ", html)
        self.assertIn("baseUrl + (apiStyle === 'responses'", html)
        self.assertIn("stream_options: { include_usage: true }", html)

    def test_models_are_isolated_to_profile_endpoint(self):
        with self.app.test_client() as client:
            response = client.get("/proxy/demo-profile/v1/models")

        self.assertEqual(response.status_code, 200)
        model = response.get_json()["data"][0]
        self.assertEqual(model["id"], "demo-model")
        self.assertEqual(
            model["supported_endpoints"],
            ["/responses", "/chat/completions", "/messages"],
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

    def test_approved_proxy_user_is_recorded_in_per_user_usage(self):
        state.enable_auth = True
        upstream = FakeResponse(payload={
            "id": "chat-1",
            "model": "private-deployment",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })

        with mock.patch.object(
            proxy_routes, "require_auth", return_value=AuthResult(user_id="alice")
        ), mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post("/proxy/demo-profile/v1/chat/completions", json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hello"}],
                })

        self.assertEqual(response.status_code, 200)
        cached = next(iter(cache.cache.values()))
        self.assertEqual(cached["user_id"], "alice")
        snapshot = cache.get_user_model_token_snapshot()
        self.assertEqual(snapshot[("alice", "chat-deployment")]["input_tokens"], 2)
        self.assertEqual(snapshot[("alice", "chat-deployment")]["output_tokens"], 1)

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
                response = client.get("/proxy/models")

        self.assertEqual(response.status_code, 401)
        require.assert_called_once()
        self.assertNotIn("/proxy/demo-profile/v1/models", PROTECTED_PATHS)


if __name__ == "__main__":
    unittest.main()
