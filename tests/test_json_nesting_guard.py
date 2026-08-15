"""Tests for the shared JSON nesting guard.

A body the JSON decoder happens to accept can still blow the stack in
``copy.deepcopy`` or ``json.dumps`` further in, and ``request.get_json()``
itself raises ``RecursionError`` (which ``silent=True`` does not catch) once
nesting gets deep enough. Both used to surface as a 500, so the guard has to
run before any of that, for every endpoint.
"""

import json
import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.json_guard import MAX_JSON_NESTING_DEPTH, exceeds_max_nesting
from ghc_api.routes import anthropic as anthropic_module
from ghc_api.routes import openai as openai_module
from ghc_api.state import state


def _nest(depth):
    return "[" * depth + "]" * depth


class JsonGuardUnitTests(unittest.TestCase):
    def test_depth_boundary_is_exact(self):
        limit = MAX_JSON_NESTING_DEPTH
        self.assertFalse(exceeds_max_nesting(_nest(limit).encode()))
        self.assertTrue(exceeds_max_nesting(_nest(limit + 1).encode()))

    def test_objects_and_mixed_containers_count_as_nesting(self):
        deep = '{"a":' * (MAX_JSON_NESTING_DEPTH + 1) + "1" + "}" * (MAX_JSON_NESTING_DEPTH + 1)
        self.assertTrue(exceeds_max_nesting(deep.encode()))

        mixed_depth = (MAX_JSON_NESTING_DEPTH + 2) // 2
        mixed = '[{"a":' * mixed_depth + "1" + "}]" * mixed_depth
        self.assertTrue(exceeds_max_nesting(mixed.encode()))

    def test_brackets_inside_strings_are_not_structure(self):
        body = json.dumps({"text": "[" * 5000, "escaped": '"[[[', "path": "a\\[b"})
        self.assertFalse(exceeds_max_nesting(body.encode()))

    def test_siblings_do_not_accumulate_depth(self):
        """Depth is nesting, not bracket count: a long conversation is flat."""
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}] * 5000})
        self.assertFalse(exceeds_max_nesting(body.encode()))

    def test_accepts_str_as_well_as_bytes(self):
        self.assertTrue(exceeds_max_nesting(_nest(MAX_JSON_NESTING_DEPTH + 1)))
        self.assertFalse(exceeds_max_nesting(_nest(2)))

    def test_multibyte_utf8_does_not_disturb_the_scan(self):
        body = json.dumps({"text": "日本語テキスト 🎉", "n": [[1]]}).encode("utf-8")
        self.assertFalse(exceeds_max_nesting(body))

    def test_unterminated_string_does_not_confuse_the_depth_count(self):
        """A stray quote leaves the scan out of sync with reality; the decoder
        rejects the body a moment later, so the guard must not mis-report."""
        self.assertFalse(exceeds_max_nesting(b'{"text":"[[[['))
        self.assertTrue(exceeds_max_nesting(b'{"text":"unterminated' + _nest(500).encode()))

    def test_custom_limit_is_honoured(self):
        self.assertTrue(exceeds_max_nesting(_nest(5).encode(), limit=4))
        self.assertFalse(exceeds_max_nesting(_nest(4).encode(), limit=4))


class DeepJsonRequestRejectionTests(unittest.TestCase):
    """Depth 100000 is the case that used to raise RecursionError out of
    ``request.get_json()`` itself; depth 2000 is the case the decoder accepted
    and that died later in ``copy.deepcopy``."""

    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self._saved_auth = state.enable_auth
        state.enable_auth = False

    def tearDown(self):
        state.enable_auth = self._saved_auth

    def _post(self, path, depth, body_template):
        body = (body_template % _nest(depth)).encode()
        upstream = mock.patch(
            "requests.post", side_effect=AssertionError("upstream must not be reached")
        )
        with upstream, mock.patch.object(
            openai_module, "ensure_copilot_token", create=True
        ), mock.patch.object(anthropic_module, "ensure_copilot_token", create=True):
            return self.client.post(path, data=body, content_type="application/json")

    def test_openai_shaped_endpoints_answer_400(self):
        paths = {
            "/v1/chat/completions": '{"model":"gpt-4o","messages":[],"nested":%s}',
            "/v1/responses": '{"model":"gpt-4o","input":"hi","nested":%s}',
            "/v1/embeddings": '{"model":"text-embedding-3-small","input":"hi","nested":%s}',
            "/proxy/demo/v1/chat/completions": '{"model":"m","messages":[],"nested":%s}',
        }
        for path, template in paths.items():
            for depth in (MAX_JSON_NESTING_DEPTH + 1, 2000, 100000):
                with self.subTest(path=path, depth=depth):
                    response = self._post(path, depth, template)
                    self.assertEqual(response.status_code, 400)
                    error = response.get_json()["error"]
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(error["code"], "invalid_json")
                    self.assertIn("nesting", error["message"].lower())

    def test_anthropic_endpoints_answer_400_in_anthropic_error_shape(self):
        paths = {
            "/v1/messages": '{"model":"gpt-5.6-sol","messages":[],"max_tokens":16,"nested":%s}',
            "/v1/messages/count_tokens": '{"model":"gpt-5.6-sol","messages":[],"nested":%s}',
        }
        for path, template in paths.items():
            for depth in (MAX_JSON_NESTING_DEPTH + 1, 2000, 100000):
                with self.subTest(path=path, depth=depth):
                    response = self._post(path, depth, template)
                    self.assertEqual(response.status_code, 400)
                    body = response.get_json()
                    self.assertEqual(body["type"], "error")
                    self.assertEqual(body["error"]["type"], "invalid_request_error")
                    self.assertIn("nesting", body["error"]["message"].lower())

    def test_count_tokens_does_not_fall_back_to_a_fabricated_count(self):
        """count_tokens answers every other error with input_tokens=1 so client
        context math never hard-fails; a body this deep is not countable and
        must not be reported as if it were."""
        response = self._post(
            "/v1/messages/count_tokens",
            2000,
            '{"model":"gpt-5.6-sol","messages":[],"nested":%s}',
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("input_tokens", response.get_json())

    def test_dashboard_and_agent_endpoints_are_covered_too(self):
        """The hook is global because get_json() raises RecursionError before
        any route code runs, including on non-LLM endpoints."""
        for path in ("/api/runtime-config", "/api/agent/sessions", "/v3/search/web"):
            with self.subTest(path=path):
                response = self._post(path, 100000, '{"query":"q","nested":%s}')
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "nesting", response.get_json()["error"]["message"].lower()
                )

    def test_non_json_bodies_are_not_scanned(self):
        """The guard keys off the JSON content type, so file uploads and other
        binary bodies never pay for the scan. Those routes still answer for
        themselves -- what must not happen is a nesting rejection."""
        deep = ('{"nested":%s}' % _nest(100000)).encode()
        for content_type, data in (
            ("text/plain", deep),
            ("application/octet-stream", deep),
            ("multipart/form-data", {"nothing": "here"}),
        ):
            with self.subTest(content_type=content_type):
                response = self.client.post(
                    "/api/requests/import", data=data, content_type=content_type
                )
                self.assertNotIn(b"nesting", response.data.lower())

    def test_shallow_request_still_reaches_the_route(self):
        with mock.patch.object(
            openai_module, "ensure_copilot_token", create=True
        ), mock.patch.object(
            openai_module, "translate_model_name", return_value="gpt-4o"
        ), mock.patch.object(
            openai_module.cache, "add_request"
        ), mock.patch.object(
            openai_module, "get_copilot_headers", return_value={}
        ), mock.patch.object(
            openai_module.requests, "post"
        ) as post:
            post.return_value = mock.Mock(
                status_code=200,
                ok=True,
                text='{"ok":true}',
                content=b'{"ok":true}',
                headers={"Content-Type": "application/json"},
                json=lambda: {"ok": True},
            )
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    # Nested well under the limit, as real tool schemas are.
                    "tools": [{"function": {"parameters": {"a": {"b": [1]}}}}],
                },
            )
            self.assertEqual(response.status_code, 200)
            post.assert_called_once()

    def test_auth_runs_before_the_guard(self):
        """Order matters: an unauthenticated client on a protected path must be
        turned away without the server scanning its body first."""
        state.enable_auth = True
        response = self.client.post(
            "/v1/chat/completions",
            data=('{"model":"gpt-4o","nested":%s}' % _nest(100000)).encode(),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
