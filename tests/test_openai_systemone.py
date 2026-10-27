import json
import unittest
from unittest import mock

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.state import state


class FakeResponse:
    def __init__(self, body, status_code=200, content_type="application/json"):
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
        self.content = self.text.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def json(self):
        if not isinstance(self._body, (dict, list)):
            raise ValueError("not json")
        return self._body


DECISION_REQUEST = {
    "model": "microsoft-decision-1",
    "state": "The payment callback API returns 500 on every call.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Charges, invoices, and refunds",
                "engineering": "Bugs, errors, and outages",
            },
        }
    },
}

DECISION_RESPONSE = {
    "model": "microsoft-decision-1",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "engineering",
            "confidence": 0.94,
            "probabilities": {"billing": 0.06, "engineering": 0.94},
        }
    },
    "usage": {"input_tokens": 57, "output_tokens": 1},
}


@mock.patch("ghc_api.routes.openai.ensure_copilot_token")
@mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={})
class CopilotSystemOneTests(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_retries = state.max_connection_retries
        state.models = {
            "data": [
                # Shape reported by Copilot's live /models listing.
                {
                    "id": "microsoft-decision-1",
                    "capabilities": {"type": "decisions", "family": "microsoft-decision-1"},
                    "supported_endpoints": ["/v1/systemone"],
                },
                {"id": "decision-bare-path", "supported_endpoints": ["/systemone"]},
                {"id": "decision-type-only", "capabilities": {"type": "decisions"}},
                {"id": "gpt-chat", "supported_endpoints": ["/chat/completions", "/responses"]},
            ]
        }
        state.max_connection_retries = 0
        self.client = create_app().test_client()

    def tearDown(self):
        state.models = self.saved_models
        state.max_connection_retries = self.saved_retries

    def test_routes_are_protected(self, *_):
        self.assertIn("/v1/systemone", PROTECTED_PATHS)
        self.assertIn("/systemone", PROTECTED_PATHS)

    @mock.patch("ghc_api.routes.openai.cache.add_request")
    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_forwards_body_unchanged_and_records_usage(self, post, add_request, *_):
        post.return_value = FakeResponse(DECISION_RESPONSE)

        response = self.client.post("/v1/systemone", json=DECISION_REQUEST)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), DECISION_RESPONSE)
        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/v1/systemone"))
        self.assertEqual(kwargs["json"], DECISION_REQUEST)
        self.assertEqual(kwargs["headers"]["X-Initiator"], "user")

        cached = add_request.call_args.args[1]
        self.assertEqual(cached["endpoint"], "/v1/systemone")
        self.assertEqual(cached["input_tokens"], 57)
        self.assertEqual(cached["output_tokens"], 1)

    @mock.patch("ghc_api.routes.openai.cache.add_request")
    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_uses_advertised_bare_path(self, post, _add_request, *_):
        post.return_value = FakeResponse(DECISION_RESPONSE)

        response = self.client.post("/systemone", json={**DECISION_REQUEST, "model": "decision-bare-path"})

        self.assertEqual(response.status_code, 200)
        url = post.call_args.args[0]
        self.assertTrue(url.endswith("/systemone"))
        self.assertFalse(url.endswith("/v1/systemone"))

    @mock.patch("ghc_api.routes.openai.cache.add_request")
    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_decisions_type_without_endpoints_uses_v1_path(self, post, _add_request, *_):
        post.return_value = FakeResponse(DECISION_RESPONSE)

        response = self.client.post("/v1/systemone", json={**DECISION_REQUEST, "model": "decision-type-only"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(post.call_args.args[0].endswith("/v1/systemone"))

    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_rejects_streaming(self, post, *_):
        response = self.client.post("/v1/systemone", json={**DECISION_REQUEST, "stream": True})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_stream")
        post.assert_not_called()

    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_rejects_non_decision_model(self, post, *_):
        response = self.client.post("/v1/systemone", json={**DECISION_REQUEST, "model": "gpt-chat"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_model")
        post.assert_not_called()

    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_rejects_missing_model(self, post, *_):
        body = dict(DECISION_REQUEST)
        body.pop("model")
        response = self.client.post("/v1/systemone", json=body)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "missing_model")
        post.assert_not_called()

    @mock.patch("ghc_api.routes.openai.log_error_request")
    @mock.patch("ghc_api.routes.openai.cache.add_request")
    @mock.patch("ghc_api.routes.openai.requests.post")
    def test_upstream_validation_error_is_passed_through(self, post, add_request, log_error, *_):
        error_body = {"detail": "A request may contain at most 64 questions."}
        post.return_value = FakeResponse(error_body, status_code=422)

        response = self.client.post("/v1/systemone", json=DECISION_REQUEST)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json(), error_body)
        self.assertEqual(add_request.call_args.args[1]["input_tokens"], 0)
        log_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
