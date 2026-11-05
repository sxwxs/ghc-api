import copy
import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.auth import AuthResult
from ghc_api.model_metadata import normalize_model_metadata
from ghc_api.state import state


MODEL = {
    "id": "vendor/example-model",
    "name": "Example Model",
    "vendor": "Example",
    "supported_endpoints": ["/responses", "/v1/chat/completions"],
    "capabilities": {
        "type": "chat",
        "limits": {
            "max_context_window_tokens": 200000,
            "max_prompt_tokens": 180000,
            "max_output_tokens": 32000,
        },
        "supports": {
            "vision": True,
            "reasoning_effort": ["low", "medium", "high"],
            "tool_calls": True,
            "streaming": True,
            "future_capability": {"enabled": True},
        },
    },
}


class ModelMetadataTest(unittest.TestCase):
    def setUp(self):
        self.models_patch = mock.patch.object(state, "models", {"data": [copy.deepcopy(MODEL)]})
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)
        self.auth_patch = mock.patch.object(state, "enable_auth", False)
        self.auth_patch.start()
        self.addCleanup(self.auth_patch.stop)
        self.client = create_app().test_client()

    def test_list_and_detail_share_metadata_without_refreshing_cached_catalog(self):
        original = copy.deepcopy(state.models)
        with mock.patch("ghc_api.api_helpers.fetch_models") as fetch:
            with mock.patch("ghc_api.api_helpers.refresh_copilot_token") as refresh:
                for prefix in ("/v1/models", "/models"):
                    with self.subTest(prefix=prefix):
                        listing = self.client.get(prefix)
                        detail = self.client.get(prefix + "/vendor/example-model")
                        self.assertEqual(listing.status_code, 200)
                        self.assertEqual(detail.status_code, 200)
                        model = listing.get_json()["data"][0]
                        self.assertEqual(model, detail.get_json())
                        self.assertEqual(model["object"], "model")
                        self.assertEqual(model["display_name"], "Example Model")
                        self.assertEqual(model["context_window"], 200000)
                        self.assertEqual(model["max_input_tokens"], 180000)
                        self.assertEqual(model["max_output_tokens"], 32000)
                        self.assertEqual(model["input"], ["text", "image"])
                        self.assertIs(model["reasoning"], True)
                        self.assertEqual(model["reasoning_efforts"], ["low", "medium", "high"])
                        self.assertIs(model["tool_calls"], True)
                        self.assertIs(model["streaming"], True)
                        self.assertEqual(model["supported_endpoints"], MODEL["supported_endpoints"])
                        self.assertEqual(model["capabilities"], MODEL["capabilities"])
                fetch.assert_not_called()
                refresh.assert_not_called()
        self.assertEqual(state.models, original)

    def test_unknown_values_are_null_and_false_capabilities_stay_false(self):
        state.models = {"data": [{"id": "unknown"}, {
            "id": "limited",
            "capabilities": {
                "limits": {"max_context_window_tokens": 10000},
                "supports": {"reasoning": False, "vision": False, "tool_calls": False, "streaming": False},
            },
        }]}
        unknown, limited = self.client.get("/v1/models").get_json()["data"]
        for field in ("context_window", "max_input_tokens", "max_output_tokens", "reasoning",
                      "reasoning_efforts", "input", "tool_calls", "streaming", "supported_endpoints", "capabilities"):
            self.assertIsNone(unknown[field], field)
        self.assertEqual(limited["context_window"], 10000)
        self.assertIsNone(limited["max_input_tokens"])
        self.assertIsNone(limited["max_output_tokens"])
        self.assertEqual(limited["input"], ["text"])
        for field in ("reasoning", "tool_calls", "streaming"):
            self.assertIs(limited[field], False)

    def test_invalid_limits_are_unknown_and_raw_capabilities_are_preserved(self):
        for value in (0, -1, True, "200000", 200000.0):
            with self.subTest(value=value):
                model = {"capabilities": {"limits": {"max_context_window_tokens": value}}}
                metadata = normalize_model_metadata(model)
                self.assertIsNone(metadata["context_window"])
                self.assertEqual(metadata["capabilities"], model["capabilities"])
                metadata["capabilities"]["limits"]["max_context_window_tokens"] = 1
                self.assertEqual(model["capabilities"]["limits"]["max_context_window_tokens"], value)

    def test_missing_model_is_404(self):
        response = self.client.get("/v1/models/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["code"], "model_not_found")

    def test_empty_catalog_is_a_valid_list(self):
        state.models = {"data": []}
        with mock.patch("ghc_api.api_helpers.fetch_models") as fetch:
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"], [])
        fetch.assert_not_called()

    def test_uninitialized_catalog_loads_once_for_list_detail_and_full_routes(self):
        def fetch_catalog():
            state.models = {"data": [copy.deepcopy(MODEL)]}
            return True

        for path in ("/v1/models", "/models/vendor/example-model", "/v1/models/full/"):
            with self.subTest(path=path):
                state.models = None
                with mock.patch("ghc_api.api_helpers.fetch_models", side_effect=fetch_catalog) as fetch:
                    response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                fetch.assert_called_once_with()

    def test_upstream_catalog_failure_is_502(self):
        for result in (False, RuntimeError("upstream unavailable")):
            for path in ("/v1/models", "/v1/models/example", "/models/full/"):
                with self.subTest(result=result, path=path):
                    state.models = None
                    kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                    with mock.patch("ghc_api.api_helpers.fetch_models", **kwargs):
                        response = self.client.get(path)
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(response.get_json()["error"]["code"], "model_catalog_unavailable")

    def test_full_routes_keep_raw_catalog_and_reserve_the_full_name(self):
        for prefix in ("/v1/models/full", "/models/full"):
            for suffix in ("", "/"):
                with self.subTest(path=prefix + suffix):
                    response = self.client.get(prefix + suffix)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_json(), state.models)

    def test_metadata_routes_require_auth_before_loading_models(self):
        state.enable_auth = True
        state.models = None
        denied = AuthResult(None, "missing_token", "missing", 401)
        paths = ("/v1/models", "/models/vendor/example-model", "/v1/models/vendor/example-model",
                 "/v1/models/full", "/v1/models/full/", "/models/full", "/models/full/")
        with mock.patch("ghc_api.app.require_auth", return_value=denied) as require:
            with mock.patch("ghc_api.api_helpers.fetch_models") as fetch:
                for path in paths:
                    with self.subTest(path=path):
                        self.assertEqual(self.client.get(path).status_code, 401)
                self.assertEqual(require.call_count, len(paths))
                fetch.assert_not_called()
