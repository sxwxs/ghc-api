import os
from unittest import mock

from ghc_api import api_helpers
from ghc_api.state import state
from ghc_api.utils import get_config_dir


def test_copilot_base_url_override_takes_precedence():
    old = state.copilot_api_base_url
    try:
        state.copilot_api_base_url = "http://127.0.0.1:18400/"
        assert api_helpers.get_copilot_base_url() == "http://127.0.0.1:18400"
    finally:
        state.copilot_api_base_url = old


def test_github_base_url_override_takes_precedence():
    old = state.github_api_base_url
    try:
        state.github_api_base_url = "http://127.0.0.1:18401/"
        assert api_helpers.get_github_api_base_url() == "http://127.0.0.1:18401"
    finally:
        state.github_api_base_url = old


def test_config_dir_environment_override_is_absolute():
    with mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": "./build/test-config"}):
        assert get_config_dir() == os.path.abspath("./build/test-config")
