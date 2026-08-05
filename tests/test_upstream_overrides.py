import os
from unittest import mock

import pytest
import yaml

from ghc_api import api_helpers
from ghc_api.main import apply_upstream_config, configure_ghe_endpoint
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


@pytest.mark.parametrize("endpoint", [
    "octocorp.ghe.com",
    "https://octocorp.ghe.com/",
    "https://api.octocorp.ghe.com",
    "https://copilot-api.octocorp.ghe.com",
])
def test_resolve_ghe_endpoints_accepts_common_endpoint_forms(endpoint):
    assert api_helpers.resolve_ghe_endpoints(endpoint) == {
        "github_web_base_url": "https://octocorp.ghe.com",
        "github_api_base_url": "https://api.octocorp.ghe.com",
        "copilot_api_base_url": "https://copilot-api.octocorp.ghe.com",
    }


@pytest.mark.parametrize("endpoint", [
    "",
    "http://octocorp.ghe.com",
    "https://ghe.com",
    "https://octocorp.example.com",
    "https://octocorp.ghe.com/path",
    "https://user:secret@octocorp.ghe.com",
])
def test_resolve_ghe_endpoints_rejects_invalid_values(endpoint):
    with pytest.raises(ValueError):
        api_helpers.resolve_ghe_endpoints(endpoint)


def test_configure_ghe_endpoint_updates_config_without_dropping_comments(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "# keep this comment\n"
        "address: localhost\n"
        "github_api_base_url: \"\"\n"
        "copilot_api_base_url: \"\"\n"
        "model_mappings:\n"
        "  exact:\n"
        "    opus: claude-opus\n",
        encoding="utf-8",
    )

    resolved = configure_ghe_endpoint(str(config_path), "octocorp.ghe.com")

    content = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    assert "# keep this comment" in content
    assert config["address"] == "localhost"
    assert config["model_mappings"]["exact"]["opus"] == "claude-opus"
    assert config["github_api_base_url"] == resolved["github_api_base_url"]
    assert config["copilot_api_base_url"] == resolved["copilot_api_base_url"]


def test_configure_ghe_endpoint_creates_minimal_config_when_missing(tmp_path):
    config_path = tmp_path / "config.yaml"

    configure_ghe_endpoint(str(config_path), "https://api.octocorp.ghe.com")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config == {
        "github_api_base_url": "https://api.octocorp.ghe.com",
        "copilot_api_base_url": "https://copilot-api.octocorp.ghe.com",
    }


def test_github_web_base_url_defaults_to_github_com():
    old = state.github_api_base_url
    try:
        state.github_api_base_url = ""
        assert api_helpers.get_github_web_base_url() == "https://github.com"
    finally:
        state.github_api_base_url = old


def test_github_web_base_url_is_derived_for_ghe_data_residency():
    old = state.github_api_base_url
    try:
        state.github_api_base_url = "https://api.octocorp.ghe.com/"
        assert api_helpers.get_github_web_base_url() == "https://octocorp.ghe.com"
    finally:
        state.github_api_base_url = old


@pytest.mark.parametrize("url", [
    "http://api.octocorp.ghe.com",
    "https://copilot-api.octocorp.ghe.com",
    "https://api.ghe.com",
    "https://api.octocorp.ghe.com/custom/path",
    "https://user:secret@api.octocorp.ghe.com",
])
def test_github_web_base_url_rejects_unsafe_or_ambiguous_overrides(url):
    old = state.github_api_base_url
    try:
        state.github_api_base_url = url
        with pytest.raises(ValueError):
            api_helpers.get_github_web_base_url()
    finally:
        state.github_api_base_url = old


def test_apply_upstream_config_sets_ghe_endpoints():
    saved = (state.account_type, state.github_api_base_url, state.copilot_api_base_url)
    try:
        apply_upstream_config({
            "account_type": "enterprise",
            "github_api_base_url": "https://api.octocorp.ghe.com",
            "copilot_api_base_url": "https://copilot-api.octocorp.ghe.com",
        })
        assert state.account_type == "enterprise"
        assert state.github_api_base_url == "https://api.octocorp.ghe.com"
        assert state.copilot_api_base_url == "https://copilot-api.octocorp.ghe.com"
    finally:
        state.account_type, state.github_api_base_url, state.copilot_api_base_url = saved


def test_config_dir_environment_override_is_absolute():
    with mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": "./build/test-config"}):
        assert get_config_dir() == os.path.abspath("./build/test-config")
