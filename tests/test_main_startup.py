"""Startup regression tests for ghc_api.main.main()."""
import sys
from unittest import mock

import pytest

import importlib

# ghc_api re-exports the main() function, so import the module explicitly.
main_module = importlib.import_module("ghc_api.main")


@pytest.fixture
def startup_env(tmp_path, monkeypatch):
    """Patch out everything main() touches except config loading and app.run()."""
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ghc-api"])

    fake_app = mock.MagicMock()
    monkeypatch.setattr(main_module, "create_app", lambda: fake_app)
    monkeypatch.setattr(main_module, "initialize_app", lambda: None)
    monkeypatch.setattr(main_module, "print_sync_diff_status", lambda: None)
    monkeypatch.setattr(main_module, "print_available_models", lambda: None)
    monkeypatch.setattr(main_module, "print_model_mappings", lambda: None)
    monkeypatch.setattr(main_module, "generate_config_file", lambda: None)
    return tmp_path, fake_app


def test_broken_config_falls_back_to_defaults(startup_env, capsys):
    """A config file that cannot be parsed must not crash startup."""
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text("address: [unclosed\n", encoding="utf-8")

    main_module.main()

    fake_app.run.assert_called_once()
    kwargs = fake_app.run.call_args.kwargs
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 8313
    assert "Error loading config file" in capsys.readouterr().out


def test_valid_config_is_used(startup_env):
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "address: 127.0.0.1\nport: 9999\ndebug: false\n", encoding="utf-8"
    )

    main_module.main()

    kwargs = fake_app.run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999
