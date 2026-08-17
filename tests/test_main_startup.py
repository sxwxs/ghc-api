"""Startup regression tests for ghc_api.main.main()."""
import copy
import importlib
import sys
from unittest import mock

import pytest

from ghc_api.state import state

# ghc_api re-exports the main() function, so import the module explicitly.
main_module = importlib.import_module("ghc_api.main")


@pytest.fixture
def startup_env(tmp_path, monkeypatch):
    """Run main() in isolation: no real app, no global state mutations leaking out."""
    monkeypatch.setenv("GHC_API_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ghc-api"])

    fake_app = mock.MagicMock()
    monkeypatch.setattr(main_module, "create_app", lambda: fake_app)
    monkeypatch.setattr(main_module, "initialize_app", lambda: None)
    monkeypatch.setattr(main_module, "print_sync_diff_status", lambda: None)
    monkeypatch.setattr(main_module, "print_available_models", lambda: None)
    monkeypatch.setattr(main_module, "print_model_mappings", lambda: None)
    monkeypatch.setattr(main_module, "generate_config_file", lambda: None)

    # Keep process-wide singletons out of these tests so they cannot leak into
    # (or depend on) unrelated tests.
    monkeypatch.setattr(main_module, "apply_upstream_config", lambda config: None)
    monkeypatch.setattr(main_module, "model_mappings", mock.MagicMock())
    monkeypatch.setattr(main_module, "chat_completions_model_support", mock.MagicMock())

    state_snapshot = {
        key: copy.copy(value)
        for key, value in vars(state).items()
        if not key.endswith("lock")
    }
    try:
        yield tmp_path, fake_app
    finally:
        for key, value in state_snapshot.items():
            setattr(state, key, value)


def test_broken_config_aborts_startup(startup_env, capsys):
    """A config file that cannot be parsed must abort instead of running half-configured."""
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text("address: [unclosed\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
    fake_app.run.assert_not_called()
    assert "Error loading config file" in capsys.readouterr().err


def test_invalid_setting_value_aborts_startup(startup_env, capsys):
    """A well-formed YAML with an unusable value must not start with defaults silently."""
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "address: 127.0.0.1\nport: 9999\nsession_flush_interval:\nenable_auth: true\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
    fake_app.run.assert_not_called()
    assert "refusing to start" in capsys.readouterr().err


def test_valid_config_is_used(startup_env):
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "address: 127.0.0.1\nport: 9999\ndebug: true\n", encoding="utf-8"
    )

    main_module.main()

    kwargs = fake_app.run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999
    assert kwargs["debug"] is True


def test_canonical_webiq_config_covers_all_services(startup_env):
    tmp_path, _fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "debug: true\n"
        "enable_webiq: true\n"
        "webiq_browse_timeout: 150\n"
        "webiq_classic_timeout: 75\n"
        "webiq_mcp_timeout: 180\n"
        "webiq_mcp_max_concurrent_streams: 6\n",
        encoding="utf-8",
    )

    main_module.main()

    assert state.enable_webiq_search is True
    assert state.webiq_browse_timeout == 150
    assert state.webiq_classic_timeout == 75
    assert state.webiq_mcp_timeout == 180
    assert state.webiq_mcp_max_concurrent_streams == 6


def test_enable_webiq_wins_over_conflicting_legacy_alias(startup_env, capsys):
    tmp_path, _fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "debug: true\nenable_webiq: false\nenable_webiq_search: true\n",
        encoding="utf-8",
    )

    main_module.main()

    assert state.enable_webiq_search is False
    assert "enable_webiq takes precedence" in capsys.readouterr().out


@pytest.mark.parametrize("configured,expected", [
    ("0.25", 0.25),
    ("0", 0.0),
    # The clamp exists because these three are not benign once they reach
    # queue.Queue.get(timeout=...): a negative raises ValueError and inf raises
    # OverflowError (turning every streaming request into a 500), while nan
    # silently disables the timeout so the route blocks like it used to.
    ("-1", 0.0),
    (".nan", 0.0),
    (".inf", 5.0),
    ("42", 5.0),
])
def test_responses_pre_header_grace_is_clamped(startup_env, configured, expected):
    tmp_path, fake_app = startup_env
    (tmp_path / "config.yaml").write_text(
        "address: 127.0.0.1\nport: 9999\ndebug: true\n"
        f"responses_pre_header_grace: {configured}\n",
        encoding="utf-8",
    )

    main_module.main()

    assert state.responses_pre_header_grace == expected
