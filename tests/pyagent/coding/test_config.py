"""config.json loading and model selection precedence."""

from pathlib import Path

import pytest

from ghc_api.pyagent.ai.models import ModelError, merge_providers
from ghc_api.pyagent.coding.config import Config, ConfigError, load_config, select_model


def test_missing_config_means_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path) == Config()


def test_config_uses_camel_case_keys(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"defaultProvider": "mine", "defaultModel": "m1", "defaultThinking": "high",'
        ' "providers": {"mine": {"api": "openai_chat", "baseUrl": "http://localhost:9/v1"}}}'
    )
    config = load_config(tmp_path)
    model = select_model(config, merge_providers(config.providers), None, None)
    assert (model.provider, model.id, model.base_url) == ("mine", "m1", "http://localhost:9/v1")
    assert config.default_thinking == "high"


def test_invalid_config_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"defaultThinking": "extreme"}')
    with pytest.raises(ConfigError, match=r"config\.json"):
        load_config(tmp_path)


def test_flags_override_config_and_credentials_pick_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = merge_providers({})
    config = Config(default_provider="deepseek", default_model="deepseek-flash")
    assert select_model(config, providers, None, None).id == "deepseek-flash"
    assert select_model(config, providers, "anthropic", None).id == "claude-opus-5-5"
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelError, match="No provider configured"):
        select_model(Config(), providers, None, None)
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    assert select_model(Config(), providers, None, None).provider == "moonshot"
