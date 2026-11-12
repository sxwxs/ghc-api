"""Provider presets, user overrides, model resolution, credentials and pricing."""

import pytest

from ghc_api.pyagent.ai.models import (
    Model,
    ModelCost,
    ModelError,
    ModelSpec,
    ProviderOverride,
    api_key_for,
    compute_cost,
    merge_providers,
    resolve_model,
)
from ghc_api.pyagent.ai.types import Usage


def test_override_changes_only_the_given_fields_of_a_preset() -> None:
    providers = merge_providers({"deepseek": ProviderOverride(api_key="sk", models={"x": ModelSpec(reasoning=True)})})
    deepseek = providers["deepseek"]
    assert deepseek.api_key == "sk"
    assert deepseek.base_url == "https://api.deepseek.com"
    assert {"deepseek-v4-pro", "x"} <= deepseek.models.keys()


def test_new_providers_need_api_and_base_url() -> None:
    with pytest.raises(ModelError, match="api and baseUrl"):
        merge_providers({"mine": ProviderOverride(api_key="k")})
    providers = merge_providers({"mine": ProviderOverride(api="openai_chat", base_url="http://localhost:1/v1")})
    assert providers["mine"].api == "openai_chat"


def test_resolve_uses_the_default_model_and_spec_defaults() -> None:
    providers = merge_providers({})
    model = resolve_model(providers, "deepseek", None)
    assert (model.id, model.context_window, model.reasoning) == ("deepseek-v4-pro", 1_000_000, True)
    unknown = resolve_model(providers, "deepseek", "future-model")
    assert (unknown.context_window, unknown.reasoning, unknown.thinking_format) == (200_000, False, "deepseek")
    with pytest.raises(ModelError, match="Unknown provider"):
        resolve_model(providers, "nope", None)
    with pytest.raises(ModelError, match="no default model"):
        resolve_model(providers, "ollama", None)


def test_a_model_can_use_another_api_than_its_provider() -> None:
    providers = merge_providers(
        {
            "gateway": ProviderOverride(
                api="openai_responses",
                base_url="http://localhost:1",
                models={"claude": ModelSpec(api="anthropic"), "gpt": ModelSpec()},
            )
        }
    )
    assert resolve_model(providers, "gateway", "claude").api == "anthropic"
    assert resolve_model(providers, "gateway", "gpt").api == "openai_responses"


def test_api_key_comes_from_config_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Model(id="m", provider="p", api="openai_chat", base_url="x", api_key_env="TEST_KEY")
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ModelError, match="TEST_KEY"):
        api_key_for(model)
    monkeypatch.setenv("TEST_KEY", "env")
    assert api_key_for(model) == "env"
    model.api_key = "config"
    assert api_key_for(model) == "config"
    assert api_key_for(Model(id="m", provider="local", api="openai_chat", base_url="x")) is None


def test_cost_uses_per_million_rates() -> None:
    model = Model(
        id="m", provider="p", api="openai_chat", base_url="x", cost=ModelCost(input=2, output=10, cache_read=0.5)
    )
    cost = compute_cost(model, Usage(input=1_000_000, output=100_000, cache_read=2_000_000))
    assert (cost.input, cost.output, cost.cache_read, cost.total) == (2.0, 1.0, 1.0, 4.0)
