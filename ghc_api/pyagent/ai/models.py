"""Models and providers: built-in provider presets, user overrides, and resolution to a flat `Model`."""

import os
from typing import Literal

import msgspec

from .types import Cost, Usage, Wire

ApiName = Literal["openai_chat", "openai_responses", "anthropic"]
# How the thinking level reaches openai_chat and anthropic endpoints; openai_responses always uses reasoning.effort.
ThinkingFormat = Literal["none", "reasoning_effort", "deepseek", "openrouter", "anthropic_adaptive", "anthropic_budget"]


class ModelCost(Wire):
    """Price in USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class ModelSpec(Wire):
    """Per-model settings. Every field has a usable default, so unlisted models still work."""

    context_window: int = 200_000
    max_tokens: int = 32_000
    reasoning: bool = False
    images: bool = False
    cost: ModelCost = msgspec.field(default_factory=ModelCost)
    thinking_format: ThinkingFormat | None = None  # overrides the provider's format
    # Overrides the provider's api, for gateways that serve each model on its own endpoint from one base URL.
    api: ApiName | None = None


class ProviderConfig(Wire):
    """One provider: endpoint, credentials and known models."""

    api: ApiName
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = {}
    thinking_format: ThinkingFormat = "none"
    # Send reasoning text back on assistant messages (DeepSeek and Kimi require it during tool use).
    replay_reasoning: bool = False
    default_model: str | None = None
    models: dict[str, ModelSpec] = {}


class ProviderOverride(Wire):
    """User configuration for a provider: overrides a preset field by field, or defines a new provider."""

    api: ApiName | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    headers: dict[str, str] | None = None
    thinking_format: ThinkingFormat | None = None
    replay_reasoning: bool | None = None
    default_model: str | None = None
    models: dict[str, ModelSpec] = {}


class Model(Wire):
    """A resolved model: everything a provider adapter needs, flattened."""

    id: str
    provider: str
    api: ApiName
    base_url: str
    headers: dict[str, str] = {}
    api_key_env: str | None = None
    api_key: str | None = None
    thinking_format: ThinkingFormat = "none"
    replay_reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 32_000
    reasoning: bool = False
    images: bool = False
    cost: ModelCost = msgspec.field(default_factory=ModelCost)


class ModelError(Exception):
    """A model or provider cannot be resolved or has no credentials."""


# Endpoints and default models follow pi's catalog (packages/ai/catalog). Model details not listed fall back
# to ModelSpec defaults; users override them in config.json.
BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        api="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        thinking_format="anthropic_adaptive",
        default_model="claude-opus-5-5",
        models={
            "claude-opus-5-5": ModelSpec(reasoning=True, images=True),
            "claude-sonnet-5-5": ModelSpec(reasoning=True, images=True),
            "claude-haiku-5-5": ModelSpec(reasoning=True, images=True),
        },
    ),
    "openai": ProviderConfig(
        api="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-6.1-sol",
        models={"gpt-6.1-sol": ModelSpec(context_window=922_000, max_tokens=128_000, reasoning=True, images=True)},
    ),
    "deepseek": ProviderConfig(
        api="openai_chat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        thinking_format="deepseek",
        replay_reasoning=True,
        default_model="deepseek-v4-pro",
        models={
            "deepseek-v4-pro": ModelSpec(
                context_window=1_000_000,
                max_tokens=384_000,
                reasoning=True,
                cost=ModelCost(input=1.32, output=3.96, cache_read=0.044),
            ),
            "deepseek-flash": ModelSpec(
                context_window=1_000_000,
                max_tokens=384_000,
                reasoning=True,
                images=True,
                cost=ModelCost(input=0.3, output=1.2, cache_read=0.006),
            ),
        },
    ),
    "moonshot": ProviderConfig(
        api="openai_chat",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        replay_reasoning=True,
        default_model="kimi-k2.7-code",
    ),
    "openrouter": ProviderConfig(
        api="openai_chat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        thinking_format="openrouter",
        default_model="moonshotai/kimi-k2.7-code",
    ),
    "ollama": ProviderConfig(api="openai_chat", base_url="http://localhost:11434/v1"),
}


def merge_providers(overrides: dict[str, ProviderOverride]) -> dict[str, ProviderConfig]:
    """Apply user overrides to the built-in presets; overrides for unknown names define new providers."""
    providers = dict(BUILTIN_PROVIDERS)
    for name, override in overrides.items():
        base = providers.get(name)
        if base is None:
            if override.api is None or override.base_url is None:
                raise ModelError(f"Provider '{name}' is not built in; config must set api and baseUrl")
            base = ProviderConfig(api=override.api, base_url=override.base_url)
        changes = {
            field: value
            for field in ProviderOverride.__struct_fields__
            if field != "models" and (value := getattr(override, field)) is not None
        }
        providers[name] = msgspec.structs.replace(base, **changes, models={**base.models, **override.models})
    return providers


def resolve_model(providers: dict[str, ProviderConfig], provider: str, model_id: str | None) -> Model:
    """Resolve a provider name and optional model id to a `Model`; the provider's default model fills a missing id."""
    config = providers.get(provider)
    if config is None:
        raise ModelError(f"Unknown provider '{provider}'. Known: {', '.join(sorted(providers))}")
    model_id = model_id or config.default_model
    if model_id is None:
        raise ModelError(f"Provider '{provider}' has no default model; pass --model")
    spec = config.models.get(model_id, ModelSpec())
    return Model(
        id=model_id,
        provider=provider,
        api=spec.api or config.api,
        base_url=config.base_url,
        headers=config.headers,
        api_key_env=config.api_key_env,
        api_key=config.api_key,
        thinking_format=spec.thinking_format or config.thinking_format,
        replay_reasoning=config.replay_reasoning,
        context_window=spec.context_window,
        max_tokens=spec.max_tokens,
        reasoning=spec.reasoning,
        images=spec.images,
        cost=spec.cost,
    )


def api_key_for(model: Model) -> str | None:
    """Return the configured key, else the provider's environment variable. Keyless providers return None."""
    if model.api_key:
        return model.api_key
    if model.api_key_env is None:
        return None
    key = os.environ.get(model.api_key_env)
    if not key:
        raise ModelError(
            f"No API key for '{model.provider}': set {model.api_key_env} or providers.{model.provider}.apiKey"
        )
    return key


def has_credentials(config: ProviderConfig) -> bool:
    """Whether a provider can be used without further setup."""
    return bool(config.api_key) or (config.api_key_env is not None and bool(os.environ.get(config.api_key_env)))


def compute_cost(model: Model, usage: Usage) -> Cost:
    """Price `usage` with the model's per-million rates."""
    rates = model.cost
    cost = Cost(
        input=usage.input * rates.input / 1e6,
        output=usage.output * rates.output / 1e6,
        cache_read=usage.cache_read * rates.cache_read / 1e6,
        cache_write=usage.cache_write * rates.cache_write / 1e6,
    )
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
    return cost
