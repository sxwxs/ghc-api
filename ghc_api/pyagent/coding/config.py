"""User configuration in <home>/config.json (home: PYAGENT_HOME, else ~/.pyagent) and model selection."""

import os
from pathlib import Path

import msgspec

from ..ai.models import Model, ModelError, ProviderConfig, ProviderOverride, has_credentials, resolve_model
from ..ai.types import ThinkingLevel, Wire


class Config(Wire):
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking: ThinkingLevel | None = None
    providers: dict[str, ProviderOverride] = {}


class ConfigError(Exception):
    """The configuration file cannot be read."""


def home_dir() -> Path:
    """The pyagent home directory."""
    return Path(os.environ.get("PYAGENT_HOME") or Path.home() / ".pyagent")


def load_config(home: Path) -> Config:
    """Read config.json; a missing file means defaults."""
    path = home / "config.json"
    if not path.is_file():
        return Config()
    try:
        return msgspec.json.decode(path.read_bytes(), type=Config)
    except msgspec.DecodeError as error:
        raise ConfigError(f"{path}: {error}") from error


def select_model(
    config: Config, providers: dict[str, ProviderConfig], provider: str | None, model_id: str | None
) -> Model:
    """Pick the model: command-line flags, then config defaults, then the first built-in provider with credentials."""
    provider = provider or config.default_provider
    if provider is None:
        provider = next((name for name, entry in providers.items() if has_credentials(entry)), None)
        if provider is None:
            raise ModelError(
                "No provider configured. Set an API key such as DEEPSEEK_API_KEY or ANTHROPIC_API_KEY, "
                "or set defaultProvider in config.json"
            )
    if model_id is None and provider == config.default_provider:
        model_id = config.default_model
    return resolve_model(providers, provider, model_id)
