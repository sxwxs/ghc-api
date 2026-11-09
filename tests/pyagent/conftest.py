"""Shared fixtures: the asyncio backend for anyio tests and a model for scripted providers."""

import pytest

from ghc_api.pyagent.ai.models import Model


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def model() -> Model:
    """A keyless model; tests pair it with a FauxProvider or a mocked HTTP transport."""
    return Model(id="test-model", provider="test", api="openai_chat", base_url="http://test.local/v1")
