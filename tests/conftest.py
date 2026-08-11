"""Shared test fixtures.

The suite exercises code paths that write to the config directory (request
JSONL dumps, error.log, the Web IQ search log). Without an override those
writes land in the developer's real ~/.ghc-api, mixing fake test records into
production data. Every test therefore runs against a throwaway config dir.
Tests that need their own location still override GHC_API_CONFIG_DIR locally.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolate_config_dir(tmp_path_factory):
    """Point GHC_API_CONFIG_DIR at a temporary directory for the whole session."""
    original = os.environ.get("GHC_API_CONFIG_DIR")
    os.environ["GHC_API_CONFIG_DIR"] = str(tmp_path_factory.mktemp("ghc-api-config"))
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("GHC_API_CONFIG_DIR", None)
        else:
            os.environ["GHC_API_CONFIG_DIR"] = original
