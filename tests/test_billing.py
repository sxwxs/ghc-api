"""Tests for the LLM billing module (price catalog + cost computation)."""

import json
from pathlib import Path

import pytest

from ghc_api import billing
from ghc_api.cache import RequestCache


@pytest.fixture()
def catalog(tmp_path: Path) -> billing.PriceCatalog:
    return billing._reset_price_catalog_for_tests(config_dir=tmp_path)


class TestNormalizeModelId:
    def test_dots_become_dashes(self):
        assert billing.normalize_model_id("Claude-3.7-Sonnet") == "claude-3-7-sonnet"

    def test_effort_suffix_stripped(self):
        assert billing.normalize_model_id("gpt-5-mini-high") == "gpt-5-mini"

    def test_date_suffix_stripped(self):
        assert billing.normalize_model_id("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"

    def test_vendor_prefix_kept(self):
        assert billing.normalize_model_id("openai/gpt-4o") == "openai/gpt-4o"

    def test_empty(self):
        assert billing.normalize_model_id("") == ""
        assert billing.normalize_model_id(None) == ""


class TestPriceResolution:
    def test_builtin_fallback(self, catalog):
        entry = catalog.resolve("gpt-4o-2024-11-20")
        assert entry is not None
        assert entry.source == "builtin"
        assert entry.input_per_1m == 2.5
        assert entry.output_per_1m == 10.0

    def test_unpriced_returns_none(self, catalog):
        assert catalog.resolve("totally-unknown-model") is None

    def test_override_beats_builtin(self, catalog):
        catalog.set_override("gpt-4o", {"input_per_1m": 9.99, "output_per_1m": 1.0})
        entry = catalog.resolve("gpt-4o")
        assert entry.source == "manual"
        assert entry.input_per_1m == 9.99

    def test_wildcard_override(self, catalog):
        catalog.set_override("claude-opus-*", {"input_per_1m": 1.0, "output_per_1m": 2.0})
        entry = catalog.resolve("claude-opus-4-7")
        assert entry is not None
        assert entry.source == "manual"
        assert entry.input_per_1m == 1.0

    def test_auto_catalog_exact_match(self, catalog):
        catalog._auto_models["gpt-5-mega"] = {
            "input_per_1m": 1.0, "output_per_1m": 4.0,
            "cache_read_per_1m": 0.1, "cache_write_per_1m": None,
            "currency": "USD",
        }
        catalog._auto_bare_index = catalog._build_bare_index(catalog._auto_models)
        entry = catalog.resolve("gpt-5-mega-high")
        assert entry is not None
        assert entry.source == "auto"
        assert entry.input_per_1m == 1.0

    def test_auto_catalog_family_fallback(self, catalog):
        catalog._auto_models["claude-opus-4"] = {
            "input_per_1m": 15.0, "output_per_1m": 75.0,
            "cache_read_per_1m": None, "cache_write_per_1m": None,
            "currency": "USD",
        }
        catalog._auto_bare_index = catalog._build_bare_index(catalog._auto_models)
        entry = catalog.resolve("claude-opus-4-7")
        assert entry is not None
        assert entry.source == "auto"
        assert entry.approximate is True
        assert entry.matched_key == "claude-opus-4"

    def test_auto_bare_name_index(self, catalog):
        catalog._store_auto_entry(catalog._auto_models, "gemini/gemini-2.5-pro", {
            "input_per_1m": 1.25, "output_per_1m": 10.0,
            "cache_read_per_1m": None, "cache_write_per_1m": None,
            "currency": "USD",
        })
        catalog._auto_bare_index = catalog._build_bare_index(catalog._auto_models)
        entry = catalog.resolve("gemini-2.5-pro")
        assert entry is not None
        assert entry.source == "auto"

    def test_override_persistence(self, catalog, tmp_path):
        catalog.set_override("my-model", {"input_per_1m": 3.0, "output_per_1m": 4.0})
        reloaded = billing.PriceCatalog(config_dir=tmp_path)
        entry = reloaded.resolve("my-model")
        assert entry is not None
        assert entry.source == "manual"
        assert entry.input_per_1m == 3.0

    def test_delete_override(self, catalog):
        catalog.set_override("my-model", {"input_per_1m": 3.0, "output_per_1m": 4.0})
        assert catalog.delete_override("my-model") is True
        assert catalog.delete_override("my-model") is False
        assert catalog.resolve("my-model") is None

    def test_invalid_override_rejected(self, catalog):
        with pytest.raises(ValueError):
            catalog.set_override("bad", {"input_per_1m": -1, "output_per_1m": 1.0})
        with pytest.raises(ValueError):
            catalog.set_override("bad", {"input_per_1m": 1.0, "output_per_1m": 1.0, "currency": "EUR"})


class TestConvertLitellmEntry:
    def test_converts_per_token_to_per_1m(self, catalog):
        converted = catalog._convert_litellm_entry({
            "mode": "chat",
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "cache_read_input_token_cost": 0.00000125,
            "cache_creation_input_token_cost": 0.000003125,
        })
        assert converted["input_per_1m"] == pytest.approx(2.5)
        assert converted["output_per_1m"] == pytest.approx(10.0)
        assert converted["cache_read_per_1m"] == pytest.approx(1.25)
        assert converted["cache_write_per_1m"] == pytest.approx(3.125)

    def test_skips_non_chat_modes(self, catalog):
        assert catalog._convert_litellm_entry({
            "mode": "audio_transcription",
            "input_cost_per_token": 1,
            "output_cost_per_token": 1,
        }) is None

    def test_skips_missing_costs(self, catalog):
        assert catalog._convert_litellm_entry({"mode": "chat"}) is None


class TestComputeCost:
    def test_exact_math_with_override(self, catalog):
        catalog.set_override("test-model", {
            "input_per_1m": 2.0,
            "output_per_1m": 10.0,
            "cache_read_per_1m": 1.0,
            "cache_write_per_1m": 2.0,
        })
        result = billing.compute_cost(
            "test-model",
            input_tokens=1000,
            output_tokens=2000,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=100,
            catalog=catalog,
        )
        # micros = tokens * price_per_1m: 2000 + 20000 + 500*2 + 100*1 = 23100
        assert result["cost_micro_usd"] == 23100
        assert result["unpriced"] is False
        assert result["currency"] == "USD"

    def test_missing_cache_prices_fall_back_to_input(self, catalog):
        catalog.set_override("test-model", {"input_per_1m": 2.0, "output_per_1m": 10.0})
        result = billing.compute_cost(
            "test-model",
            input_tokens=1000,
            output_tokens=1000,
            cache_creation_input_tokens=1000,
            cache_read_input_tokens=1000,
            catalog=catalog,
        )
        # 16000 = 2000 (input) + 10000 (output) + 1000*2 + 1000*2 (cache prices default to input price)
        assert result["cost_micro_usd"] == 16000

    def test_unpriced_model(self, catalog):
        result = billing.compute_cost("nope", input_tokens=1000, catalog=catalog)
        assert result["unpriced"] is True
        assert result["cost_micro_usd"] == 0
        assert result["price"] is None

    def test_zero_tokens_cost_zero(self, catalog):
        catalog.set_override("test-model", {"input_per_1m": 2.0, "output_per_1m": 10.0})
        result = billing.compute_cost("test-model", catalog=catalog)
        assert result["cost_micro_usd"] == 0


def _write_usage_file(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "token_usage.jl"
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


class TestBillingOverview:
    @pytest.fixture()
    def patched_usage_files(self, tmp_path, monkeypatch):
        usage_file = _write_usage_file(tmp_path, [
            {
                "timestamp": 1_700_000_000,
                "user_id": "alice",
                "models": [
                    {"provider": "copilot", "model": "gpt-4o", "request_count": 2,
                     "input_tokens": 1_000_000, "output_tokens": 100_000,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                    {"provider": "configured_proxy", "model": "gpt-4o", "request_count": 1,
                     "input_tokens": 999_999_999, "output_tokens": 0,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                ],
            },
            {
                "timestamp": 1_700_000_001,
                "user_id": "bob",
                "models": [
                    # Legacy line without provider -> treated as copilot.
                    {"model": "mystery-model", "request_count": 1,
                     "input_tokens": 10, "output_tokens": 10,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                ],
            },
        ])
        monkeypatch.setattr(
            "ghc_api.token_usage_reporter._resolve_usage_files",
            lambda: [("test-machine", usage_file)],
        )
        return usage_file

    def test_only_copilot_provider_billed(self, patched_usage_files, catalog):
        overview = billing.get_billing_overview("all", refresh_catalog=False)
        assert overview["provider"] == "copilot"

        alice = next(r for r in overview["rows"] if r["user_id"] == "alice")
        assert alice["model_id"] == "gpt-4o"
        # The configured_proxy row (999999999 input tokens) must be excluded.
        assert alice["input_tokens"] == 1_000_000

        bob = next(r for r in overview["rows"] if r["user_id"] == "bob")
        assert bob["model_id"] == "mystery-model"
        assert bob["unpriced"] is True

    def test_costs_computed_and_totals(self, patched_usage_files, catalog):
        overview = billing.get_billing_overview("all", refresh_catalog=False)
        gpt4o = next(r for r in overview["model_rows"] if r["model_id"] == "gpt-4o")
        # 1M input at $2.5/1M = 2_500_000 micros + 100k output at $10/1M = 1_000_000 micros
        assert gpt4o["cost_micro_usd"] == 3_500_000
        assert gpt4o["cost_usd"] == pytest.approx(3.5)
        assert gpt4o["price_source"] == "builtin"

        assert overview["totals"]["input_tokens"] == 1_000_010
        assert overview["totals"]["output_tokens"] == 100_010
        assert overview["unpriced_models"] == ["mystery-model"]

    def test_user_filter(self, patched_usage_files, catalog):
        overview = billing.get_billing_overview("all", user_filter="alice", refresh_catalog=False)
        assert [r["user_id"] for r in overview["rows"]] == ["alice"]

    def test_range_cutoff(self, patched_usage_files, catalog, monkeypatch):
        monkeypatch.setattr(billing.time, "time", lambda: 1_700_000_000 + 7200)
        overview = billing.get_billing_overview("hour", refresh_catalog=False)
        assert overview["rows"] == []

    def test_multiple_deltas_are_each_priced(self, tmp_path, catalog, monkeypatch):
        """Regression: per-line costs must use each line's own token counts,
        not reuse the cost computed from the first delta line."""
        usage_file = _write_usage_file(tmp_path, [
            {
                "timestamp": 1_700_000_000,
                "user_id": "alice",
                "models": [
                    {"provider": "copilot", "model": "gpt-4o", "request_count": 1,
                     "input_tokens": 1_000_000, "output_tokens": 0,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                ],
            },
            {
                "timestamp": 1_700_000_300,
                "user_id": "alice",
                "models": [
                    {"provider": "copilot", "model": "gpt-4o", "request_count": 1,
                     "input_tokens": 3_000_000, "output_tokens": 0,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                ],
            },
        ])
        monkeypatch.setattr(
            "ghc_api.token_usage_reporter._resolve_usage_files",
            lambda: [("m", usage_file)],
        )
        overview = billing.get_billing_overview("all", refresh_catalog=False)
        # gpt-4o input at $2.5/1M: line 1 = 2.5 USD, line 2 = 7.5 USD.
        row = overview["model_rows"][0]
        assert row["input_tokens"] == 4_000_000
        assert row["cost_micro_usd"] == 10_000_000
        assert overview["totals"]["cost_micro_usd"] == 10_000_000


class TestReadSideNormalization:
    """Pre-fix usage rows (no provider field) must not double-count cached
    tokens that are already included in input_tokens."""

    @pytest.fixture()
    def patched_usage_files(self, tmp_path, monkeypatch):
        usage_file = _write_usage_file(tmp_path, [
            {
                # Pre-fix OpenAI-style row: input INCLUDES the cached reads.
                "timestamp": 1_700_000_000,
                "user_id": "alice",
                "models": [
                    {"model": "gpt-x", "request_count": 1,
                     "input_tokens": 2_000_000, "output_tokens": 100_000,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_500_000},
                ],
            },
            {
                # Post-fix row (has provider field): input is already uncached.
                "timestamp": 1_700_000_001,
                "user_id": "alice",
                "models": [
                    {"provider": "copilot", "model": "gpt-x", "request_count": 1,
                     "input_tokens": 500_000, "output_tokens": 100_000,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_500_000},
                ],
            },
            {
                # Pre-fix Claude row: anthropic semantics, input already uncached.
                "timestamp": 1_700_000_002,
                "user_id": "alice",
                "models": [
                    {"model": "claude-x", "request_count": 1,
                     "input_tokens": 300_000, "output_tokens": 10_000,
                     "cache_creation_input_tokens": 50_000, "cache_read_input_tokens": 200_000},
                ],
            },
        ])
        monkeypatch.setattr(
            "ghc_api.token_usage_reporter._resolve_usage_files",
            lambda: [("m", usage_file)],
        )

    def test_prefix_rows_not_double_counted(self, patched_usage_files, catalog):
        catalog.set_override("gpt-x", {
            "input_per_1m": 1.0, "output_per_1m": 10.0, "cache_read_per_1m": 0.1,
        })
        catalog.set_override("claude-x", {
            "input_per_1m": 1.0, "output_per_1m": 10.0,
            "cache_read_per_1m": 0.1, "cache_write_per_1m": 1.0,
        })
        overview = billing.get_billing_overview("all", refresh_catalog=False)
        rows = {r["model_id"]: r for r in overview["model_rows"]}

        # Pre-fix gpt-x normalized to uncached = 2M - 1.5M = 0.5M; the post-fix
        # row (already uncached) adds another 0.5M -> 1.0M aggregated.
        # Per-row cost = 0.5*1 + 1.5*0.1 + 0.1*10 = 1.65 USD, two rows = 3.3 USD.
        assert rows["gpt-x"]["input_tokens"] == 1_000_000
        assert rows["gpt-x"]["cache_read_input_tokens"] == 3_000_000
        assert rows["gpt-x"]["cost_micro_usd"] == 3_300_000

        # Post-fix row keeps its own semantics; totals aggregate both rows.
        assert rows["gpt-x"]["output_tokens"] == 200_000

        # Claude pre-fix row: input already uncached, create stays a write.
        cl = rows["claude-x"]
        assert cl["input_tokens"] == 300_000
        assert cl["cache_creation_input_tokens"] == 50_000
        assert cl["cache_read_input_tokens"] == 200_000
        # 0.3*1 + 0.05*1 + 0.2*0.1 + 0.01*10 = 0.47 USD
        assert cl["cost_micro_usd"] == 470_000


class TestCacheProviderStats:
    def test_default_provider_is_copilot(self):
        cache = RequestCache()
        cache.start_request("r1", {"model": "gpt-4o", "user_id": "alice"})
        cache.complete_request("r1", {
            "model": "gpt-4o", "input_tokens": 10, "output_tokens": 5,
            "request_size": 1, "response_size": 2, "user_id": "alice",
        })
        snapshot = cache.get_user_provider_model_token_snapshot()
        assert ("alice", "copilot", "gpt-4o") in snapshot
        assert snapshot[("alice", "copilot", "gpt-4o")]["input_tokens"] == 10

    def test_upstream_provider_isolated(self):
        cache = RequestCache()
        cache.start_request("r1", {"model": "gpt-4o", "user_id": "alice"})
        cache.complete_request("r1", {
            "model": "gpt-4o", "input_tokens": 10, "output_tokens": 5,
            "request_size": 1, "response_size": 2, "user_id": "alice",
            "upstream_provider": "configured_proxy",
        })
        snapshot = cache.get_user_provider_model_token_snapshot()
        assert ("alice", "configured_proxy", "gpt-4o") in snapshot
        assert ("alice", "copilot", "gpt-4o") not in snapshot
