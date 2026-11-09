"""LLM billing: price catalog resolution and cost computation.

Phase 1 scope: cost accounting for Copilot API traffic only, computed at read
time from the token-usage JSONL files (no new write path).

Prices resolve through layered fallbacks (highest priority first):

1. Manual overrides   -- ``<config_dir>/llm_pricing_overrides.json`` (user
   editable, also managed via the dashboard API; never touched by auto fetch).
2. Auto catalog cache -- ``<config_dir>/llm_pricing_cache.json`` fetched from
   the LiteLLM community price table with a TTL; stale entries are used when
   the fetch fails so billing never depends on network availability.
3. Builtin defaults   -- a small table of well-known public list prices.

A model that matches nothing is reported as ``unpriced`` instead of silently
costing zero.

Cost math uses integer micro-USD (1 USD = 1e6 micros). For a price expressed
in USD per 1M tokens, ``micros = tokens * price_per_1m`` (rounded half-up),
so per-row costs aggregate exactly.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .utils import get_config_dir

# Community-maintained price table (USD per token fields).
PRICE_SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
PRICE_REFRESH_TTL_SECONDS = 24 * 60 * 60
PRICE_FETCH_TIMEOUT_SECONDS = 15
PRICE_FAILURE_RETRY_SECONDS = 10 * 60

OVERRIDES_FILENAME = "llm_pricing_overrides.json"
CACHE_FILENAME = "llm_pricing_cache.json"

DEFAULT_CURRENCY = "USD"
MICROS_PER_USD = 1_000_000

# Trailing tokens that encode reasoning-effort variants rather than a distinct
# model (e.g. "gpt-5-mini-high"), stripped before catalog lookups.
_EFFORT_SUFFIXES = {"high", "low", "medium", "minimal", "xhigh"}
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Well-known public list prices (USD per 1M tokens) used when neither the
# overrides nor the auto catalog can resolve a model. Deliberately small:
# the auto catalog is the real workhorse, these only soften first-run gaps.
BUILTIN_PRICES: Dict[str, Dict[str, Optional[float]]] = {
    "gpt-4o": {"input_per_1m": 2.5, "output_per_1m": 10.0, "cache_read_per_1m": 1.25},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6, "cache_read_per_1m": 0.075},
    "gpt-5": {"input_per_1m": 1.25, "output_per_1m": 10.0, "cache_read_per_1m": 0.125},
    "gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.0, "cache_read_per_1m": 0.025},
    "gpt-5-nano": {"input_per_1m": 0.05, "output_per_1m": 0.4, "cache_read_per_1m": 0.005},
    "o3": {"input_per_1m": 2.0, "output_per_1m": 8.0, "cache_read_per_1m": 0.5},
    "o4-mini": {"input_per_1m": 1.1, "output_per_1m": 4.4, "cache_read_per_1m": 0.275},
    "claude-3-5-haiku": {"input_per_1m": 0.8, "output_per_1m": 4.0, "cache_write_per_1m": 1.0, "cache_read_per_1m": 0.08},
    "claude-3-5-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0, "cache_write_per_1m": 3.75, "cache_read_per_1m": 0.3},
    "claude-3-7-sonnet": {"input_per_1m": 3.0, "output_per_1m": 15.0, "cache_write_per_1m": 3.75, "cache_read_per_1m": 0.3},
    "claude-haiku-4": {"input_per_1m": 0.8, "output_per_1m": 4.0, "cache_write_per_1m": 1.0, "cache_read_per_1m": 0.08},
    "claude-sonnet-4": {"input_per_1m": 3.0, "output_per_1m": 15.0, "cache_write_per_1m": 3.75, "cache_read_per_1m": 0.3},
    "claude-opus-4": {"input_per_1m": 15.0, "output_per_1m": 75.0, "cache_write_per_1m": 18.75, "cache_read_per_1m": 1.5},
    "gemini-2.5-pro": {"input_per_1m": 1.25, "output_per_1m": 10.0},
    "gemini-2.5-flash": {"input_per_1m": 0.3, "output_per_1m": 2.5},
}

_PRICE_FIELDS = ("input_per_1m", "output_per_1m", "cache_read_per_1m", "cache_write_per_1m")
_REQUIRED_PRICE_FIELDS = ("input_per_1m", "output_per_1m")


@dataclass(frozen=True)
class PriceEntry:
    """A resolved price, in USD per 1M tokens."""

    input_per_1m: Optional[float]
    output_per_1m: Optional[float]
    cache_read_per_1m: Optional[float]
    cache_write_per_1m: Optional[float]
    currency: str
    source: str          # "manual" | "auto" | "builtin"
    matched_key: str     # the catalog/override key that matched
    approximate: bool = False  # True when matched via family fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_per_1m": self.input_per_1m,
            "output_per_1m": self.output_per_1m,
            "cache_read_per_1m": self.cache_read_per_1m,
            "cache_write_per_1m": self.cache_write_per_1m,
            "currency": self.currency,
            "source": self.source,
            "matched_key": self.matched_key,
            "approximate": self.approximate,
        }


def normalize_model_id(model: str) -> str:
    """Normalize a model id for catalog lookups.

    Lowercase, dots to dashes (Copilot uses "claude-3.7-sonnet", catalogs use
    "claude-3-7-sonnet"), collapse repeated dashes, and strip reasoning-effort
    and date suffixes.
    """
    text = (model or "").strip().lower()
    text = text.replace(".", "-")
    text = re.sub(r"-+", "-", text).strip("-")
    while True:
        if _DATE_SUFFIX_RE.search(text):
            text = _DATE_SUFFIX_RE.sub("", text)
            continue
        base, _, last = text.rpartition("-")
        if last in _EFFORT_SUFFIXES and base:
            text = base
            continue
        return text


def _bare_model_name(normalized: str) -> str:
    """Strip a vendor prefix such as "openai/" or "gemini/" if present."""
    return normalized.rpartition("/")[2] or normalized


def _valid_price_map(raw: Any) -> Optional[Dict[str, Optional[float]]]:
    """Validate a price map; None if invalid.

    Input and output prices are required (0 is a valid, intentionally free
    price) so an incomplete entry can never silently make a model free.
    Cache prices are optional and fall back to the input price.
    """
    if not isinstance(raw, dict):
        return None
    cleaned: Dict[str, Optional[float]] = {}
    for field in _PRICE_FIELDS:
        value = raw.get(field)
        if value is None:
            if field in _REQUIRED_PRICE_FIELDS:
                return None
            cleaned[field] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        cleaned[field] = number
    currency = raw.get("currency") or DEFAULT_CURRENCY
    if not isinstance(currency, str) or currency.upper() != DEFAULT_CURRENCY:
        return None
    cleaned["currency"] = DEFAULT_CURRENCY
    return cleaned


class PriceCatalog:
    """Layered price lookup with auto-fetch and manual overrides."""

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else Path(get_config_dir())
        self._lock = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._overrides: Dict[str, Dict[str, Optional[float]]] = {}
        # Lookup index: non-wildcard override keys mapped through
        # normalize_model_id, so a stored "gpt-5.6-sol" matches a lookup for
        # "gpt-5-6-sol" (and vice versa). Wildcard patterns stay raw.
        self._override_index: Dict[str, Dict[str, Optional[float]]] = {}
        self._auto_models: Dict[str, Dict[str, Optional[float]]] = {}
        self._auto_bare_index: Dict[str, str] = {}
        self._auto_meta: Dict[str, Any] = {}
        self._last_fetch_failure_ts: float = 0.0
        self._load_overrides()
        self._load_auto_cache()

    # ------------------------------------------------------------- paths

    @property
    def overrides_path(self) -> Path:
        return self._config_dir / OVERRIDES_FILENAME

    @property
    def cache_path(self) -> Path:
        return self._config_dir / CACHE_FILENAME

    # ---------------------------------------------------------- overrides

    def _load_overrides(self) -> None:
        try:
            raw = json.loads(self.overrides_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        models = raw.get("models") if isinstance(raw, dict) else raw
        loaded: Dict[str, Dict[str, Optional[float]]] = {}
        if isinstance(models, dict):
            for key, value in models.items():
                cleaned = _valid_price_map(value)
                if cleaned is not None and isinstance(key, str) and key.strip():
                    loaded[key.strip()] = cleaned
        with self._lock:
            self._overrides = loaded
            self._rebuild_override_index()

    def _rebuild_override_index(self) -> None:
        """Caller must hold self._lock."""
        self._override_index = {
            normalize_model_id(key): prices
            for key, prices in self._overrides.items()
            if not any(c in key for c in "*?[")
        }

    def _save_overrides(self) -> None:
        with self._lock:
            payload = {"models": self._overrides}
            self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
            self.overrides_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def set_override(self, model: str, prices: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = _valid_price_map(prices)
        if cleaned is None:
            raise ValueError(
                "input_per_1m and output_per_1m are required; all prices must be "
                f"finite non-negative numbers (fields {list(_PRICE_FIELDS)}, "
                f"currency {DEFAULT_CURRENCY})"
            )
        key = (model or "").strip()
        if not key:
            raise ValueError("model is required")
        with self._lock:
            self._overrides[key] = cleaned
            self._rebuild_override_index()
            self._save_overrides()
        return dict(cleaned)

    def delete_override(self, model: str) -> bool:
        key = (model or "").strip()
        with self._lock:
            if key in self._overrides:
                del self._overrides[key]
                self._rebuild_override_index()
                self._save_overrides()
                return True
        return False

    def list_overrides(self) -> Dict[str, Dict[str, Optional[float]]]:
        with self._lock:
            return {k: dict(v) for k, v in self._overrides.items()}

    # -------------------------------------------------------- auto catalog

    @staticmethod
    def _store_auto_entry(models: Dict[str, Dict[str, Optional[float]]], key: str, converted: Dict[str, Optional[float]]) -> None:
        """Insert a converted catalog entry under its normalized key.

        Normalization matches the lookup side (dots to dashes, effort/date
        suffixes stripped); on collision the first sorted key wins so the
        catalog stays deterministic.
        """
        normalized = normalize_model_id(key)
        if normalized and normalized not in models:
            models[normalized] = converted

    def _load_auto_cache(self) -> None:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        models = raw.get("models") if isinstance(raw, dict) else None
        loaded: Dict[str, Dict[str, Optional[float]]] = {}
        if isinstance(models, dict):
            for key, value in models.items():
                cleaned = _valid_price_map(value)
                if cleaned is not None and isinstance(key, str):
                    self._store_auto_entry(loaded, key, cleaned)
        with self._lock:
            self._auto_models = loaded
            self._auto_bare_index = self._build_bare_index(loaded)
            if isinstance(raw, dict):
                self._auto_meta = {
                    "fetched_at": raw.get("fetched_at"),
                    "source": raw.get("source"),
                }

    @staticmethod
    def _build_bare_index(models: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, str]:
        """Map bare model names ("gpt-4o") to a preferred full catalog key.

        Vendored keys ("azure/gpt-4o") are only chosen when no bare key exists
        for the same name; ties keep the first key in sorted order.
        """
        bare_index: Dict[str, str] = {}
        for key in sorted(models):
            bare = _bare_model_name(key)
            current = bare_index.get(bare)
            if current is None or ("/" in current and "/" not in key):
                bare_index[bare] = key
        return bare_index

    def _convert_litellm_entry(self, entry: Any) -> Optional[Dict[str, Optional[float]]]:
        if not isinstance(entry, dict):
            return None
        mode = entry.get("mode")
        if mode not in ("chat", "embedding"):
            return None
        try:
            input_per_token = float(entry.get("input_cost_per_token"))
            output_per_token = float(entry.get("output_cost_per_token"))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) and v >= 0 for v in (input_per_token, output_per_token)):
            return None

        def per_1m(value: Any) -> Optional[float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number * MICROS_PER_USD if math.isfinite(number) and number >= 0 else None

        return {
            "input_per_1m": input_per_token * MICROS_PER_USD,
            "output_per_1m": output_per_token * MICROS_PER_USD,
            "cache_read_per_1m": per_1m(entry.get("cache_read_input_token_cost")),
            "cache_write_per_1m": per_1m(entry.get("cache_creation_input_token_cost")),
            "currency": DEFAULT_CURRENCY,
        }

    def refresh(self, force: bool = False) -> Dict[str, Any]:
        """Fetch the auto catalog. Never raises; returns fetch status."""
        now = time.time()
        with self._lock:
            fetched_at = self._auto_meta.get("fetched_at") or 0
            stale = not fetched_at or (now - float(fetched_at)) >= PRICE_REFRESH_TTL_SECONDS
        if not force and not stale:
            return {"ok": True, "skipped": True, "fetched_at": fetched_at}
        if not force and (now - self._last_fetch_failure_ts) < PRICE_FAILURE_RETRY_SECONDS:
            return {"ok": False, "skipped": True, "reason": "recent failure; retry later"}

        acquired = self._fetch_lock.acquire(blocking=False)
        if not acquired:
            return {"ok": True, "skipped": True, "reason": "refresh already in progress"}
        try:
            return self._fetch_and_store()
        finally:
            self._fetch_lock.release()

    def _fetch_and_store(self) -> Dict[str, Any]:
        """Fetch, convert and persist the auto catalog. Caller holds _fetch_lock."""
        try:
            response = requests.get(PRICE_SOURCE_URL, timeout=PRICE_FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            self._last_fetch_failure_ts = time.time()
            return {"ok": False, "error": str(e)}

        models: Dict[str, Dict[str, Optional[float]]] = {}
        if isinstance(payload, dict):
            for key, entry in sorted(payload.items(), key=lambda item: str(item[0])):
                if not isinstance(key, str):
                    continue
                converted = self._convert_litellm_entry(entry)
                if converted is not None:
                    self._store_auto_entry(models, key, converted)

        if not models:
            # Keep the previous (possibly stale) catalog and retry on the
            # failure interval rather than waiting a full TTL.
            self._last_fetch_failure_ts = time.time()
            return {"ok": False, "error": "price source returned no usable entries"}

        with self._lock:
            self._auto_models = models
            self._auto_bare_index = self._build_bare_index(models)
            self._auto_meta = {"fetched_at": int(time.time()), "source": PRICE_SOURCE_URL}
            self._last_fetch_failure_ts = 0.0
            meta = dict(self._auto_meta)
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(
                        {"fetched_at": meta["fetched_at"], "source": meta["source"], "models": models},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                meta["persist_error"] = str(e)
        return {"ok": True, "count": len(models), **meta}

    def ensure_fresh(self) -> Dict[str, Any]:
        """Best-effort TTL refresh; safe to call on every billing read."""
        try:
            return self.refresh(force=False)
        except Exception as e:  # defensive: billing must never raise
            return {"ok": False, "error": str(e)}

    def catalog_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source": self._auto_meta.get("source") or PRICE_SOURCE_URL,
                "source_url": PRICE_SOURCE_URL,
                "fetched_at": self._auto_meta.get("fetched_at"),
                "model_count": len(self._auto_models),
                "ttl_seconds": PRICE_REFRESH_TTL_SECONDS,
                "overrides_path": str(self.overrides_path),
                "cache_path": str(self.cache_path),
            }

    # ------------------------------------------------------------ lookup

    def _lookup_auto(self, normalized: str) -> Optional[Tuple[Dict[str, Optional[float]], str, bool]]:
        with self._lock:
            models = self._auto_models
            bare_index = self._auto_bare_index
        if not models:
            return None

        direct = models.get(normalized)
        if direct is not None:
            return direct, normalized, False

        bare = _bare_model_name(normalized)
        bare_key = bare_index.get(bare)
        if bare_key is not None:
            return models[bare_key], bare_key, False

        # Family fallback: drop trailing components ("claude-opus-4-7" ->
        # "claude-opus-4" -> "claude-opus") so unseen point releases still
        # resolve to the closest known family price.
        parts = normalized.split("-")
        while len(parts) > 1:
            parts.pop()
            family = "-".join(parts)
            family_key = models.get(family) or (
                models[bare_index[family]] if family in bare_index else None
            )
            if family_key is not None:
                matched = family if models.get(family) else bare_index[family]
                return family_key, matched, True
        return None

    def _match_override(self, normalized: str) -> Optional[Tuple[Dict[str, Optional[float]], str]]:
        with self._lock:
            overrides = dict(self._overrides)
            index = dict(self._override_index)
        exact = index.get(normalized) or index.get(_bare_model_name(normalized))
        if exact is not None:
            return exact, normalized
        # Wildcard patterns (raw keys), longest pattern wins for determinism.
        pattern_hits = [p for p in overrides if any(c in p for c in "*?[")]
        for pattern in sorted(pattern_hits, key=len, reverse=True):
            try:
                if fnmatch.fnmatchcase(normalized, pattern.lower()):
                    return overrides[pattern], pattern
            except Exception:
                continue
        return None

    def _match_builtin(self, normalized: str) -> Optional[Tuple[Dict[str, Optional[float]], str]]:
        if normalized in BUILTIN_PRICES:
            return dict(BUILTIN_PRICES[normalized]), normalized
        bare = _bare_model_name(normalized)
        if bare in BUILTIN_PRICES:
            return dict(BUILTIN_PRICES[bare]), bare
        parts = normalized.split("-")
        while len(parts) > 1:
            parts.pop()
            family = "-".join(parts)
            if family in BUILTIN_PRICES:
                return dict(BUILTIN_PRICES[family]), family
        return None

    def resolve(self, model: str) -> Optional[PriceEntry]:
        """Resolve the effective price for a model id, or None if unpriced."""
        normalized = normalize_model_id(model)
        if not normalized:
            return None

        override = self._match_override(normalized)
        if override is not None:
            prices, key = override
            return PriceEntry(
                input_per_1m=prices.get("input_per_1m"),
                output_per_1m=prices.get("output_per_1m"),
                cache_read_per_1m=prices.get("cache_read_per_1m"),
                cache_write_per_1m=prices.get("cache_write_per_1m"),
                currency=prices.get("currency") or DEFAULT_CURRENCY,
                source="manual",
                matched_key=key,
            )

        auto = self._lookup_auto(normalized)
        if auto is not None:
            prices, key, approximate = auto
            return PriceEntry(
                input_per_1m=prices.get("input_per_1m"),
                output_per_1m=prices.get("output_per_1m"),
                cache_read_per_1m=prices.get("cache_read_per_1m"),
                cache_write_per_1m=prices.get("cache_write_per_1m"),
                currency=prices.get("currency") or DEFAULT_CURRENCY,
                source="auto",
                matched_key=key,
                approximate=approximate,
            )

        builtin = self._match_builtin(normalized)
        if builtin is not None:
            prices, key = builtin
            return PriceEntry(
                input_per_1m=prices.get("input_per_1m"),
                output_per_1m=prices.get("output_per_1m"),
                cache_read_per_1m=prices.get("cache_read_per_1m"),
                cache_write_per_1m=prices.get("cache_write_per_1m"),
                currency=DEFAULT_CURRENCY,
                source="builtin",
                matched_key=key,
            )
        return None


_catalog: Optional[PriceCatalog] = None
_catalog_lock = threading.Lock()


def get_price_catalog() -> PriceCatalog:
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = PriceCatalog()
        return _catalog


def _reset_price_catalog_for_tests(config_dir: Optional[Path] = None) -> PriceCatalog:
    global _catalog
    with _catalog_lock:
        _catalog = PriceCatalog(config_dir=config_dir)
        return _catalog


def _tokens_to_micros(tokens: int, price_per_1m: Optional[float]) -> int:
    """micro-USD for a token count at a USD-per-1M price: tokens * price."""
    if not price_per_1m or tokens <= 0:
        return 0
    product = Decimal(tokens) * Decimal(str(price_per_1m))
    return int(product.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _cost_micros_from_entry(
    entry: PriceEntry,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> int:
    """micro-USD for a usage bucket at a resolved price.

    Token semantics follow the existing usage pipeline: ``input_tokens`` is
    the uncached input slice; cache write/read tokens are separate. Missing
    cache prices fall back to the plain input price (conservative).
    """
    input_price = entry.input_per_1m or 0.0
    cache_write_price = entry.cache_write_per_1m if entry.cache_write_per_1m is not None else input_price
    cache_read_price = entry.cache_read_per_1m if entry.cache_read_per_1m is not None else input_price
    output_price = entry.output_per_1m or 0.0

    return (
        _tokens_to_micros(input_tokens, input_price)
        + _tokens_to_micros(cache_creation_input_tokens, cache_write_price)
        + _tokens_to_micros(cache_read_input_tokens, cache_read_price)
        + _tokens_to_micros(output_tokens, output_price)
    )


def _cost_result(
    entry: Optional[PriceEntry],
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> Dict[str, Any]:
    if entry is None:
        return {
            "cost_micro_usd": 0,
            "currency": DEFAULT_CURRENCY,
            "unpriced": True,
            "price": None,
        }
    return {
        "cost_micro_usd": _cost_micros_from_entry(
            entry, input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens
        ),
        "currency": entry.currency,
        "unpriced": False,
        "price": entry.to_dict(),
    }


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    catalog: Optional[PriceCatalog] = None,
) -> Dict[str, Any]:
    """Compute the cost of one usage bucket."""
    entry = (catalog or get_price_catalog()).resolve(model)
    return _cost_result(entry, input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens)


def _usage_fields() -> Dict[str, int]:
    return {
        "request_count": 0,
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def get_billing_overview(
    range_key: str = "all",
    user_filter: Optional[str] = None,
    provider: str = "copilot",
    refresh_catalog: bool = True,
) -> Dict[str, Any]:
    """Aggregate Copilot traffic from token-usage JSONL files and price it.

    Reads the same files as the token-usage overview; only entries whose
    ``provider`` matches are billed (older lines without a provider field are
    treated as ``copilot`` for back-compat). Costs are computed at read time
    against the layered price catalog.

    Counts are billed as recorded. Lines written before the cached-token
    semantics fix mixed inclusive and uncached input counts per model and
    cannot be told apart, so their costs are approximate.
    """
    from .token_usage_reporter import _range_cutoff_ts, _resolve_usage_files

    catalog = get_price_catalog()
    refresh_status: Dict[str, Any] = {}
    if refresh_catalog:
        refresh_status = catalog.ensure_fresh()

    cutoff = _range_cutoff_ts(range_key)
    provider_lower = (provider or "").lower()

    # (machine, user_id, model) -> usage
    user_rows: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    # model -> usage (aggregated across machines/users, for the dashboard table)
    model_rows: Dict[str, Dict[str, int]] = {}
    totals = _usage_fields()
    unpriced_models: set[str] = set()
    # Resolved price per model. Only the *price* is cached here; every row's
    # cost must be computed from that row's own token counts (JSONL lines are
    # deltas, so per-line usage differs wildly).
    resolved_prices: Dict[str, Optional[PriceEntry]] = {}

    for machine, path in _resolve_usage_files():
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    ts = int(payload.get("timestamp", 0) or 0)
                    if cutoff is not None and ts < cutoff:
                        continue
                    line_user = str(payload.get("user_id") or "anonymous")
                    if user_filter is not None and line_user != user_filter:
                        continue
                    models = payload.get("models")
                    if not isinstance(models, list):
                        continue
                    for model_usage in models:
                        if not isinstance(model_usage, dict):
                            continue
                        entry_provider = str(model_usage.get("provider") or "copilot").lower()
                        if entry_provider != provider_lower:
                            continue
                        model_id = str(model_usage.get("model") or model_usage.get("model_id") or "unknown")
                        counts = {
                            "request_count": int(model_usage.get("request_count", 0) or 0),
                            "input_tokens": int(model_usage.get("input_tokens", 0) or 0),
                            "cache_creation_input_tokens": int(model_usage.get("cache_creation_input_tokens", 0) or 0),
                            "cache_read_input_tokens": int(model_usage.get("cache_read_input_tokens", 0) or 0),
                            "output_tokens": int(model_usage.get("output_tokens", 0) or 0),
                        }
                        counts["total_tokens"] = (
                            counts["input_tokens"]
                            + counts["cache_creation_input_tokens"]
                            + counts["cache_read_input_tokens"]
                            + counts["output_tokens"]
                        )
                        if not any(counts.values()):
                            continue

                        for target_key, target_map in (
                            ((machine, line_user, model_id), user_rows),
                            (model_id, model_rows),
                        ):
                            bucket = target_map.setdefault(target_key, _usage_fields())
                            for field, value in counts.items():
                                bucket[field] += value
                        for field, value in counts.items():
                            totals[field] += value

                        if model_id not in resolved_prices:
                            entry = catalog.resolve(model_id)
                            resolved_prices[model_id] = entry
                            if entry is None:
                                unpriced_models.add(model_id)
        except Exception:
            continue

    def _cost_for(model_id: str, usage: Dict[str, int]) -> Dict[str, Any]:
        return _cost_result(
            resolved_prices.get(model_id),
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cache_creation_input_tokens"],
            usage["cache_read_input_tokens"],
        )

    rows: List[Dict[str, Any]] = []
    # Model and grand totals are sums of the already-rounded row costs, so
    # every total equals the sum of the rows a client would add up.
    model_cost_micros: Dict[str, int] = {}
    for (machine, user_id, model_id), usage in sorted(
        user_rows.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower(), item[0][2].lower())
    ):
        cost = _cost_for(model_id, usage)
        model_cost_micros[model_id] = model_cost_micros.get(model_id, 0) + cost["cost_micro_usd"]
        rows.append({
            "machine": machine,
            "user_id": user_id,
            "model_id": model_id,
            **usage,
            "cost_micro_usd": cost["cost_micro_usd"],
            "cost_usd": cost["cost_micro_usd"] / MICROS_PER_USD,
            "currency": cost["currency"],
            "unpriced": cost["unpriced"],
        })

    model_row_list: List[Dict[str, Any]] = []
    for model_id, usage in sorted(model_rows.items(), key=lambda item: item[0].lower()):
        cost = _cost_for(model_id, usage)
        price = cost.get("price")
        micros = model_cost_micros.get(model_id, 0)
        model_row_list.append({
            "model_id": model_id,
            **usage,
            "cost_micro_usd": micros,
            "cost_usd": micros / MICROS_PER_USD,
            "currency": cost["currency"],
            "unpriced": cost["unpriced"],
            "price_source": price.get("source") if price else None,
            "price_matched_key": price.get("matched_key") if price else None,
            "price_approximate": price.get("approximate") if price else None,
        })

    total_micros = sum(row["cost_micro_usd"] for row in model_row_list)
    return {
        "range": range_key,
        "provider": provider_lower,
        "currency": DEFAULT_CURRENCY,
        "totals": {**totals, "cost_micro_usd": total_micros, "cost_usd": total_micros / MICROS_PER_USD},
        "rows": rows,
        "model_rows": model_row_list,
        "unpriced_models": sorted(unpriced_models),
        "catalog_refresh": refresh_status,
    }
