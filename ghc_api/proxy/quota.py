"""In-memory upstream quota accounting for configured proxies.

Design: docs/decisions/UPSTREAM_QUOTA.md

The ledger, alerts and fetch schedule all live in memory and are rebuilt from
the external quota command after a restart. The command itself is generic:
ghc-api never knows which platform it talks to, only how many "points" remain
and when each batch expires.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import ProxyProfileConfig, ProxyQuotaConfig, ProxyRegistry


RETRY_INTERVAL_SECONDS = 60
# A batch stops being usable this many seconds before its nominal expiry, so a
# few seconds of clock skew can never let an about-to-expire batch through.
EXPIRY_MARGIN_SECONDS = 60
MAX_ALERTS_PER_PROFILE = 50
CALIBRATION_TOLERANCE = 0.01
TICK_SECONDS = 1.0
STDERR_PREVIEW_CHARS = 200


@dataclass
class ReserveResult:
    allowed: bool
    receipt: List[Tuple[int, float]] = field(default_factory=list)
    # "" | "exhausted" | "unavailable"
    reason: str = ""


@dataclass
class _Account:
    name: str
    quota: ProxyQuotaConfig
    lock: threading.RLock = field(default_factory=threading.RLock)
    # expires_at (unix seconds) -> remaining points. Batches sharing an
    # expires_at are merged into one entry.
    batches: Dict[int, float] = field(default_factory=dict)
    initialized: bool = False
    had_failure: bool = False
    last_fetch_at: Optional[float] = None
    last_success_at: Optional[float] = None
    last_error: Optional[str] = None
    alerts: deque = field(default_factory=lambda: deque(maxlen=MAX_ALERTS_PER_PROFILE))
    fetch_in_progress: bool = False
    # Ledger snapshot taken when the current fetch started, used for calibration.
    pre_fetch_batches: Optional[Dict[int, float]] = None
    # Deductions made while the command was running, re-applied to the fetched
    # data so in-flight requests are never counted twice (see design section 7).
    deductions_during_fetch: List[Tuple[int, float]] = field(default_factory=list)


def _prune_expired(batches: Dict[int, float], now: float) -> None:
    cutoff = now + EXPIRY_MARGIN_SECONDS
    for expires_at in [e for e in batches if e <= cutoff]:
        del batches[expires_at]


def _deduct(batches: Dict[int, float], amount: float) -> List[Tuple[int, float]]:
    """Remove ``amount`` from the earliest-expiring batches first."""
    receipt: List[Tuple[int, float]] = []
    remaining = amount
    for expires_at in sorted(batches):
        if remaining <= 0:
            break
        available = batches[expires_at]
        taken = min(available, remaining)
        receipt.append((expires_at, taken))
        batches[expires_at] = available - taken
        remaining -= taken
    for expires_at in [e for e, value in batches.items() if value <= 0]:
        del batches[expires_at]
    return receipt


def _parse_batches(stdout: str) -> Dict[int, float]:
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        raise ValueError("output is not valid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
        raise ValueError("output must be an object with a 'batches' array")

    batches: Dict[int, float] = {}
    for item in data["batches"]:
        if not isinstance(item, dict):
            raise ValueError("each batch must be an object")
        amount = item.get("amount")
        expires_at = item.get("expires_at")
        if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount < 0):
            raise ValueError("batch 'amount' must be a non-negative number")
        if isinstance(expires_at, bool) or not isinstance(expires_at, int):
            raise ValueError("batch 'expires_at' must be an integer unix timestamp")
        batches[expires_at] = batches.get(expires_at, 0.0) + amount
    return {e: v for e, v in batches.items() if v > 0}


def _run_command(quota: ProxyQuotaConfig) -> Tuple[bool, object]:
    """Return (True, batches) or (False, error string). Never raises."""
    try:
        completed = subprocess.run(
            list(quota.command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Never fail on undecodable bytes: bad output then simply fails
            # JSON validation like any other malformed output.
            encoding="utf-8",
            errors="replace",
            timeout=quota.command_timeout_seconds,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Only the exception class is surfaced: str(exc) can contain the
        # command path, which /api/quota must never expose.
        return False, f"quota command could not run: {type(exc).__name__}"
    if completed.returncode != 0:
        preview = (completed.stderr or "")[:STDERR_PREVIEW_CHARS]
        return False, f"quota command exited with code {completed.returncode}: {preview}"
    try:
        return True, _parse_batches(completed.stdout or "")
    except ValueError as exc:
        return False, f"invalid quota command output: {exc}"


class QuotaManager:
    """Owns one ledger per profile plus a single background refresh thread."""

    def __init__(self, registry: Optional[ProxyRegistry] = None) -> None:
        self.registry = registry or ProxyRegistry()
        self._lock = threading.RLock()
        self._accounts: Dict[str, _Account] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Replaceable in tests that need to advance time without sleeping.
        self._now: Callable[[], float] = time.time

    # -- request path -----------------------------------------------------

    def reserve(self, profile: ProxyProfileConfig, cost: Optional[float]) -> ReserveResult:
        if profile.quota is None or cost is None:
            return ReserveResult(allowed=True)
        with self._lock:
            account = self._accounts.get(profile.name)
        if account is None:
            # No ledger yet (e.g. before the first reconcile). Treat quota
            # information as unavailable: "宁可少估".
            return ReserveResult(allowed=False, reason="unavailable")

        now = self._now()
        with account.lock:
            _prune_expired(account.batches, now)
            if not account.initialized or account.had_failure:
                return ReserveResult(allowed=False, reason="unavailable")
            total = sum(account.batches.values())
            if total - cost < profile.quota.min_remaining:
                return ReserveResult(allowed=False, reason="exhausted")
            receipt = _deduct(account.batches, cost)
            if account.fetch_in_progress:
                account.deductions_during_fetch.extend(receipt)
            return ReserveResult(allowed=True, receipt=receipt)

    def refund(self, profile_name: str, receipt: List[Tuple[int, float]]) -> None:
        if not receipt:
            return
        with self._lock:
            account = self._accounts.get(profile_name)
        if account is None:
            return
        now = self._now()
        with account.lock:
            for expires_at, amount in receipt:
                if expires_at <= now + EXPIRY_MARGIN_SECONDS:
                    continue
                account.batches[expires_at] = account.batches.get(expires_at, 0.0) + amount

    # -- fetching ---------------------------------------------------------

    @staticmethod
    def _is_due(account: _Account, now: float) -> bool:
        if account.fetch_in_progress:
            return False
        if account.last_fetch_at is None:
            return True
        if account.had_failure or not account.initialized:
            return now - account.last_fetch_at >= RETRY_INTERVAL_SECONDS
        return now - account.last_fetch_at >= account.quota.refresh_interval_seconds

    def _run_fetch(self, account: _Account, quota: ProxyQuotaConfig) -> bool:
        now = self._now()
        with account.lock:
            account.fetch_in_progress = True
            account.pre_fetch_batches = dict(account.batches)
            account.deductions_during_fetch = []
            account.last_fetch_at = now

        # The command runs WITHOUT holding the ledger lock, so requests are
        # never slowed down by a slow quota command.
        try:
            ok, payload = _run_command(quota)
        except Exception as exc:  # never leave fetch_in_progress stuck
            ok, payload = False, f"quota command failed: {type(exc).__name__}"

        now = self._now()
        with account.lock:
            if not ok:
                # A failed fetch means "no quota information": clear the ledger
                # so restricted models are rejected until a later success.
                account.batches = {}
                account.had_failure = True
                account.last_error = str(payload)
                account.pre_fetch_batches = None
                account.deductions_during_fetch = []
            else:
                # Drop expired batches first, so calibration ignores them and
                # in-flight deductions are replayed onto usable points only.
                new_batches: Dict[int, float] = dict(payload)  # type: ignore[arg-type]
                _prune_expired(new_batches, now)
                if (account.initialized and not account.had_failure
                        and account.pre_fetch_batches is not None):
                    _prune_expired(account.pre_fetch_batches, now)
                    self._calibrate(account, account.pre_fetch_batches, new_batches, now)
                for expires_at, amount in account.deductions_during_fetch:
                    _deduct(new_batches, amount)
                account.batches = new_batches
                account.initialized = True
                account.had_failure = False
                account.last_success_at = now
                account.last_error = None
                account.pre_fetch_batches = None
                account.deductions_during_fetch = []
            account.fetch_in_progress = False
        return ok

    @staticmethod
    def _calibrate(
        account: _Account,
        ours: Dict[int, float],
        fetched: Dict[int, float],
        now: float,
    ) -> None:
        for expires_at, ours_amount in ours.items():
            # Both sides are already pruned, so an unexpired batch missing from
            # the fetched data has been used up upstream: compare it as 0.
            fetched_amount = fetched.get(expires_at, 0.0)
            diff = ours_amount - fetched_amount
            # Our ledger higher than reality => upstream charged more than we
            # recorded (quota_cost too small). New batches and small deltas are
            # ignored; the ledger being lower is expected (cost set high).
            if diff >= CALIBRATION_TOLERANCE:
                account.alerts.append({
                    "time": now,
                    "profile": account.name,
                    "expires_at": expires_at,
                    "ours": ours_amount,
                    "fetched": fetched_amount,
                    "diff": diff,
                })

    # -- scheduling / config hot reload -----------------------------------

    def _reconcile(self) -> None:
        """Sync ledgers with the current config and fetch those that are due.

        Runs on the background thread (and once synchronously at startup). It
        is intentionally sequential and callers may invoke it directly in tests
        to drive a full refresh without starting a thread.
        """
        snapshot = self.registry.snapshot()
        now = self._now()
        wanted = {
            name for name, profile in snapshot.profiles.items() if profile.quota is not None
        }
        with self._lock:
            for name in list(self._accounts):
                if name not in wanted:
                    del self._accounts[name]

        for name, profile in snapshot.profiles.items():
            if profile.quota is None:
                continue
            quota = profile.quota
            with self._lock:
                account = self._accounts.get(name)
                created = account is None
                if created:
                    account = _Account(name=name, quota=quota)
                    self._accounts[name] = account

            due = created
            with account.lock:
                if not created and account.quota.fingerprint() != quota.fingerprint():
                    # command / command_timeout changed: clear and re-fetch now.
                    account.batches = {}
                    account.initialized = False
                    account.had_failure = False
                    account.pre_fetch_batches = None
                    account.deductions_during_fetch = []
                    account.last_fetch_at = None
                    due = True
                # Always adopt the newest config so min_remaining / interval /
                # quota_cost changes take effect without a re-fetch.
                account.quota = quota
                if not due:
                    due = self._is_due(account, now)

            if due:
                self._run_fetch(account, quota)

    def _loop(self) -> None:
        while not self._stop_event.wait(TICK_SECONDS):
            try:
                self._reconcile()
            except Exception as exc:  # keep the refresh thread alive
                print(f"[Quota] refresh loop error: {exc}")

    def start(self, initial_sync: bool = True) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        if initial_sync:
            # One synchronous fetch per profile, each bounded by its own
            # command_timeout_seconds. Failures never prevent startup.
            self._reconcile()
        self._thread = threading.Thread(target=self._loop, name="quota-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def refresh_now(self, name: str, quota: Optional[ProxyQuotaConfig] = None) -> bool:
        """Run one fetch synchronously; used by tests and manual refreshes."""
        with self._lock:
            account = self._accounts.get(name)
        if account is None:
            return False
        if quota is None:
            with account.lock:
                quota = account.quota
        return self._run_fetch(account, quota)

    # -- dashboard --------------------------------------------------------

    def api_snapshot(self) -> Dict:
        with self._lock:
            accounts = list(self._accounts.values())
        now = self._now()
        profiles = []
        for account in accounts:
            with account.lock:
                _prune_expired(account.batches, now)
                profiles.append({
                    "profile": account.name,
                    "remaining": sum(account.batches.values()),
                    "min_remaining": account.quota.min_remaining,
                    "batches": [
                        {
                            "amount": amount,
                            "expires_at": expires_at,
                            "seconds_remaining": expires_at - now,
                        }
                        for expires_at, amount in sorted(account.batches.items())
                    ],
                    "last_fetch_at": account.last_fetch_at,
                    "last_success_at": account.last_success_at,
                    "last_error": account.last_error,
                    "alerts": list(account.alerts),
                })
        return {"profiles": profiles}


quota_manager = QuotaManager()
