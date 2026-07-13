"""Persistent replay storage for Responses reasoning state.

The Anthropic Messages protocol does not carry every Responses output item
back to the server.  This module stores the authoritative, completed
``response.output`` snapshot so a later Messages request can replay it without
normalising away encrypted reasoning, assistant ``phase`` values, or tool
items.

The store deliberately returns *all* records matching a visible assistant
fingerprint.  Retries can produce the same visible projection with different
hidden state, so silently selecting the newest row would turn the database
back into a lossy single-slot cache.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


_SCHEMA_VERSION = 1
_PAYLOAD_VERSION = 1
_INTEGRITY_KEY_NAME = "row_hmac_key_v1"


class ReasoningReplayError(Exception):
    """Base class for replay-store errors."""


class ReplaySerializationError(ReasoningReplayError, ValueError):
    """Raised when a value cannot be represented as strict JSON."""


class ReplayEncryptionConfigurationError(ReasoningReplayError, ValueError):
    """Raised when requested Fernet encryption cannot be configured."""


class ReplayStoreClosedError(ReasoningReplayError, RuntimeError):
    """Raised when an operation is attempted after ``close``."""


class ReplayConflictError(ReasoningReplayError, ValueError):
    """Raised when a caller-supplied replay id already exists."""


class ReplayParentError(ReasoningReplayError, ValueError):
    """Raised when a parent is missing, expired, corrupt, or out of scope."""


class ReplayQuotaExceededError(ReasoningReplayError, ValueError):
    """Raised before a row is written when a logical byte quota is exceeded."""


@dataclass(frozen=True)
class ReplayEncryptionStatus:
    """How newly written payloads are protected.

    ``message`` is intentionally explicit for runtime status pages and startup
    validation.  In particular, plaintext mode is never reported as an
    implicit/default success state.
    """

    mode: str
    key_configured: bool
    message: str

    @property
    def encrypted(self) -> bool:
        return self.mode == "fernet"


@dataclass(frozen=True)
class ReplayRecord:
    """One immutable replay node in a transcript DAG."""

    replay_id: str
    tenant_id: str
    session_id: str
    model: str
    assistant_visible_fingerprint: str
    output_items: List[Any]
    assistant_visible_blocks: List[Any]
    profile: Any
    created_at: float
    expires_at: float
    parent_replay_id: Optional[str]
    encrypted: bool


@dataclass(frozen=True)
class ReplayReadIssue:
    """A fail-closed reason for omitting a row from lookup results."""

    replay_id: str
    code: str
    message: str


@dataclass(frozen=True)
class ReplayLookupResult:
    """Valid replay candidates and any rows rejected while reading them."""

    records: Tuple[ReplayRecord, ...]
    issues: Tuple[ReplayReadIssue, ...] = ()
    expired_count: int = 0

    @property
    def found(self) -> bool:
        return bool(self.records)

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def ambiguous(self) -> bool:
        """Whether an exact visible projection has multiple valid retries."""

        return len(self.records) > 1


def _validate_json_value(
    value: Any, path: str = "$", active_containers: Optional[set] = None
) -> None:
    """Reject Python values that JSON would coerce or encode ambiguously."""

    if active_containers is None:
        active_containers = set()
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplaySerializationError("%s contains NaN or infinity" % path)
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ReplaySerializationError(
                "%s contains an invalid Unicode surrogate" % path
            ) from exc
        return
    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            raise ReplaySerializationError("%s contains a reference cycle" % path)
        active_containers.add(container_id)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item, "%s[%d]" % (path, index), active_containers
                )
        finally:
            active_containers.remove(container_id)
        return
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in active_containers:
            raise ReplaySerializationError("%s contains a reference cycle" % path)
        active_containers.add(container_id)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ReplaySerializationError(
                        "%s contains a non-string object key" % path
                    )
                _validate_json_value(key, path + ".<key>", active_containers)
                _validate_json_value(
                    item, "%s.%s" % (path, key), active_containers
                )
        finally:
            active_containers.remove(container_id)
        return
    raise ReplaySerializationError(
        "%s contains unsupported value type %s" % (path, type(value).__name__)
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes.

    NaN and infinities are rejected because they are not JSON and would make
    cross-process fingerprints ambiguous.  No content is truncated or
    filtered.
    """

    _validate_json_value(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReplaySerializationError("value is not strict JSON: %s" % exc) from exc
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ReplaySerializationError("value is not valid UTF-8 JSON") from exc


def canonical_json(value: Any) -> str:
    """String form of :func:`canonical_json_bytes`."""

    return canonical_json_bytes(value).decode("utf-8")


def assistant_visible_fingerprint(assistant_visible_blocks: Any) -> str:
    """Fingerprint an assistant's complete visible Anthropic content blocks."""

    digest = hashlib.sha256(
        b"ghc-assistant-visible-fingerprint-v1\0"
        + canonical_json_bytes(assistant_visible_blocks)
    ).hexdigest()
    return "sha256:" + digest


def _identity(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("%s must be a string" % name)
    value = value.strip()
    if not value:
        raise ValueError("%s must not be empty" % name)
    return value


def _optional_identity(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _identity(name, value)


def _timestamp(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite Unix timestamp" % name) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("%s must be a finite, non-negative Unix timestamp" % name)
    return result


class ReasoningReplayStore:
    """Thread-safe SQLite store for completed Responses output snapshots.

    Args:
        db_path: SQLite database path. Parent directories are created.
        ttl_seconds: Default lifetime for new rows.
        encryption_key: Optional urlsafe-base64 Fernet key (``str`` or bytes).
            ``cryptography`` is imported only when a key is configured.
        require_encryption: Fail construction instead of using plaintext when
            no key is supplied.
        clock: Injectable Unix-time function, primarily for deterministic
            tests.

    Lookup identity is the HMAC-indexed tuple ``tenant_id + session_id + model
    + assistant_visible_fingerprint``.  A generated ``replay_id`` and optional
    ``parent_replay_id`` allow multiple retries and branches under that tuple.
    """

    def __init__(
        self,
        db_path: Union[str, os.PathLike],
        *,
        ttl_seconds: float = 86400,
        encryption_key: Optional[Union[str, bytes]] = None,
        require_encryption: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._default_ttl = self._validate_ttl(ttl_seconds)
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        self._fernet = None

        normalized_key: Optional[bytes]
        if isinstance(encryption_key, str):
            try:
                normalized_key = encryption_key.strip().encode("ascii")
            except UnicodeEncodeError as exc:
                raise ReplayEncryptionConfigurationError(
                    "Fernet encryption key must be ASCII"
                ) from exc
        elif isinstance(encryption_key, bytes):
            normalized_key = encryption_key.strip()
        elif encryption_key is None:
            normalized_key = None
        else:
            raise TypeError("encryption_key must be str, bytes, or None")

        if not normalized_key:
            if require_encryption:
                raise ReplayEncryptionConfigurationError(
                    "replay encryption is required but no Fernet key was configured"
                )
            self.encryption_status = ReplayEncryptionStatus(
                mode="plaintext",
                key_configured=False,
                message=(
                    "No Fernet key configured; replay payloads are stored in plaintext. "
                    "Configure a key before using lossless_required mode."
                ),
            )
        else:
            try:
                from cryptography.fernet import Fernet
            except ImportError as exc:
                raise ReplayEncryptionConfigurationError(
                    "a Fernet key was configured but the optional 'cryptography' "
                    "package is not installed"
                ) from exc
            try:
                self._fernet = Fernet(normalized_key)
            except (TypeError, ValueError) as exc:
                raise ReplayEncryptionConfigurationError(
                    "invalid Fernet encryption key"
                ) from exc
            self.encryption_status = ReplayEncryptionStatus(
                mode="fernet",
                key_configured=True,
                message="Replay payload encryption is enabled with Fernet.",
            )

        path_text = os.fspath(db_path)
        if not isinstance(path_text, str) or not path_text.strip():
            raise ValueError("db_path must not be empty")
        if path_text != ":memory:":
            parent = Path(path_text).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            path_text,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        self._connection = connection
        try:
            with self._lock:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA busy_timeout = 30000")
                if path_text != ":memory:":
                    self._enable_wal_with_retry(connection)
                    connection.execute("PRAGMA synchronous = NORMAL")
                self._initialise_schema()
                self._integrity_key = self._load_or_create_integrity_key()
        except Exception:
            connection.close()
            self._connection = None
            raise

    @staticmethod
    def _validate_ttl(value: Any) -> float:
        try:
            ttl = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("ttl_seconds must be a finite positive number") from exc
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_seconds must be a finite positive number")
        return ttl

    @staticmethod
    def _enable_wal_with_retry(
        connection: sqlite3.Connection, timeout_seconds: float = 30.0
    ) -> None:
        """Enable WAL even when several workers initialise the DB together.

        Unlike ordinary writes, SQLite's ``PRAGMA journal_mode`` can report a
        lock immediately without honouring ``busy_timeout``.  A short bounded
        retry closes that first-start race; any non-lock error is still raised
        unchanged.
        """

        deadline = time.monotonic() + timeout_seconds
        delay = 0.005
        while True:
            try:
                connection.execute("PRAGMA journal_mode = WAL").fetchone()
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if (
                    ("locked" not in message and "busy" not in message)
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    def _conn(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise ReplayStoreClosedError("reasoning replay store is closed")
        return connection

    def _begin(self) -> sqlite3.Connection:
        connection = self._conn()
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _commit(connection: sqlite3.Connection) -> None:
        connection.execute("COMMIT")

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass

    def _initialise_schema(self) -> None:
        connection = self._conn()
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version not in (0, _SCHEMA_VERSION):
            raise ReasoningReplayError(
                "unsupported reasoning replay schema version %d" % current_version
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reasoning_replay_metadata (
                name TEXT PRIMARY KEY,
                value BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reasoning_replay_records (
                replay_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL CHECK(length(trim(tenant_id)) > 0),
                session_id TEXT NOT NULL CHECK(length(trim(session_id)) > 0),
                model TEXT NOT NULL CHECK(length(trim(model)) > 0),
                assistant_visible_fingerprint TEXT NOT NULL,
                lookup_hmac BLOB NOT NULL,
                parent_replay_id TEXT,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                encrypted INTEGER NOT NULL CHECK(encrypted IN (0, 1)),
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL,
                payload_hmac BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS reasoning_replay_lookup_idx
                ON reasoning_replay_records(lookup_hmac, created_at, replay_id);
            CREATE INDEX IF NOT EXISTS reasoning_replay_scope_idx
                ON reasoning_replay_records(tenant_id, session_id, model);
            CREATE INDEX IF NOT EXISTS reasoning_replay_parent_idx
                ON reasoning_replay_records(parent_replay_id);
            CREATE INDEX IF NOT EXISTS reasoning_replay_expiry_idx
                ON reasoning_replay_records(expires_at);
            """
        )
        if current_version == 0:
            connection.execute("PRAGMA user_version = %d" % _SCHEMA_VERSION)

    def _load_or_create_integrity_key(self) -> bytes:
        connection = self._begin()
        try:
            candidate = secrets.token_bytes(32)
            connection.execute(
                "INSERT OR IGNORE INTO reasoning_replay_metadata(name, value) VALUES (?, ?)",
                (_INTEGRITY_KEY_NAME, candidate),
            )
            row = connection.execute(
                "SELECT value FROM reasoning_replay_metadata WHERE name = ?",
                (_INTEGRITY_KEY_NAME,),
            ).fetchone()
            if row is None:
                raise ReasoningReplayError("failed to initialise replay integrity key")
            key = bytes(row["value"])
            if len(key) != 32:
                raise ReasoningReplayError("replay integrity key is corrupt")
            self._commit(connection)
            return key
        except Exception:
            self._rollback(connection)
            raise

    def _lookup_hmac(
        self,
        tenant_id: str,
        session_id: str,
        model: str,
        fingerprint: str,
    ) -> bytes:
        identity = [tenant_id, session_id, model, fingerprint]
        return hmac.new(
            self._integrity_key,
            b"reasoning-replay-lookup-v1\0" + canonical_json_bytes(identity),
            hashlib.sha256,
        ).digest()

    def _payload_hmac(
        self,
        *,
        replay_id: str,
        tenant_id: str,
        session_id: str,
        model: str,
        fingerprint: str,
        lookup_hmac: bytes,
        parent_replay_id: Optional[str],
        created_at: float,
        expires_at: float,
        encrypted: bool,
        payload_version: int,
        payload: bytes,
    ) -> bytes:
        header = {
            "replay_id": replay_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
            "assistant_visible_fingerprint": fingerprint,
            "lookup_hmac": lookup_hmac.hex(),
            "parent_replay_id": parent_replay_id,
            "created_at": created_at,
            "expires_at": expires_at,
            "encrypted": encrypted,
            "payload_version": payload_version,
        }
        authenticated = (
            b"reasoning-replay-row-v1\0"
            + canonical_json_bytes(header)
            + b"\0"
            + payload
        )
        return hmac.new(self._integrity_key, authenticated, hashlib.sha256).digest()

    @staticmethod
    def _resolve_fingerprint(
        assistant_visible_blocks: Optional[Sequence[Any]],
        fingerprint: Optional[str],
        *,
        require_blocks: bool,
    ) -> Tuple[Optional[List[Any]], str]:
        resolved_blocks: Optional[List[Any]] = None
        calculated: Optional[str] = None
        if assistant_visible_blocks is not None:
            if not isinstance(assistant_visible_blocks, (list, tuple)):
                raise TypeError("assistant_visible_blocks must be a JSON array")
            resolved_blocks = list(assistant_visible_blocks)
            calculated = assistant_visible_fingerprint(resolved_blocks)
        elif require_blocks:
            raise ValueError("assistant_visible_blocks is required")

        if fingerprint is not None:
            fingerprint = _identity("assistant_visible_fingerprint", fingerprint)
        if calculated is not None and fingerprint is not None and calculated != fingerprint:
            raise ValueError(
                "assistant_visible_fingerprint does not match assistant_visible_blocks"
            )
        resolved = calculated or fingerprint
        if resolved is None:
            raise ValueError(
                "assistant_visible_blocks or assistant_visible_fingerprint is required"
            )
        return resolved_blocks, resolved

    def put(
        self,
        *,
        tenant_id: str,
        session_id: str,
        model: str,
        output_items: Sequence[Any],
        assistant_visible_blocks: Sequence[Any],
        profile: Any,
        assistant_visible_fingerprint: Optional[str] = None,
        replay_id: Optional[str] = None,
        parent_replay_id: Optional[str] = None,
        created_at: Optional[float] = None,
        ttl_seconds: Optional[float] = None,
        max_record_bytes: Optional[int] = None,
        max_tenant_bytes: Optional[int] = None,
        max_total_bytes: Optional[int] = None,
    ) -> ReplayRecord:
        """Insert one completed response as a new DAG node.

        Inserts never overwrite another row.  Reusing a caller-supplied
        ``replay_id`` raises :class:`ReplayConflictError`; retries with generated
        ids remain independently retrievable even if their visible fingerprints
        are identical.
        """

        tenant_id = _identity("tenant_id", tenant_id)
        session_id = _identity("session_id", session_id)
        model = _identity("model", model)
        replay_id = _identity("replay_id", replay_id or uuid.uuid4().hex)
        parent_replay_id = _optional_identity("parent_replay_id", parent_replay_id)
        if parent_replay_id == replay_id:
            raise ReplayParentError("a replay row cannot be its own parent")

        if not isinstance(output_items, (list, tuple)):
            raise TypeError("output_items must be a JSON array")
        output_list = list(output_items)
        visible_list, fingerprint = self._resolve_fingerprint(
            assistant_visible_blocks,
            assistant_visible_fingerprint,
            require_blocks=True,
        )
        assert visible_list is not None

        created = _timestamp(
            "created_at", self._clock() if created_at is None else created_at
        )
        ttl = self._default_ttl if ttl_seconds is None else self._validate_ttl(ttl_seconds)
        expires = created + ttl
        if not math.isfinite(expires):
            raise ValueError("created_at + ttl_seconds must be finite")

        plain_payload = canonical_json_bytes(
            {
                "version": _PAYLOAD_VERSION,
                "output_items": output_list,
                "assistant_visible_blocks": visible_list,
                "profile": profile,
            }
        )
        # Parse once before writing so the returned object cannot share mutable
        # references with caller-owned lists/dicts.
        payload_object = json.loads(plain_payload.decode("utf-8"))

        encrypted = self._fernet is not None
        stored_payload = (
            self._fernet.encrypt(plain_payload) if self._fernet is not None else plain_payload
        )
        lookup_hmac = self._lookup_hmac(tenant_id, session_id, model, fingerprint)
        payload_hmac = self._payload_hmac(
            replay_id=replay_id,
            tenant_id=tenant_id,
            session_id=session_id,
            model=model,
            fingerprint=fingerprint,
            lookup_hmac=lookup_hmac,
            parent_replay_id=parent_replay_id,
            created_at=created,
            expires_at=expires,
            encrypted=encrypted,
            payload_version=_PAYLOAD_VERSION,
            payload=stored_payload,
        )

        row_logical_bytes = self._row_logical_bytes(
            replay_id=replay_id,
            tenant_id=tenant_id,
            session_id=session_id,
            model=model,
            fingerprint=fingerprint,
            lookup_hmac=lookup_hmac,
            parent_replay_id=parent_replay_id,
            payload=stored_payload,
            payload_hmac=payload_hmac,
        )
        max_record_bytes = self._optional_positive_quota(
            "max_record_bytes", max_record_bytes
        )
        max_tenant_bytes = self._optional_positive_quota(
            "max_tenant_bytes", max_tenant_bytes
        )
        max_total_bytes = self._optional_positive_quota(
            "max_total_bytes", max_total_bytes
        )
        if max_record_bytes is not None and row_logical_bytes > max_record_bytes:
            raise ReplayQuotaExceededError(
                "replay row exceeds the configured per-record byte quota"
            )

        with self._lock:
            connection = self._begin()
            try:
                if parent_replay_id is not None:
                    parent = connection.execute(
                        "SELECT * FROM reasoning_replay_records WHERE replay_id = ?",
                        (parent_replay_id,),
                    ).fetchone()
                    if parent is None:
                        raise ReplayParentError("parent replay row does not exist")
                    if (
                        parent["tenant_id"] != tenant_id
                        or parent["session_id"] != session_id
                        or parent["model"] != model
                    ):
                        raise ReplayParentError(
                            "parent replay row belongs to a different tenant/session/model"
                        )
                    parent_record, parent_issue = self._decode_row(parent)
                    if parent_issue is not None or parent_record is None:
                        raise ReplayParentError("parent replay row failed integrity validation")
                    if parent_record.expires_at <= created:
                        raise ReplayParentError("parent replay row has expired")

                # Quotas are based on live logical row bytes, not SQLite/WAL
                # file length.  Checking under BEGIN IMMEDIATE makes the
                # reservation atomic across workers and rejects an oversized
                # snapshot before it can consume database space.
                if max_total_bytes is not None:
                    total = self._logical_size_bytes_locked(connection)
                    if total + row_logical_bytes > max_total_bytes:
                        raise ReplayQuotaExceededError(
                            "replay store exceeds the configured total byte quota"
                        )
                if max_tenant_bytes is not None:
                    tenant_total = self._logical_size_bytes_locked(
                        connection, tenant_id=tenant_id
                    )
                    if tenant_total + row_logical_bytes > max_tenant_bytes:
                        raise ReplayQuotaExceededError(
                            "replay tenant exceeds the configured byte quota"
                        )

                try:
                    connection.execute(
                        """
                        INSERT INTO reasoning_replay_records(
                            replay_id, tenant_id, session_id, model,
                            assistant_visible_fingerprint, lookup_hmac,
                            parent_replay_id, created_at, expires_at, encrypted,
                            payload_version, payload, payload_hmac
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            replay_id,
                            tenant_id,
                            session_id,
                            model,
                            fingerprint,
                            lookup_hmac,
                            parent_replay_id,
                            created,
                            expires,
                            int(encrypted),
                            _PAYLOAD_VERSION,
                            stored_payload,
                            payload_hmac,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    if "replay_id" in str(exc).lower() or "unique" in str(exc).lower():
                        raise ReplayConflictError(
                            "replay_id already exists: %s" % replay_id
                        ) from exc
                    raise
                self._commit(connection)
            except Exception:
                self._rollback(connection)
                raise

        return ReplayRecord(
            replay_id=replay_id,
            tenant_id=tenant_id,
            session_id=session_id,
            model=model,
            assistant_visible_fingerprint=fingerprint,
            output_items=payload_object["output_items"],
            assistant_visible_blocks=payload_object["assistant_visible_blocks"],
            profile=payload_object["profile"],
            created_at=created,
            expires_at=expires,
            parent_replay_id=parent_replay_id,
            encrypted=encrypted,
        )

    @staticmethod
    def _optional_positive_quota(name: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("%s must be an integer > 0" % name)
        return value

    @staticmethod
    def _row_logical_bytes(
        *,
        replay_id: str,
        tenant_id: str,
        session_id: str,
        model: str,
        fingerprint: str,
        lookup_hmac: bytes,
        parent_replay_id: Optional[str],
        payload: bytes,
        payload_hmac: bytes,
    ) -> int:
        # The fixed allowance covers SQLite numeric fields, record headers,
        # indexes and conservative row overhead. Variable fields are counted
        # exactly as UTF-8/BLOB bytes. This is deliberately independent of
        # freelists, pages and WAL checkpoints.
        text_values = (
            replay_id,
            tenant_id,
            session_id,
            model,
            fingerprint,
            parent_replay_id or "",
        )
        return (
            256
            + sum(len(value.encode("utf-8")) for value in text_values)
            + len(lookup_hmac)
            + len(payload)
            + len(payload_hmac)
        )

    @staticmethod
    def _logical_size_expression() -> str:
        return """
            256
            + length(CAST(replay_id AS BLOB))
            + length(CAST(tenant_id AS BLOB))
            + length(CAST(session_id AS BLOB))
            + length(CAST(model AS BLOB))
            + length(CAST(assistant_visible_fingerprint AS BLOB))
            + length(CAST(COALESCE(parent_replay_id, '') AS BLOB))
            + length(lookup_hmac)
            + length(payload)
            + length(payload_hmac)
        """

    def _logical_size_bytes_locked(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: Optional[str] = None,
    ) -> int:
        sql = "SELECT COALESCE(SUM(%s), 0) FROM reasoning_replay_records" % (
            self._logical_size_expression(),
        )
        params: Tuple[Any, ...] = ()
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            params = (tenant_id,)
        return int(connection.execute(sql, params).fetchone()[0] or 0)

    def logical_size_bytes(self, *, tenant_id: Optional[str] = None) -> int:
        """Return live logical bytes used globally or by one tenant."""

        tenant_id = _optional_identity("tenant_id", tenant_id)
        with self._lock:
            return self._logical_size_bytes_locked(
                self._conn(), tenant_id=tenant_id
            )

    def _row_integrity_issue(self, row: sqlite3.Row) -> Optional[ReplayReadIssue]:
        replay_id = str(row["replay_id"])
        try:
            if not isinstance(row["payload"], (bytes, bytearray, memoryview)):
                raise TypeError("payload is not a BLOB")
            if not isinstance(row["lookup_hmac"], (bytes, bytearray, memoryview)):
                raise TypeError("lookup_hmac is not a BLOB")
            if not isinstance(row["payload_hmac"], (bytes, bytearray, memoryview)):
                raise TypeError("payload_hmac is not a BLOB")
            payload = bytes(row["payload"])
            lookup_hmac = bytes(row["lookup_hmac"])
            created_at = float(row["created_at"])
            expires_at = float(row["expires_at"])
            encrypted_value = int(row["encrypted"])
            payload_version = int(row["payload_version"])
            if not math.isfinite(created_at) or not math.isfinite(expires_at):
                raise ValueError("timestamps are not finite")
            if encrypted_value not in (0, 1):
                raise ValueError("encrypted flag is invalid")
            expected_lookup = self._lookup_hmac(
                row["tenant_id"],
                row["session_id"],
                row["model"],
                row["assistant_visible_fingerprint"],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return ReplayReadIssue(
                replay_id,
                "row_corrupt",
                "Replay row has invalid SQLite field types: %s" % exc,
            )
        if not hmac.compare_digest(lookup_hmac, expected_lookup):
            return ReplayReadIssue(
                replay_id,
                "lookup_integrity_failed",
                "Replay lookup identity failed HMAC validation.",
            )

        expected_payload_hmac = self._payload_hmac(
            replay_id=replay_id,
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            model=row["model"],
            fingerprint=row["assistant_visible_fingerprint"],
            lookup_hmac=lookup_hmac,
            parent_replay_id=row["parent_replay_id"],
            created_at=created_at,
            expires_at=expires_at,
            encrypted=bool(encrypted_value),
            payload_version=payload_version,
            payload=payload,
        )
        if not hmac.compare_digest(bytes(row["payload_hmac"]), expected_payload_hmac):
            return ReplayReadIssue(
                replay_id,
                "integrity_check_failed",
                "Replay row failed HMAC validation.",
            )
        return None

    def _decode_row(
        self, row: sqlite3.Row
    ) -> Tuple[Optional[ReplayRecord], Optional[ReplayReadIssue]]:
        replay_id = str(row["replay_id"])
        integrity_issue = self._row_integrity_issue(row)
        if integrity_issue is not None:
            return None, integrity_issue

        payload = bytes(row["payload"])

        if bool(row["encrypted"]):
            if self._fernet is None:
                return None, ReplayReadIssue(
                    replay_id,
                    "encryption_key_missing",
                    "Replay row is encrypted but no Fernet key is configured.",
                )
            try:
                plain_payload = self._fernet.decrypt(payload)
            except Exception as exc:
                # Importing InvalidToken just to catch it would make this path
                # depend on cryptography even when encryption is disabled.  The
                # Fernet decrypt API reports authentication/decryption failures
                # as exceptions and no plaintext is used here.
                return None, ReplayReadIssue(
                    replay_id,
                    "decryption_failed",
                    "Replay row could not be authenticated/decrypted (%s)."
                    % type(exc).__name__,
                )
        else:
            plain_payload = payload

        try:
            decoded = json.loads(plain_payload.decode("utf-8"))
            if canonical_json_bytes(decoded) != plain_payload:
                raise ValueError("payload is not in canonical JSON form")
            if not isinstance(decoded, dict):
                raise ValueError("payload root is not an object")
            if decoded.get("version") != _PAYLOAD_VERSION:
                raise ValueError("unsupported payload version")
            if not isinstance(decoded.get("output_items"), list):
                raise ValueError("output_items is not an array")
            if not isinstance(decoded.get("assistant_visible_blocks"), list):
                raise ValueError("assistant_visible_blocks is not an array")
            if "profile" not in decoded:
                raise ValueError("profile is missing")
            calculated = assistant_visible_fingerprint(
                decoded["assistant_visible_blocks"]
            )
            if calculated != row["assistant_visible_fingerprint"]:
                raise ValueError("visible assistant fingerprint mismatch")
        except (UnicodeDecodeError, json.JSONDecodeError, ReplaySerializationError, ValueError) as exc:
            return None, ReplayReadIssue(
                replay_id,
                "payload_corrupt",
                "Replay payload is invalid: %s" % exc,
            )

        return ReplayRecord(
            replay_id=replay_id,
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            model=row["model"],
            assistant_visible_fingerprint=row["assistant_visible_fingerprint"],
            output_items=decoded["output_items"],
            assistant_visible_blocks=decoded["assistant_visible_blocks"],
            profile=decoded["profile"],
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
            parent_replay_id=row["parent_replay_id"],
            encrypted=bool(row["encrypted"]),
        ), None

    def get(
        self,
        *,
        tenant_id: str,
        session_id: str,
        model: str,
        assistant_visible_blocks: Optional[Sequence[Any]] = None,
        assistant_visible_fingerprint: Optional[str] = None,
        replay_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> ReplayLookupResult:
        """Return every valid retry matching the exact replay identity.

        Corrupt, unauthenticated, or undecryptable rows are omitted and reported
        in ``issues``.  Expired rows are deleted and counted.  Supplying
        ``replay_id`` selects one exact DAG node without changing the other key
        dimensions.
        """

        tenant_id = _identity("tenant_id", tenant_id)
        session_id = _identity("session_id", session_id)
        model = _identity("model", model)
        replay_id = _optional_identity("replay_id", replay_id)
        _, fingerprint = self._resolve_fingerprint(
            assistant_visible_blocks,
            assistant_visible_fingerprint,
            require_blocks=False,
        )
        lookup_hmac = self._lookup_hmac(tenant_id, session_id, model, fingerprint)
        current_time = _timestamp("now", self._clock() if now is None else now)

        sql = """
            SELECT * FROM reasoning_replay_records
            WHERE lookup_hmac = ?
              AND tenant_id = ? AND session_id = ? AND model = ?
              AND assistant_visible_fingerprint = ?
        """
        params: List[Any] = [
            lookup_hmac,
            tenant_id,
            session_id,
            model,
            fingerprint,
        ]
        if replay_id is not None:
            sql += " AND replay_id = ?"
            params.append(replay_id)
        sql += " ORDER BY created_at ASC, replay_id ASC"

        with self._lock:
            connection = self._begin()
            try:
                rows = list(connection.execute(sql, params).fetchall())
                precheck_issues = {}
                expired_ids = []
                for row in rows:
                    issue = self._row_integrity_issue(row)
                    if issue is not None:
                        precheck_issues[row["replay_id"]] = issue
                    elif float(row["expires_at"]) <= current_time:
                        expired_ids.append(row["replay_id"])
                if expired_ids:
                    connection.executemany(
                        "DELETE FROM reasoning_replay_records WHERE replay_id = ?",
                        ((item,) for item in expired_ids),
                    )
                self._commit(connection)
            except Exception:
                self._rollback(connection)
                raise

            records: List[ReplayRecord] = []
            issues: List[ReplayReadIssue] = []
            expired_set = set(expired_ids)
            for row in rows:
                if row["replay_id"] in expired_set:
                    continue
                if row["replay_id"] in precheck_issues:
                    issues.append(precheck_issues[row["replay_id"]])
                    continue
                record, issue = self._decode_row(row)
                if issue is not None:
                    issues.append(issue)
                elif record is not None:
                    records.append(record)

        return ReplayLookupResult(
            records=tuple(records),
            issues=tuple(issues),
            expired_count=len(expired_ids),
        )

    def purge(
        self,
        *,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        assistant_visible_fingerprint: Optional[str] = None,
        replay_id: Optional[str] = None,
        expired_only: bool = True,
        now: Optional[float] = None,
    ) -> int:
        """Delete expired rows, or explicitly delete a filtered scope.

        With its safe default (``expired_only=True``), ``purge()`` only removes
        expired rows.  Passing ``expired_only=False`` is an explicit request to
        remove every row matching the supplied filters; with no filters it
        clears the store.
        """

        tenant_id = _optional_identity("tenant_id", tenant_id)
        session_id = _optional_identity("session_id", session_id)
        model = _optional_identity("model", model)
        assistant_visible_fingerprint = _optional_identity(
            "assistant_visible_fingerprint", assistant_visible_fingerprint
        )
        replay_id = _optional_identity("replay_id", replay_id)
        current_time = _timestamp("now", self._clock() if now is None else now)

        conditions: List[str] = []
        params: List[Any] = []
        for column, value in (
            ("tenant_id", tenant_id),
            ("session_id", session_id),
            ("model", model),
            ("assistant_visible_fingerprint", assistant_visible_fingerprint),
            ("replay_id", replay_id),
        ):
            if value is not None:
                conditions.append(column + " = ?")
                params.append(value)
        if expired_only:
            conditions.append("expires_at <= ?")
            params.append(current_time)

        sql = "DELETE FROM reasoning_replay_records"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        with self._lock:
            connection = self._begin()
            try:
                cursor = connection.execute(sql, params)
                count = int(cursor.rowcount)
                self._commit(connection)
                return count
            except Exception:
                self._rollback(connection)
                raise

    def close(self) -> None:
        """Close the SQLite connection. The operation is idempotent."""

        with self._lock:
            connection = self._connection
            if connection is None:
                return
            self._connection = None
            connection.close()

    @property
    def closed(self) -> bool:
        return self._connection is None

    def __enter__(self) -> "ReasoningReplayStore":
        self._conn()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ReasoningReplayStore",
    "ReplayRecord",
    "ReplayLookupResult",
    "ReplayReadIssue",
    "ReplayEncryptionStatus",
    "ReasoningReplayError",
    "ReplaySerializationError",
    "ReplayEncryptionConfigurationError",
    "ReplayStoreClosedError",
    "ReplayConflictError",
    "ReplayParentError",
    "canonical_json",
    "canonical_json_bytes",
    "assistant_visible_fingerprint",
]
