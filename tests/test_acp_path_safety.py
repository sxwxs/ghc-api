"""Session ids and machine names are used as path components, so they must not
be able to escape the sessions root."""

import pytest

from ghc_api.acp import session_manager as sm


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    """Point the sessions root at a temp dir and disable OneDrive lookups."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(sm, "get_config_dir", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr("ghc_api.utils.get_config_dir", lambda: str(tmp_path))
    monkeypatch.setattr("ghc_api.config_sync.onedrive_access_disabled", lambda: True)
    return root


@pytest.mark.parametrize(
    "session_id",
    [
        "../../../../etc/passwd",
        "..%2Fescape",  # already-decoded separators are covered below
        "sub/dir",
        "sub\\dir",
        "..",
        ".",
        "",
        "   ",
        "with\x00nul",
        "with\nnewline",
        "x" * 201,
        None,
        123,
    ],
)
def test_safe_session_id_rejects_unsafe_values(session_id):
    if session_id == "..%2Fescape":
        # Percent-encoding is not a separator once decoded by Werkzeug; the
        # decoded form ("../escape") is what must be rejected.
        session_id = "../escape"
    with pytest.raises(sm.InvalidPathComponent):
        sm._safe_session_id(session_id)


@pytest.mark.parametrize(
    "session_id",
    [
        "0d3f8a2e-1c4b-4f7a-9d1e-2b6c8a0f5e77",
        "sess_abc123",
        "session.2026-07-30",
    ],
)
def test_safe_session_id_accepts_agent_ids(session_id):
    assert sm._safe_session_id(session_id) == session_id


def test_safe_machine_name_allows_none_and_plain_names():
    assert sm._safe_machine_name(None) is None
    assert sm._safe_machine_name("host-1_linux") == "host-1_linux"


@pytest.mark.parametrize("machine", ["../other", "..", "", "a/b", "a b", "x" * 129])
def test_safe_machine_name_rejects_unsafe_values(machine):
    with pytest.raises(sm.InvalidPathComponent):
        sm._safe_machine_name(machine)


def test_resolve_session_path_rejects_traversal(sessions_root):
    manager = sm.SessionManager()
    with pytest.raises(sm.InvalidPathComponent):
        manager._resolve_session_path("../../secret")


def test_get_session_detail_rejects_traversal(sessions_root):
    manager = sm.SessionManager()
    with pytest.raises(sm.InvalidPathComponent):
        manager.get_session_detail("../../secret")


def test_get_session_storage_path_rejects_traversal(sessions_root):
    manager = sm.SessionManager()
    with pytest.raises(sm.InvalidPathComponent):
        manager.get_session_storage_path("../../secret")


def test_write_session_header_rejects_traversal(sessions_root):
    manager = sm.SessionManager()
    with pytest.raises(sm.InvalidPathComponent):
        manager._write_session_header("../escape", {"session_id": "../escape"})


def test_agent_routes_return_400_for_unsafe_identifiers(sessions_root):
    """The HTTP layer must translate the rejection into a 400, not a 500."""
    from ghc_api.app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    for url in (
        "/api/agent/sessions?machine=../other",
        "/api/agent/sessions/..%2F..%2Fsecret",
        "/api/agent/sessions/x/storage-path?machine=../other",
    ):
        response = client.get(url)
        assert response.status_code in (400, 404), (url, response.status_code)
        if response.status_code == 400:
            assert "Invalid session id or machine name" in response.get_data(as_text=True)
