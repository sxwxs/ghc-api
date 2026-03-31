"""
Config sync and backup helpers for Claude Code, Codex, and ghc-api.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import socket
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .state import state
from .utils import get_config_dir


TOOL_INSTALL_COMMANDS = {
    "codex": ["npm", "install", "-g", "@openai/codex"],
    "claude": ["npm", "install", "-g", "@anthropic-ai/claude-code"],
    "copilot-cli": ["npm", "install", "-g", "@github/copilot"],
}


SOFTWARE_VERSION_COMMANDS = [
    ("nodejs", "Node.js", [["node", "--version"]]),
    ("python", "Python", [["python", "--version"], ["py", "--version"]]),
    ("claude_code", "Claude Code", [["claude", "--version"]]),
    ("codex", "Codex", [["codex", "--version"]]),
    ("copilot_cli", "Copilot CLI", [["copilot", "--version"]]),
    ("visual_studio_code", "Visual Studio Code", [["code", "--version"]]),
]


@dataclass(frozen=True)
class ConfigEntry:
    key: str
    local_path: Path
    sync_filename: str


def _resolve_npm_executable() -> Optional[str]:
    if _is_wsl():
        # In WSL, prefer Linux npm and avoid Windows npm.cmd shims on /mnt/*.
        npm_native = shutil.which("npm")
        if npm_native:
            npm_path = Path(npm_native)
            if not str(npm_path).lower().endswith(".cmd") and not str(npm_path).startswith("/mnt/"):
                return str(npm_path)

        for candidate in ["/usr/bin/npm", "/usr/local/bin/npm", "/bin/npm"]:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                return str(candidate_path)
        return None

    # On Windows, npm is usually a .cmd shim and may not resolve as plain "npm".
    for candidate in ["npm.cmd", "npm"]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    if platform.system() == "Windows":
        common_paths = [
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "npm.cmd",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "npm.cmd",
        ]
        for path in common_paths:
            if path.exists():
                return str(path)

    return None


def _resolve_install_command(command: list[str]) -> tuple[list[str], Optional[str]]:
    if not command:
        return command, None
    if command[0] != "npm":
        return command, None

    npm_exec = _resolve_npm_executable()
    if npm_exec:
        return [npm_exec, *command[1:]], None

    return command, "npm was not found in PATH. Install Node.js/npm and restart the service process."


def _install_log_file() -> Path:
    return Path.home() / ".ghc-api" / "code_agent_install.log"


def _log_install_event(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line)
    try:
        log_path = _install_log_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[Config Sync] Failed to write install log file: {e}")


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()


def _windows_path_to_wsl_path(raw_path: str) -> Optional[Path]:
    value = raw_path.strip().strip('"')
    if not value:
        return None
    if value.startswith("/"):
        return Path(value)

    match = re.match(r"^([a-zA-Z]):[\\/](.*)$", value)
    if not match:
        return None

    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/").strip("/")
    if not tail:
        return Path(f"/mnt/{drive}")
    return Path(f"/mnt/{drive}/{tail}")


def _resolve_windows_home_from_wsl_interop() -> Optional[Path]:
    commands = [
        ["cmd.exe", "/c", "echo", "%USERPROFILE%"],
        ["powershell.exe", "-NoProfile", "-Command", "[Environment]::GetFolderPath('UserProfile')"],
    ]
    for command in commands:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3)
        except Exception:
            continue

        for line in (proc.stdout or "").splitlines():
            print(f"[Config Sync] WSL interop path candidate: {line}")
            home = _windows_path_to_wsl_path(line)
            print(home)

            if home:
                print(f"[Config Sync] Resolved Windows home from WSL interop: {home}")
                return home
    return None


def _prioritize_windows_user_homes(
    candidates: list[Path],
    preferred_home: Optional[Path],
    preferred_names: list[str],
) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()

    def add_candidate(candidate: Path) -> None:
        key = str(candidate).lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(candidate)

    if preferred_home:
        for candidate in candidates:
            if candidate == preferred_home:
                add_candidate(candidate)
                break

    for preferred_name in preferred_names:
        if not preferred_name:
            continue
        preferred_name_lower = preferred_name.lower()
        for candidate in candidates:
            if candidate.name.lower() == preferred_name_lower:
                add_candidate(candidate)

    for candidate in candidates:
        add_candidate(candidate)
    return ordered


def _resolve_windows_user_homes_from_wsl() -> list[Path]:
    users_root = Path("/mnt/c/Users")
    if not users_root.exists():
        return []
    preferred_home = _resolve_windows_home_from_wsl_interop()
    preferred_names = [name for name in [os.environ.get("USERNAME"), os.environ.get("USER")] if name]
    candidates = sorted([p for p in users_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if preferred_home and preferred_home.is_dir() and preferred_home not in candidates:
        candidates.insert(0, preferred_home)
    return _prioritize_windows_user_homes(candidates, preferred_home, preferred_names)


def onedrive_access_disabled() -> bool:
    return bool(state.disable_onedrive_access)


def get_onedrive_path() -> Optional[Path]:
    """Locate OneDrive root, preferring 'OneDrive - *' over 'OneDrive'."""
    if onedrive_access_disabled():
        return None
    if _is_wsl():
        base_homes = _resolve_windows_user_homes_from_wsl()
    else:
        base_homes = [Path.home()]

    for base_home in base_homes:
        if not base_home.exists():
            continue

        prefixed = sorted([p for p in base_home.glob("OneDrive - *") if p.is_dir()], key=lambda p: p.name.lower())
        plain = base_home / "OneDrive"

        if prefixed:
            return prefixed[0]
        if plain.is_dir():
            return plain
    return None


def get_sync_root() -> Optional[Path]:
    onedrive = get_onedrive_path()
    if not onedrive:
        return None
    return onedrive / ".ghc-api" / "configSync"


def _os_label() -> str:
    if _is_wsl():
        return "WSL"
    if platform.system() == "Windows":
        return "Win"
    return "Linux"


def get_agent_root() -> Optional[Path]:
    onedrive = get_onedrive_path()
    if not onedrive:
        return None
    host = socket.gethostname()
    return onedrive / ".ghc-api" / "agents" / f"{host}_{_os_label()}"


def get_machines_list() -> List[str]:
    """List available machine names from OneDrive agents directory. Current machine is always first."""
    current = f"{socket.gethostname()}_{_os_label()}"
    if onedrive_access_disabled():
        return [current]
    onedrive = get_onedrive_path()
    if not onedrive:
        return [current]
    agents_root = onedrive / ".ghc-api" / "agents"
    machines = []
    if agents_root.exists() and agents_root.is_dir():
        machines = sorted(
            [p.name for p in agents_root.iterdir() if p.is_dir()],
            key=lambda n: n.lower(),
        )
    # Always put current machine first
    machines = [m for m in machines if m != current]
    machines.insert(0, current)
    return machines


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def _read_bytes(path: Path) -> Optional[bytes]:
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


def _split_codex_config_sections(raw: bytes) -> tuple[bytes, bytes]:
    marker = b"\n[projects."
    idx = raw.find(marker)
    if idx == -1:
        if raw.startswith(b"[projects."):
            return b"", raw
        return raw, b""
    return raw[:idx], raw[idx + 1:]


def _hash_bytes_for_entry(entry: ConfigEntry, raw: bytes) -> bytes:
    if entry.key != "codex":
        return raw
    header, _ = _split_codex_config_sections(raw)
    return header


def _files_different(entry: ConfigEntry, left_path: Path, right_path: Path) -> bool:
    left_raw = _read_bytes(left_path)
    right_raw = _read_bytes(right_path)
    if left_raw is None or right_raw is None:
        return left_raw != right_raw
    return _hash_bytes_for_entry(entry, left_raw) != _hash_bytes_for_entry(entry, right_raw)


def _restore_codex_config_preserving_projects(source: Path, target: Path) -> None:
    source_raw = _read_bytes(source) or b""
    local_raw = _read_bytes(target) or b""
    source_header, _ = _split_codex_config_sections(source_raw)
    _, local_projects = _split_codex_config_sections(local_raw)
    merged = source_header + (b"\n" + local_projects if local_projects else b"")
    target.write_bytes(merged)


def _config_hash_text() -> str:
    entries = get_config_entries()
    sha1 = hashlib.sha1()
    for key in ["ghc-api", "claude", "codex"]:
        entry = entries[key]
        sha1.update(f"[{key}]".encode("utf-8"))
        if entry.local_path.exists():
            sha1.update(_hash_bytes_for_entry(entry, entry.local_path.read_bytes()))
        else:
            sha1.update(b"<missing>")
    return sha1.hexdigest()


def _latest_local_config_mtime(entries: Dict[str, ConfigEntry]) -> Optional[float]:
    mtimes: list[float] = []
    for entry in entries.values():
        if entry.local_path.exists():
            mtimes.append(entry.local_path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def _write_hash_file_if_stale(path: Path, latest_mtime: Optional[float]) -> tuple[bool, Optional[str]]:
    needs_refresh = not path.exists()
    if not needs_refresh and latest_mtime is not None:
        needs_refresh = latest_mtime > path.stat().st_mtime
    if not needs_refresh:
        return False, None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_config_hash_text(), encoding="utf-8")
        return True, None
    except Exception as e:
        return False, str(e)


def refresh_config_hash_files() -> Dict[str, object]:
    sync_root = get_sync_root()
    agent_root = get_agent_root()
    entries = get_config_entries()
    latest_mtime = _latest_local_config_mtime(entries)

    refreshed_paths: list[str] = []
    errors: list[str] = []
    hash_paths: Dict[str, Optional[str]] = {
        "config_sync_hash": None,
        "agent_hash": None,
    }

    if sync_root:
        sync_hash = sync_root / "config.sha1"
        hash_paths["config_sync_hash"] = str(sync_hash)
        refreshed, error = _write_hash_file_if_stale(sync_hash, latest_mtime)
        if refreshed:
            refreshed_paths.append(str(sync_hash))
        if error:
            errors.append(f"{sync_hash}: {error}")

    if agent_root:
        agent_hash = agent_root / "config.sha1"
        hash_paths["agent_hash"] = str(agent_hash)
        refreshed, error = _write_hash_file_if_stale(agent_hash, latest_mtime)
        if refreshed:
            refreshed_paths.append(str(agent_hash))
        if error:
            errors.append(f"{agent_hash}: {error}")

    return {
        "hash_paths": hash_paths,
        "refreshed_paths": refreshed_paths,
        "errors": errors,
    }


def _hash_file_info(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {
            "path": None,
            "exists": False,
            "hash": None,
            "created_ts": None,
            "created_at": None,
        }
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "hash": None,
            "created_ts": None,
            "created_at": None,
        }
    created_ts = int(path.stat().st_ctime)
    try:
        hash_value = path.read_text(encoding="utf-8").strip()
    except Exception:
        hash_value = None
    return {
        "path": str(path),
        "exists": True,
        "hash": hash_value,
        "created_ts": created_ts,
        "created_at": datetime.fromtimestamp(created_ts).isoformat(),
    }


def get_config_hash_overview() -> Dict[str, object]:
    refresh_config_hash_files()
    onedrive = get_onedrive_path()
    sync_root = get_sync_root()
    shared_hash_path = sync_root / "config.sha1" if sync_root else None

    machine_hashes = []
    if onedrive:
        agents_root = onedrive / ".ghc-api" / "agents"
        if agents_root.exists() and agents_root.is_dir():
            for machine_dir in sorted([p for p in agents_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
                preferred = machine_dir / "ghc-api" / "config.sha1"
                legacy = machine_dir / "config.sha1"
                selected = preferred if preferred.exists() else legacy
                info = _hash_file_info(selected)
                info["machine"] = machine_dir.name
                machine_hashes.append(info)

    return {
        "onedrive_access_disabled": onedrive_access_disabled(),
        "onedrive_path": str(onedrive) if onedrive else None,
        "shared_hash": _hash_file_info(shared_hash_path),
        "machines": machine_hashes,
    }


def get_config_entries() -> Dict[str, ConfigEntry]:
    ghc_config_dir = Path(get_config_dir())
    return {
        "claude": ConfigEntry("claude", Path.home() / ".claude" / "settings.json", "claude_settings.json"),
        "codex": ConfigEntry("codex", Path.home() / ".codex" / "config.toml", "codex_config.toml"),
        "ghc-api": ConfigEntry("ghc-api", ghc_config_dir / "config.yaml", "ghc_api_config.yaml"),
    }


def get_sync_status() -> Dict[str, object]:
    hash_refresh = refresh_config_hash_files()
    sync_root = get_sync_root()
    onedrive_path = get_onedrive_path()
    agent_root = get_agent_root()
    entries = get_config_entries()

    files: Dict[str, object] = {}
    has_sync_files = False
    has_differences = False

    for key, entry in entries.items():
        sync_path = sync_root / entry.sync_filename if sync_root else None
        local_exists = entry.local_path.exists()
        sync_exists = bool(sync_path and sync_path.exists())
        if sync_exists:
            has_sync_files = True

        different = False
        if sync_exists and local_exists:
            different = _files_different(entry, sync_path, entry.local_path)
        elif sync_exists != local_exists:
            different = True

        if different:
            has_differences = True

        files[key] = {
            "local_path": str(entry.local_path),
            "sync_path": str(sync_path) if sync_path else None,
            "local_exists": local_exists,
            "sync_exists": sync_exists,
            "different": different,
        }

    return {
        "onedrive_access_disabled": onedrive_access_disabled(),
        "onedrive_path": str(onedrive_path) if onedrive_path else None,
        "sync_root": str(sync_root) if sync_root else None,
        "agent_root": str(agent_root) if agent_root else None,
        "has_sync_files": has_sync_files,
        "has_differences": has_differences,
        "files": files,
        "hash_paths": hash_refresh["hash_paths"],
        "hash_refreshed": hash_refresh["refreshed_paths"],
        "hash_errors": hash_refresh["errors"],
    }


def print_sync_diff_status() -> Dict[str, object]:
    status = get_sync_status()
    if status.get("hash_errors"):
        for err in status["hash_errors"]:
            print(f"[Config Sync] Failed to update hash file: {err}")
    if status["onedrive_access_disabled"]:
        print("[Config Sync] OneDrive access disabled by config.")
        return status
    if not status["onedrive_path"]:
        print("[Config Sync] OneDrive path not found.")
        return status

    if not status["has_sync_files"]:
        print(f"[Config Sync] No synced config files found in {status['sync_root']}.")
        return status

    if status["has_differences"]:
        print("[Config Sync] Local config differs from synced config:")
        for key, file_status in status["files"].items():
            if file_status["different"]:
                print(f"  - {key}: local={file_status['local_path']} sync={file_status['sync_path']}")
    else:
        print("[Config Sync] Local config matches synced config.")
    return status


def sync_local_to_onedrive() -> Dict[str, object]:
    if onedrive_access_disabled():
        return {"ok": False, "error": "OneDrive access is disabled by config."}
    sync_root = get_sync_root()
    if not sync_root:
        return {"ok": False, "error": "OneDrive path not found."}

    sync_root.mkdir(parents=True, exist_ok=True)
    entries = get_config_entries()
    copied: Dict[str, str] = {}
    skipped: Dict[str, str] = {}

    for key, entry in entries.items():
        if not entry.local_path.exists():
            skipped[key] = "Local config file not found."
            continue
        target = sync_root / entry.sync_filename
        if key == "codex":
            target.parent.mkdir(parents=True, exist_ok=True)
            _restore_codex_config_preserving_projects(entry.local_path, target)
        else:
            shutil.copy2(entry.local_path, target)
        copied[key] = str(target)

    return {
        "ok": True,
        "sync_root": str(sync_root),
        "copied": copied,
        "skipped": skipped,
        "status": get_sync_status(),
    }


def sync_onedrive_to_local() -> Dict[str, object]:
    if onedrive_access_disabled():
        return {"ok": False, "error": "OneDrive access is disabled by config."}
    sync_root = get_sync_root()
    if not sync_root:
        return {"ok": False, "error": "OneDrive path not found."}
    if not sync_root.exists():
        return {"ok": False, "error": f"Sync folder not found: {sync_root}"}

    entries = get_config_entries()
    restored: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    backups: Dict[str, str] = {}

    for key, entry in entries.items():
        source = sync_root / entry.sync_filename
        if not source.exists():
            skipped[key] = "Synced file not found."
            continue

        entry.local_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = _backup_file(entry.local_path)
        if backup_path:
            backups[key] = str(backup_path)

        if key == "codex":
            _restore_codex_config_preserving_projects(source, entry.local_path)
        else:
            shutil.copy2(source, entry.local_path)
        restored[key] = str(entry.local_path)

    return {
        "ok": True,
        "sync_root": str(sync_root),
        "restored": restored,
        "skipped": skipped,
        "backups": backups,
        "status": get_sync_status(),
    }


def install_code_agents() -> Dict[str, object]:
    _log_install_event("[Config Sync] Starting code agent install/update.")
    entries = get_config_entries()
    backups: Dict[str, str] = {}

    # Back up target config files first, before any install command may modify them.
    for key in ["claude", "codex"]:
        backup = _backup_file(entries[key].local_path)
        if backup:
            backups[key] = str(backup)
            _log_install_event(f"[Config Sync] Backed up {key} config: {backup}")

    command_results: Dict[str, object] = {}
    failures = 0

    for tool_name, command in TOOL_INSTALL_COMMANDS.items():
        resolved_command, resolution_error = _resolve_install_command(command)
        _log_install_event(f"[Config Sync] Installing {tool_name}: {' '.join(resolved_command)}")
        if resolution_error:
            failures += 1
            _log_install_event(f"[Config Sync] {tool_name} install failed before execution: {resolution_error}")
            command_results[tool_name] = {
                "command": " ".join(command),
                "resolved_command": " ".join(resolved_command),
                "returncode": -1,
                "stdout": "",
                "stderr": resolution_error,
            }
            continue

        try:
            proc = subprocess.run(resolved_command, capture_output=True, text=True, check=False)
            command_results[tool_name] = {
                "command": " ".join(command),
                "resolved_command": " ".join(resolved_command),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
            if proc.returncode != 0:
                failures += 1
                err_preview = (proc.stderr or "").strip().splitlines()
                err_preview_text = err_preview[-1] if err_preview else "unknown error"
                _log_install_event(f"[Config Sync] {tool_name} install failed (rc={proc.returncode}): {err_preview_text}")
            else:
                _log_install_event(f"[Config Sync] {tool_name} install succeeded.")
        except Exception as e:
            failures += 1
            _log_install_event(f"[Config Sync] {tool_name} install exception: {e}")
            command_results[tool_name] = {
                "command": " ".join(command),
                "resolved_command": " ".join(resolved_command),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }

    if failures == 0:
        _log_install_event("[Config Sync] Code agent install/update completed successfully.")
    else:
        _log_install_event(f"[Config Sync] Code agent install/update completed with {failures} failure(s).")

    return {
        "ok": failures == 0,
        "log_file": str(_install_log_file()),
        "backups": backups,
        "results": command_results,
    }


def _probe_version_command(command: list[str]) -> Optional[Dict[str, object]]:
    executable = shutil.which(command[0])
    if not executable:
        return None
    resolved_command = [executable, *command[1:]]

    try:
        proc = subprocess.run(
            resolved_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception as e:
        return {
            "installed": True,
            "version": "unknown",
            "detail": str(e),
            "command": " ".join(resolved_command),
        }

    combined = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    first_line = ""
    for line in combined.splitlines():
        clean = line.strip()
        if clean:
            first_line = clean
            break

    version_text = first_line or "unknown"
    if proc.returncode != 0:
        version_text = f"error (rc={proc.returncode})"

    return {
        "installed": True,
        "version": version_text,
        "detail": combined[-2000:],
        "command": " ".join(resolved_command),
    }


def get_software_versions() -> Dict[str, object]:
    tools = []

    for tool_id, name, candidates in SOFTWARE_VERSION_COMMANDS:
        result = None
        for command in candidates:
            result = _probe_version_command(command)
            if result:
                break

        if not result:
            tools.append({
                "id": tool_id,
                "name": name,
                "installed": False,
                "version": "Not installed",
                "command": " / ".join(" ".join(c) for c in candidates),
                "detail": "",
            })
            continue

        tools.append({
            "id": tool_id,
            "name": name,
            "installed": bool(result["installed"]),
            "version": result["version"],
            "command": result["command"],
            "detail": result["detail"],
        })

    return {
        "tools": tools,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }
