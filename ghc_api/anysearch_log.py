"""Daily AnySearch audit records. Callers must supply only sanitized responses."""

import json
from datetime import datetime
from pathlib import Path


def record_search_to_file(entry: dict) -> None:
    from .state import state
    from .utils import get_config_dir

    if not state.log_anysearch_requests:
        return
    try:
        directory = Path(get_config_dir()) / "anysearch"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{datetime.now():%Y-%m-%d}.jl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Do not interpolate exceptions: even a logging failure must not expose
        # a response, key, or private filesystem path.
        print("[AnySearch] Failed to append audit record")
