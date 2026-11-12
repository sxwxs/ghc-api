"""Resolves paths the model supplies against the working directory."""

import os
import re
import sys
from pathlib import Path


def resolve_path(path: str, cwd: str) -> Path:
    """Resolve against cwd. Accepts `~`, a leading `@` (some models prefix file mentions), and `/c/...` on Windows."""
    value = path.strip()
    if value.startswith("@"):
        value = value[1:]
    value = os.path.expanduser(value)
    if sys.platform == "win32" and (drive := re.match(r"^/([a-zA-Z])(/|$)", value)):
        value = f"{drive[1]}:/{value[3:]}"
    return Path(cwd, value)
