"""pyagent stays self-contained inside ghc_api: none of its modules may import the rest of ghc_api."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "ghc_api" / "pyagent"


def _escaping_imports(path: Path) -> list[str]:
    # ghc_api/pyagent/a/b.py (or a/__init__.py) resolves relative imports from package pyagent.a: `from ..` stays
    # inside pyagent, `from ...` leaves it. So the allowed level is the number of path parts.
    depth = len(path.relative_to(ROOT).parts)
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level > depth or (node.level == 0 and (node.module or "").split(".")[0] == "ghc_api"):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        elif isinstance(node, ast.Import):
            found += [
                f"{path.relative_to(ROOT)}:{node.lineno}" for a in node.names if a.name.split(".")[0] == "ghc_api"
            ]
    return found


def test_pyagent_does_not_import_the_rest_of_ghc_api() -> None:
    files = sorted(ROOT.rglob("*.py"))
    assert files
    assert [hit for path in files for hit in _escaping_imports(path)] == []
