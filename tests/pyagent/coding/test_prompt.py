"""System prompt assembly and context file discovery."""

from pathlib import Path

from ghc_api.pyagent.coding.prompt import build_system_prompt, load_context_files
from ghc_api.pyagent.coding.tools.edit import EditTool
from ghc_api.pyagent.coding.tools.read import ReadTool


def test_context_files_come_from_home_then_root_down_to_cwd(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "repo" / "pkg"
    project.mkdir(parents=True)
    home.mkdir()
    (home / "AGENTS.md").write_text("global")
    (tmp_path / "repo" / "AGENTS.md").write_text("repo")
    (tmp_path / "repo" / "CLAUDE.md").write_text("ignored: AGENTS.md wins")
    (project / "CLAUDE.md").write_text("pkg")
    files = load_context_files(str(project), home)
    assert [content for _, content in files] == ["global", "repo", "pkg"]


def test_prompt_lists_tools_rules_context_and_environment(tmp_path: Path) -> None:
    prompt = build_system_prompt(str(tmp_path), [ReadTool(), EditTool()], [("/x/AGENTS.md", "Use tabs.")])
    assert "- read: Read file contents" in prompt
    assert "- Use edit for targeted changes. Batch changes to one file in a single call." in prompt
    assert '<project_instructions path="/x/AGENTS.md">\nUse tabs.\n</project_instructions>' in prompt
    assert f"Working directory: {tmp_path.as_posix()}" in prompt
