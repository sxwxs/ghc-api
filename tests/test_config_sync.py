import unittest
from pathlib import Path
from unittest import mock

from ghc_api.config_sync import (
    ConfigEntry,
    _files_different,
    _hash_bytes_for_entry,
    _restore_codex_config_preserving_projects,
    _split_codex_config_sections,
)


class ConfigSyncCodexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codex_entry = ConfigEntry("codex", Path("unused"), "codex_config.toml")
        self.claude_entry = ConfigEntry("claude", Path("unused"), "claude_settings.json")

    def test_split_codex_config_without_projects(self) -> None:
        raw = b"model = \"gpt-5\"\napproval = \"on-request\"\n"
        header, projects = _split_codex_config_sections(raw)
        self.assertEqual(header, raw)
        self.assertEqual(projects, b"")

    def test_split_codex_config_with_projects(self) -> None:
        raw = (
            b"model = \"gpt-5\"\n"
            b"approval = \"on-request\"\n"
            b"[projects.\"C:\\\\src\\\\app\"]\n"
            b"trust_level = \"trusted\"\n"
        )
        header, projects = _split_codex_config_sections(raw)
        self.assertEqual(header, b"model = \"gpt-5\"\napproval = \"on-request\"")
        self.assertEqual(
            projects,
            b"[projects.\"C:\\\\src\\\\app\"]\ntrust_level = \"trusted\"\n",
        )

    def test_hash_bytes_for_codex_ignores_projects_section(self) -> None:
        left = (
            b"model = \"gpt-5\"\n"
            b"[projects.\"C:\\\\src\\\\a\"]\n"
            b"trust_level = \"trusted\"\n"
        )
        right = (
            b"model = \"gpt-5\"\n"
            b"[projects.\"C:\\\\src\\\\b\"]\n"
            b"trust_level = \"untrusted\"\n"
        )
        self.assertEqual(
            _hash_bytes_for_entry(self.codex_entry, left),
            _hash_bytes_for_entry(self.codex_entry, right),
        )

    def test_hash_bytes_for_non_codex_uses_full_content(self) -> None:
        raw = b"{\"foo\": 1}\n"
        self.assertEqual(_hash_bytes_for_entry(self.claude_entry, raw), raw)

    def test_files_different_for_codex_ignores_projects_only_diff(self) -> None:
        left = Path("left.toml")
        right = Path("right.toml")
        file_map = {
            left: b"model = \"gpt-5\"\n[projects.\"C:\\\\src\\\\a\"]\ntrust_level = \"trusted\"\n",
            right: b"model = \"gpt-5\"\n[projects.\"C:\\\\src\\\\b\"]\ntrust_level = \"untrusted\"\n",
        }
        with mock.patch("ghc_api.config_sync._read_bytes", side_effect=lambda p: file_map.get(p)):
            self.assertFalse(_files_different(self.codex_entry, left, right))

    def test_files_different_for_codex_detects_header_diff(self) -> None:
        left = Path("left.toml")
        right = Path("right.toml")
        file_map = {
            left: b"model = \"gpt-5\"\n[projects.\"C:\\\\src\\\\a\"]\n",
            right: b"model = \"gpt-4.1\"\n[projects.\"C:\\\\src\\\\a\"]\n",
        }
        with mock.patch("ghc_api.config_sync._read_bytes", side_effect=lambda p: file_map.get(p)):
            self.assertTrue(_files_different(self.codex_entry, left, right))

    def test_restore_codex_config_preserving_projects(self) -> None:
        source = Path("source.toml")
        target = Path("target.toml")
        file_map = {
            source: b"model = \"gpt-5\"\napproval = \"on-request\"\n[projects.\"C:\\\\src\\\\sync\"]\ntrust_level = \"trusted\"\n",
            target: b"model = \"old\"\n[projects.\"C:\\\\src\\\\local\"]\ntrust_level = \"untrusted\"\n",
        }
        expected = (
            b"model = \"gpt-5\"\n"
            b"approval = \"on-request\"\n"
            b"[projects.\"C:\\\\src\\\\local\"]\n"
            b"trust_level = \"untrusted\"\n"
        )

        with mock.patch("ghc_api.config_sync._read_bytes", side_effect=lambda p: file_map.get(p)):
            with mock.patch.object(Path, "write_bytes", autospec=True) as write_mock:
                _restore_codex_config_preserving_projects(source, target)
                write_mock.assert_called_once_with(target, expected)


if __name__ == "__main__":
    unittest.main()
