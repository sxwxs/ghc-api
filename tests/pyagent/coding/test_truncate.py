"""Head and tail truncation by lines and bytes."""

from ghc_api.pyagent.coding.tools.truncate import format_size, truncate_head, truncate_tail


def test_short_text_is_untouched() -> None:
    result = truncate_head("a\nb\n")
    assert (result.content, result.truncated, result.total_lines) == ("a\nb\n", False, 2)


def test_head_keeps_whole_leading_lines_within_both_limits() -> None:
    text = "\n".join(f"line{i}" for i in range(10))
    by_lines = truncate_head(text, max_lines=3)
    assert (by_lines.content, by_lines.output_lines, by_lines.total_lines) == ("line0\nline1\nline2", 3, 10)
    by_bytes = truncate_head(text, max_bytes=12)
    assert by_bytes.content == "line0\nline1"


def test_head_reports_a_first_line_that_cannot_fit() -> None:
    result = truncate_head("x" * 100 + "\nshort", max_bytes=10)
    assert result.first_line_too_long and result.content == ""


def test_tail_keeps_whole_trailing_lines() -> None:
    text = "\n".join(f"line{i}" for i in range(10))
    assert truncate_tail(text, max_lines=2).content == "line8\nline9"


def test_tail_keeps_the_end_of_an_oversized_last_line_without_splitting_characters() -> None:
    result = truncate_tail("first\n" + "中" * 10, max_bytes=7)
    assert result.last_line_partial
    assert result.content == "中中"


def test_format_size() -> None:
    assert (format_size(10), format_size(2048), format_size(3 * 1024 * 1024)) == ("10B", "2.0KB", "3.0MB")
