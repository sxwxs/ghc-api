"""Responses usage -> cache-record token counts (uncached-slice semantics)."""

import pytest

from ghc_api.utils import responses_usage_counts


def test_splits_cached_reads_and_cache_writes_out_of_input():
    counts = responses_usage_counts({
        "input_tokens": 100,
        "output_tokens": 7,
        "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 30},
    })
    assert counts == {
        "input_tokens": 10,
        "output_tokens": 7,
        "cache_creation_input_tokens": 30,
        "cache_read_input_tokens": 60,
    }


def test_without_details_input_is_all_uncached():
    assert responses_usage_counts({"input_tokens": 5, "output_tokens": 1}) == {
        "input_tokens": 5,
        "output_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


@pytest.mark.parametrize("usage", [
    None,
    "bad",
    {"input_tokens": None, "output_tokens": "3", "input_tokens_details": None},
    {"input_tokens": True, "input_tokens_details": {"cached_tokens": "x"}},
])
def test_malformed_usage_counts_as_zero(usage):
    assert responses_usage_counts(usage) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_inconsistent_details_never_go_negative():
    counts = responses_usage_counts({
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": 1, "cache_write_tokens": 1},
    })
    assert counts["input_tokens"] == 0
