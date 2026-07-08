"""Pipeline orchestration helpers — the alternative run-selector."""

import pytest

from erosion.pipeline import _parse_run_spec


class TestParseRunSpec:
    AVAIL = ["1", "2", "3", "4", "8", "11", "12", "13", "14", "55", "FWOP", "FWP"]

    def test_all_returns_everything(self):
        assert _parse_run_spec("all", self.AVAIL) == self.AVAIL

    def test_all_case_insensitive_and_whitespace(self):
        assert _parse_run_spec("  ALL ", self.AVAIL) == self.AVAIL

    def test_none_returns_everything(self):
        assert _parse_run_spec(None, self.AVAIL) == self.AVAIL

    def test_single_id(self):
        assert _parse_run_spec("8", self.AVAIL) == ["8"]

    def test_range(self):
        assert _parse_run_spec("1-4", self.AVAIL) == ["1", "2", "3", "4"]

    def test_mixed_ranges_and_singletons(self):
        # the headline example
        assert _parse_run_spec("1-4,8,11-14,55", self.AVAIL) == [
            "1",
            "2",
            "3",
            "4",
            "8",
            "11",
            "12",
            "13",
            "14",
            "55",
        ]

    def test_explicit_string_ids(self):
        assert _parse_run_spec("FWOP,FWP", self.AVAIL) == ["FWOP", "FWP"]

    def test_mixed_int_and_string(self):
        assert _parse_run_spec("1-2,FWP", self.AVAIL) == ["1", "2", "FWP"]

    def test_order_follows_available_not_spec(self):
        # deterministic output order regardless of how the spec is written
        assert _parse_run_spec("8,1", self.AVAIL) == ["1", "8"]

    def test_list_spec(self):
        assert _parse_run_spec([1, 2, "FWP"], self.AVAIL) == ["1", "2", "FWP"]

    def test_whitespace_and_empty_tokens_tolerated(self):
        assert _parse_run_spec(" 1 , , 2 ", self.AVAIL) == ["1", "2"]

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="unknown alternative id"):
            _parse_run_spec("1-3,99", self.AVAIL)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown alternative id"):
            _parse_run_spec("FWXX", self.AVAIL)

    def test_descending_range_raises(self):
        with pytest.raises(ValueError, match="descending range"):
            _parse_run_spec("4-1", self.AVAIL)
