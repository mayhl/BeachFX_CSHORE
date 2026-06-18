"""Tests for SnapshotLabel and StormResponseType enums."""
import pytest
from framework.types import SnapshotLabel, StormResponseType

CPLUS_LABELS = {
    "INIT", "PreStorm", "PostStorm", "INUNDATION", "RECS", "REC",
    "Pre-PDI", "Post-PDI", "SSN", "ESN", "SEN", "EEN",
    "EndIteration", "Periodic",
}


class TestSnapshotLabel:
    def test_all_labels_present(self):
        actual = {label.value for label in SnapshotLabel}
        assert actual == CPLUS_LABELS

    def test_no_extra_labels(self):
        actual = {label.value for label in SnapshotLabel}
        assert not (actual - CPLUS_LABELS)


class TestStormResponseType:
    def test_values(self):
        assert StormResponseType.NORMAL        == 0
        assert StormResponseType.CAT_DUNE_LOST == 1
        assert StormResponseType.CAT_PARTIAL   == 2
        assert StormResponseType.CAT_TOTAL     == 3
        assert StormResponseType.INUNDATION    == 4

    def test_is_int(self):
        assert isinstance(StormResponseType.NORMAL, int)

    def test_ordering(self):
        assert StormResponseType.NORMAL < StormResponseType.CAT_DUNE_LOST
        assert StormResponseType.CAT_DUNE_LOST < StormResponseType.CAT_PARTIAL
        assert StormResponseType.CAT_PARTIAL < StormResponseType.CAT_TOTAL
        assert StormResponseType.CAT_TOTAL < StormResponseType.INUNDATION
