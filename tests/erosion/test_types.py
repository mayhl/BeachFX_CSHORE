"""Tests for SnapshotLabel and StormResponseType enums."""
from erosion.types import SnapshotLabel, StormResponseType

# Canonical snapshot-label vocabulary — must stay in sync with the C++/RAG label
# set (see the morphology-labels reference). Pinned deliberately: changing a
# label is a contract change that should require editing this set too.
CPLUS_LABELS = {
    "INIT", "PreStorm", "PostStorm", "INUNDATION", "RECS", "REC",
    "Pre-PDI", "Post-PDI", "SSN", "ESN", "SEN", "EEN",
    "EndIteration", "Periodic",
}


def test_snapshot_label_vocabulary():
    assert {label.value for label in SnapshotLabel} == CPLUS_LABELS


def test_storm_response_type_contract():
    # Int values are a serialized contract (written to profile_events.parquet),
    # so both the mapping and the int-ness matter.
    assert [(t.name, int(t)) for t in StormResponseType] == [
        ("NORMAL", 0), ("CAT_DUNE_LOST", 1), ("CAT_PARTIAL", 2),
        ("CAT_TOTAL", 3), ("INUNDATION", 4),
    ]
    assert isinstance(StormResponseType.NORMAL, int)
