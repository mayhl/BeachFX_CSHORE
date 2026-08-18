"""Tests for SnapshotLabel and StormResponseType enums."""

from erosion.types import SnapshotLabel, StormResponseType

# Canonical snapshot-label vocabulary — the C++/RAG label set (see the
# morphology-labels reference) plus the Python-model extensions the port adds
# (``Periodic``, ``INUNDATION``, ``RECN``, ``EENS``, ``ESNS``).  Pinned deliberately: changing a label
# is a contract change that should require editing this set too.
SNAPSHOT_LABELS = {
    "INIT",
    "PreStorm",
    "PostStorm",
    "INUNDATION",  # Python-model: CSHORE failure, profile reused
    "RECS",
    "REC",
    "RECN",  # Python-model: recovery cut short by the crew arriving to nourish
    "Pre-PDI",
    "Post-PDI",
    "SSN",
    "ESN",
    "SEN",
    "EEN",
    "EENS",  # Python-model: storm-triggered campaign cut short by the next storm
    "ESNS",  # Python-model: planned cycle cut short by the next storm
    "EndIteration",
    "Periodic",  # Python-model: inter-storm erosion tick
}


def test_snapshot_label_vocabulary():
    assert {label.value for label in SnapshotLabel} == SNAPSHOT_LABELS


def test_storm_response_type_contract():
    # Int values are a serialized contract (written to snapshots.parquet),
    # so both the mapping and the int-ness matter.
    assert [(t.name, int(t)) for t in StormResponseType] == [
        ("NORMAL", 0),
        ("CAT_DUNE_LOST", 1),
        ("CAT_PARTIAL", 2),
        ("CAT_TOTAL", 3),
        ("INUNDATION", 4),
    ]
    assert isinstance(StormResponseType.NORMAL, int)
