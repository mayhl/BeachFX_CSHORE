"""Event-log capture tests: the per-profile append-only ``EventRecord`` stream.

Complements ``test_event_sequences`` (which asserts snapshot-label ordering) by
checking the parallel ``profile.events`` log that the Phase-A output spine
persists — event types, scalar payloads, and the inert ``ref_pos`` seam.
Physics-independent via ``ScriptedRunner``.
"""

from __future__ import annotations

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.profile import Profile, StormResponse
from erosion.runner.base import CSHOREResult
from erosion.types import SnapshotLabel as L
from tests.builders import RecordingSink, ncfg, run, storms_at, template_profile
from tests.doubles import MASSIVE, MINOR, NONE, SEVERE, Inundation, ScriptedRunner

# Observation snapshots (not state transitions) carry no event-log entry.
_OBSERVATION_LABELS = {L.INIT.value, L.PreStorm.value, L.EndIteration.value}


def _etypes(p) -> list[str]:
    return [e.event_type for e in p.events]


# z_berm masks recovery to the sub-berm face — without it recovery blends the dune
# toward the shoreline-shifted pre-storm bed (see test_event_sequences._STORM).
_STORM = {"z_berm": 2.0}


def _nourish_cfg(production_rate: float = 100.0) -> ReachConfig:
    return ReachConfig(
        storm=_STORM,
        nourishment=ncfg(volume_trigger=30.0, production_rate=production_rate, assessor="volume"),
    )


def _run_one(runner, cfg):
    (profiles, _sink) = run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=runner,
        cfg=cfg,
        sink=RecordingSink(),
    )
    return profiles[0]


def test_storm_and_recovery_logged():
    p = _run_one(ScriptedRunner(MINOR), _nourish_cfg())  # under the gate -> recover, no nourish
    types = _etypes(p)
    assert "StormResponse" in types
    assert "Recovery" in types
    rec = next(e for e in p.events if e.event_type == "Recovery")
    assert set(rec.payload) == {"fraction", "interrupted"}


def test_full_nourishment_logged():
    p = _run_one(ScriptedRunner(SEVERE), _nourish_cfg())  # over the gate -> campaign
    types = _etypes(p)
    assert "NourishmentStart" in types  # SEN marker
    assert "FullNourishment" in types  # EEN completion


def test_inundation_logged():
    p = _run_one(ScriptedRunner(Inundation()), _nourish_cfg())
    assert "Inundation" in _etypes(p)


def test_erosion_tick_logged_with_payload():
    cfg = ReachConfig(storm=_STORM, erosion=UniformErosionConfig(rate=0.01, tick_days=10.0))
    p = _run_one(ScriptedRunner(MINOR), cfg)
    ticks = [e for e in p.events if e.event_type == "ErosionTick"]
    assert ticks
    assert set(ticks[0].payload) == {"dz_erosion", "dz_slc", "dt_days", "held"}
    # the storm's recovery holds that gap's ticks -> exactly one held catch-up tick
    held = [e for e in ticks if e.payload["held"]]
    assert len(held) == 1
    assert held[0].payload["dt_days"] == pytest.approx(21.001)  # offset + T_recover


def test_storm_response_payload_carries_valid_domain_metadata():
    p = _run_one(ScriptedRunner(MINOR), _nourish_cfg())
    sr = next(e for e in p.events if e.event_type == "StormResponse")
    assert set(sr.payload) >= {"runup_m", "jr", "n_extrapolated"}
    # ScriptedRunner returns result.x == profile.x -> nothing extrapolated, whole domain wet
    assert sr.payload["n_extrapolated"] == 0
    assert sr.payload["jr"] == len(p.x)


def test_storm_response_counts_landward_extrapolated_nodes():
    # CSHORE returns a grid short by 2 landward nodes -> StormResponse.apply constant-
    # extrapolates them, and the event records the count + the JR wet limit.
    x = np.arange(0.0, 10.0)
    p = Profile("p", x.copy(), np.zeros(10), 0.3)
    r = CSHOREResult(
        zb=np.zeros(8), x=np.arange(0.0, 8.0), eta=np.zeros(8), Hs=np.zeros(8), runup_m=0.0, jr=6
    )
    StormResponse(t=1.0, result=r).apply(p)
    ev = p.events[-1]
    assert ev.event_type == "StormResponse"
    assert ev.payload["n_extrapolated"] == 2  # nodes at x=8,9 beyond CSHORE's returned grid
    assert ev.payload["jr"] == 6


def test_event_labels_match_transition_snapshots():
    # Every mutating snapshot has a matching event; observation snapshots
    # (INIT / PreStorm / EndIteration) do not.  No partial nourishment here, so
    # every event carries a snapshot label.
    p = _run_one(ScriptedRunner(SEVERE), _nourish_cfg())
    snap_labels = {s.label.value for s in p.snapshots} - _OBSERVATION_LABELS
    event_labels = {e.label for e in p.events if e.label is not None}
    assert event_labels == snap_labels


def test_ref_pos_seam_is_inert():
    # ``ref_pos`` is recorded but unused until Phase C — every entry is None.
    p = _run_one(ScriptedRunner(SEVERE), _nourish_cfg())
    assert p.events
    assert all(e.ref_pos is None for e in p.events)


def test_events_are_time_ordered():
    p = _run_one(ScriptedRunner(SEVERE), _nourish_cfg())
    ts = [e.t for e in p.events]
    assert ts == sorted(ts)


def test_events_parquet_roundtrips_full_lifecycle():
    """The persisted ``events.parquet`` reproduces the in-memory event log for a
    full run through the sink — type / time / label / scalar payload all survive
    serialization, across a rich nourish+interrupt lifecycle.

    Uses the interrupt scenario so the stream spans every event type, including a
    ``PartialNourishment`` whose ``label`` is ``None`` (the sparse-column /
    null-label edge the count-only round-trip in ``test_run_lifecycle`` misses).
    """
    import os
    import tempfile

    import pandas as pd

    from erosion.results import ParquetResultsSink

    with tempfile.TemporaryDirectory() as root:
        sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
        profiles, _ = run(
            [template_profile("p0")],
            storms_df=storms_at([20, 34]),
            sim_end=100.0,
            runner=ScriptedRunner({"p0": [MASSIVE, NONE]}),  # ~15.5d fill vs 13.5d gap → partial
            cfg=_nourish_cfg(production_rate=4.0),
            sink=sink,
        )
        p = profiles[0]
        df = pd.read_parquet(os.path.join(sink.out_dir, "events.parquet")).sort_values("event_seq")

        # one row per applied event, same order — type and time round-trip exactly
        assert len(df) == len(p.events)
        assert list(df["event_type"]) == [e.event_type for e in p.events]
        assert list(df["t"]) == pytest.approx([e.t for e in p.events])

        # label round-trips, including the None the partial placement contributes
        def _none(v):
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

        assert [_none(v) for v in df["label"]] == [e.label for e in p.events]
        assert "PartialNourishment" in set(df["event_type"])  # the label=None row is present

        # scalar payloads survive the sparse-column flatten (a populated payload row
        # coexists with the empty-payload NourishmentStart/FullNourishment rows) —
        # spot-check a float (fraction) and the valid-domain ints (jr / n_extrapolated).
        pn_df, pn_mem = (
            df[df["event_type"] == "PartialNourishment"].iloc[0],
            next(e for e in p.events if e.event_type == "PartialNourishment"),
        )
        assert pn_df["fraction"] == pytest.approx(pn_mem.payload["fraction"])

        sr_df, sr_mem = (
            df[df["event_type"] == "StormResponse"].iloc[0],
            next(e for e in p.events if e.event_type == "StormResponse"),
        )
        assert sr_df["jr"] == sr_mem.payload["jr"]
        assert sr_df["n_extrapolated"] == sr_mem.payload["n_extrapolated"]


# ---------------------------------------------------------------------------
# EVENT_SCHEMA registry — closed vocabulary + pinned parquet schema
# ---------------------------------------------------------------------------


def test_registry_closes_the_vocabulary():
    """Every ``ProfileEvent`` subclass is registered and emits exactly its
    registered payload keys.

    ``StormResponse`` hand-rolls its ``record_event`` inside ``apply`` (the payload
    needs the CSHORE result), so its keys are pinned by the runtime check plus the
    round-trip test above rather than by ``_payload()`` here.
    """
    from erosion.profile import (
        EVENT_SCHEMA,
        ErosionTick,
        FullNourishment,
        PartialNourishment,
        ProfileEvent,
        Recovery,
    )

    assert {c.event_type for c in ProfileEvent.__subclasses__()} <= set(EVENT_SCHEMA)

    zb = np.zeros(3)
    for ev in (
        ErosionTick(t=0.0),
        Recovery(t=0.0, fraction=0.5, zb_post_storm=zb, zb_pre_storm=zb),
        FullNourishment(t=0.0, template_zb=zb),
        PartialNourishment(t=0.0, template_zb=zb, fraction=0.5),
    ):
        assert set(ev._payload()) == set(EVENT_SCHEMA[ev.event_type]), ev.event_type


def test_record_event_rejects_unregistered_type():
    with pytest.raises(ValueError, match="unregistered"):
        template_profile("p0").record_event("StromResponse", 0.0)


def test_record_event_rejects_wrong_payload_keys():
    with pytest.raises(ValueError, match="payload keys"):
        template_profile("p0").record_event("Recovery", 0.0, fraction=0.5)


def test_events_parquet_columns_pinned():
    """``events.parquet`` carries base + ``EVENT_PAYLOAD_COLUMNS`` regardless of
    which events fired — an all-inundation run (every CSHORE call fails) emits no
    StormResponse/Recovery, yet still writes their columns (all-null), so
    downstream readers see one schema."""
    import os
    import tempfile

    import pandas as pd

    from erosion.profile import EVENT_PAYLOAD_COLUMNS
    from erosion.results import ParquetResultsSink

    base = ["profile_id", "event_seq", "event_type", "t", "label", "ref_pos"]
    with tempfile.TemporaryDirectory() as root:
        sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
        run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            runner=ScriptedRunner(Inundation()),
            cfg=ReachConfig(storm=_STORM),
            sink=sink,
        )
        df = pd.read_parquet(os.path.join(sink.out_dir, "events.parquet"))
        assert list(df.columns) == base + list(EVENT_PAYLOAD_COLUMNS)
        # no storm response fired, so its payload columns exist but are all null
        assert set(df["event_type"]) <= {"ErosionTick", "Inundation"}
        assert df["runup_m"].isna().all()
