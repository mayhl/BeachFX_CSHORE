"""Event-log capture tests: the per-profile append-only ``EventRecord`` stream.

Complements ``test_event_sequences`` (which asserts snapshot-label ordering) by
checking the parallel ``profile.events`` log that the Phase-A output spine
persists — event types, scalar payloads, and the inert ``ref_pos`` seam.
Physics-independent via ``MockCSStorm``.
"""

from __future__ import annotations

import numpy as np

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.profile import Profile, StormResponse
from erosion.runner.base import CSHOREResult
from erosion.runner.mock import MockCSStorm
from erosion.types import SnapshotLabel as L
from tests.builders import RecordingSink, ncfg, run, storms_at, template_profile

# Observation snapshots (not state transitions) carry no event-log entry.
_OBSERVATION_LABELS = {L.INIT.value, L.PreStorm.value, L.EndIteration.value}


def _etypes(p) -> list[str]:
    return [e.event_type for e in p.events]


def _nourish_cfg(production_rate: float = 100.0) -> ReachConfig:
    return ReachConfig(
        nourishment=ncfg(volume_trigger=30.0, production_rate=production_rate, assessor="volume")
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
    p = _run_one(MockCSStorm(0.2), _nourish_cfg())  # small chunk -> recover, no nourish
    types = _etypes(p)
    assert "StormResponse" in types
    assert "Recovery" in types
    rec = next(e for e in p.events if e.event_type == "Recovery")
    assert set(rec.payload) == {"fraction", "interrupted"}


def test_full_nourishment_logged():
    p = _run_one(MockCSStorm(1.0), _nourish_cfg())  # big chunk -> campaign
    types = _etypes(p)
    assert "NourishmentStart" in types  # SSN marker
    assert "FullNourishment" in types  # ESN completion


def test_inundation_logged():
    p = _run_one(MockCSStorm("inundation"), _nourish_cfg())
    assert "Inundation" in _etypes(p)


def test_erosion_tick_logged_with_payload():
    cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, interval=10.0))
    p = _run_one(MockCSStorm(0.5), cfg)
    ticks = [e for e in p.events if e.event_type == "ErosionTick"]
    assert ticks
    assert set(ticks[0].payload) == {"dz_erosion", "dz_slc"}


def test_storm_response_payload_carries_valid_domain_metadata():
    p = _run_one(MockCSStorm(0.2), _nourish_cfg())
    sr = next(e for e in p.events if e.event_type == "StormResponse")
    assert set(sr.payload) >= {"runup_m", "jr", "n_extrapolated"}
    # MockCSStorm returns result.x == profile.x -> nothing extrapolated, whole domain wet
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
    p = _run_one(MockCSStorm(1.0), _nourish_cfg())
    snap_labels = {s.label.value for s in p.snapshots} - _OBSERVATION_LABELS
    event_labels = {e.label for e in p.events if e.label is not None}
    assert event_labels == snap_labels


def test_ref_pos_seam_is_inert():
    # ``ref_pos`` is recorded but unused until Phase C — every entry is None.
    p = _run_one(MockCSStorm(1.0), _nourish_cfg())
    assert p.events
    assert all(e.ref_pos is None for e in p.events)


def test_events_are_time_ordered():
    p = _run_one(MockCSStorm(1.0), _nourish_cfg())
    ts = [e.t for e in p.events]
    assert ts == sorted(ts)
