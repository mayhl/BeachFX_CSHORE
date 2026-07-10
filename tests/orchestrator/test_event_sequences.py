"""Orchestrator event-generation tests.

These exercise ``run_lifecycle`` end-to-end and assert the *sequence* of events
(snapshot labels) each profile accumulates — i.e. that the orchestrator emits
the right ordering of PreStorm / PostStorm / REC / SSN / ESN across a storm
schedule.  They test event handling, not the nourishment/recovery numerics.

CSHORE is the only mock: ``MockCSStorm(depth)`` scoops a uniform chunk off the
bed, and the *real* ``VolumeAssessor`` turns the emergent deficit into a
campaign (bigger chunk / slower ``production_rate`` → longer, interruptible
placement).  Blackout scenarios use ``blackout_windows``.

Each ``Case`` declares the full expected label sequence per profile, optionally
time-pinning entries as ``(L, t)``; ``durations`` assert intervals in days.  Run
with ``--events-table`` to print a per-scenario table (event / elapsed / date /
OK, plus reach decisions + intervals).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.profile import Profile
from erosion.runner.mock import MockCSStorm
from erosion.types import DecisionKind as D
from erosion.types import SnapshotLabel as L
from erosion.types import StormResponseType
from tests.builders import SIM_START, RecordingSink, ncfg, run, storms_at, template_profile

TRIGGER = 30.0  # deficit trigger: chunk depth 0.2→17.9 m³ (recover) vs ≥1.0→77.5 m³ (nourish)


# --- trace helpers ---------------------------------------------------------


def labels(p) -> list[L]:
    return [s.label for s in p.snapshots]


def times_of(p, label: L) -> list[float]:
    return [s.t for s in p.snapshots if s.label == label]


def _nourish_cfg(production_rate=500.0) -> ReachConfig:
    return ReachConfig(
        nourishment=ncfg(volume_trigger=TRIGGER, production_rate=production_rate, assessor="volume")
    )


# --- scenario factories → (profiles, RecordingSink) after one lifecycle -----
# Profiles start on their as-built restore geometry (deficit ≈ 0); MockCSStorm(depth)
# scoops a chunk → the real assessor sizes the campaign. Sink captures reach decisions.

Made = tuple[list[Profile], RecordingSink]


def _recovery() -> Made:
    # nourishment configured, but chunk 0.2 → deficit ~16.6 < trigger 30 → NOURISH_SKIP + recover
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=MockCSStorm(0.2),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _nourish_single() -> Made:
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=MockCSStorm(1.0),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _nourish_multi() -> Made:
    return run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20]),
        runner=MockCSStorm(1.0),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _interrupt_single() -> Made:
    # Storms 14 days apart; a big chunk at a slow rate can't finish in 14d → interrupt.
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20, 34]),
        sim_end=100.0,
        runner=MockCSStorm({"p0": [2.0, 0.2]}),
        cfg=_nourish_cfg(production_rate=5.0),
        sink=RecordingSink(),
    )


def _defer_single() -> Made:
    # Same setup as _interrupt_single, but the BeachFX "defer" policy delays the
    # storm-hit placement to the next storm-free gap instead of splitting it.
    cfg = _nourish_cfg(production_rate=5.0)
    cfg.nourishment.storm_conflict = "defer"
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20, 34]),
        sim_end=100.0,
        runner=MockCSStorm({"p0": [2.0, 0.2]}),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _interrupt_multi() -> Made:
    return run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20, 34]),
        sim_end=140.0,
        runner=MockCSStorm({"p0": [2.0, 0.2], "p1": [2.0, 0.2]}),
        cfg=_nourish_cfg(production_rate=5.0),
        sink=RecordingSink(),
    )


def _periodic() -> Made:
    cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, interval=10.0))
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=MockCSStorm(0.5),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _inundation() -> Made:
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=MockCSStorm("inundation"),
        cfg=_nourish_cfg(),
        sink=RecordingSink(),
    )


def _blackout() -> Made:
    cfg = _nourish_cfg(production_rate=100.0)
    cfg.nourishment.blackout_windows = [(20.0, 50.0)]
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        sim_end=120.0,
        runner=MockCSStorm(1.0),
        cfg=cfg,
        sink=RecordingSink(),
    )


# --- expected full sequences per scenario ----------------------------------

_I, _PRE, _POST = L.INIT, L.PreStorm, L.PostStorm
_REC, _RECS, _SSN, _ESN, _END = L.REC, L.RECS, L.SSN, L.ESN, L.EndIteration
_INUN = L.INUNDATION
_PER = L.Periodic

# reach-scope decision aliases
_TRIG, _SKIP, _DEFER, _INTR, _SDEFER = (
    D.NOURISH_TRIGGER,
    D.NOURISH_SKIP,
    D.BLACKOUT_DEFER,
    D.INTERRUPT,
    D.STORM_DEFER,
)


@dataclass
class Case:
    id: str
    desc: str
    make: Callable[[], tuple[list[Profile], RecordingSink]]
    expected: dict[str, list]  # per profile: a sequence of L or (L, t) time-pinned entries
    # interval assertions in days: (profile_id, from_idx, to_idx, days)
    durations: list = field(default_factory=list)
    # reach-scope decision kinds, in emission order (asserted once per scenario)
    decisions: list = field(default_factory=list)
    reach_id: str = "test"  # matches builders.run() default; shown in the events table


CASES = [
    Case(
        "1_recovery",
        "sub-trigger deficit → NOURISH_SKIP + recovery",
        _recovery,
        {"p0": [_I, (_PRE, 20.0), (_POST, 20.5), _REC, _END]},
        decisions=[_SKIP],
    ),
    Case(
        "2_nourish_single",
        "chunk deficit → single nourishment",
        _nourish_single,
        {"p0": [_I, _PRE, (_POST, 20.5), (_SSN, 20.5), _ESN, _END]},
        decisions=[_TRIG],
    ),
    Case(
        "3_nourish_multi",
        "multi-profile nourishment, serial crew",
        _nourish_multi,
        {
            "p0": [_I, _PRE, (_POST, 20.5), (_SSN, 20.5), _ESN, _END],
            "p1": [_I, _PRE, (_POST, 20.5), _REC, _SSN, _ESN, _END],
        },
        decisions=[_TRIG],  # one reach-level trigger; crew serves both profiles
    ),
    Case(
        "4_interrupt_single",
        "nourishment interrupted 14 days in, then resumes",
        _interrupt_single,
        # SSN starts at the storm (t=20); the 2nd storm interrupts 14 days later
        # (t=34); the campaign resumes there (SSN t=34). Placement length is emergent.
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_SSN, 20.5),
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SSN, 34.5),
                _ESN,
                _END,
            ]
        },
        durations=[("p0", 3, 6, 14.0)],  # SSN(start) → SSN(resume) = 14-day interrupt gap
        decisions=[_TRIG, _INTR, _TRIG],  # launch → interrupted → resume
    ),
    Case(
        "5_interrupt_multi",
        "multi-profile campaign interrupted",
        _interrupt_multi,
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_SSN, 20.5),
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SSN, 34.5),
                _ESN,
                _END,
            ],
            "p1": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_RECS, 34.0),  # 2nd storm cuts recovery short (13.5d < T_recover 21d)
                _PRE,
                (_POST, 34.5),
                _REC,  # recovery up to this profile's own nourishment start (not storm-forced)
                _SSN,
                _ESN,
                _END,
            ],
        },
        durations=[("p0", 3, 6, 14.0)],
        decisions=[_TRIG, _INTR, _TRIG],
    ),
    Case(
        "7_blackout",
        "nourishment deferred past a blackout window",
        _blackout,
        {"p0": [_I, _PRE, (_POST, 20.5), _REC, (_SSN, 50.0), _ESN, _END]},
        decisions=[_TRIG, _DEFER],
    ),
    Case(
        "8_inundation",
        "CSHORE failure → storm skipped, no recovery (INUNDATION)",
        _inundation,
        {"p0": [_I, _PRE, _INUN, _END]},
        decisions=[_SKIP],  # inundated profile excluded → empty plans → sub-trigger skip
    ),
    Case(
        "9_periodic",
        "erosion ticks continue through every inter-storm gap (pre- AND post-storm)",
        _periodic,
        # ticks @10,@20 before the storm; the storm ends at 20.5, so the tail ticks
        # @30.5…@80.5 (6 ticks, one fewer than a storm-start window) before REC
        {
            "p0": [
                _I,
                (_PER, 10.0),
                (_PER, 20.0),
                (_PRE, 20.0),
                (_POST, 20.5),
                (_PER, 30.5),
                _PER,
                _PER,
                _PER,
                _PER,
                (_PER, 80.5),
                _REC,
                _END,
            ]
        },
        decisions=[],  # no nourishment configured
    ),
    Case(
        "10_storm_defer",
        "storm-conflicting placement deferred to next window (BeachFX defer policy)",
        _defer_single,
        # Counterpart to 4_interrupt_single: rather than split SSN@20.5 → SSN@34.5,
        # the placement is deferred (recovery-only over gap 1), then placed atomically
        # in gap 2 — one SSN→ESN, no partial.
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_RECS, 34.0),  # placement deferred → recovery-only, cut short by the 2nd storm
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SSN, 34.5),
                _ESN,
                _END,
            ]
        },
        decisions=[_TRIG, _SDEFER, _TRIG],  # launch → storm-defer → resume in clear window
    ),
]


def _labels_times(expected: list):
    """Split a sequence of ``L`` or ``(L, t)`` entries into (labels, pinned times).

    An entry may pin an event's time; unpinned entries carry ``None`` and only
    the label is asserted.
    """
    labs, times = [], []
    for e in expected:
        lab, tm = e if isinstance(e, tuple) else (e, None)
        labs.append(lab)
        times.append(tm)
    return labs, times


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_event_sequence(case: Case, record_event):
    profiles, _sink = case.make()
    by_id = {p.id: p for p in profiles}
    durs_by_pid: dict[str, list] = {}
    for pid_, a, b, days in case.durations:
        durs_by_pid.setdefault(pid_, []).append((a, b, days))

    first_pid = next(iter(case.expected))
    for pid, expected in case.expected.items():
        exp_labels, exp_times = _labels_times(expected)
        p = by_id[pid]
        generated = labels(p)
        gen_times = [s.t for s in p.snapshots]
        record_event(
            case.id,
            case.desc,
            pid,
            expected,
            generated,
            gen_times,
            sim_start=SIM_START,
            durations=durs_by_pid.get(pid),
            decisions=case.decisions if pid == first_pid else None,
            reach_id=case.reach_id,
        )
        assert generated == exp_labels, f"{case.id}/{pid}"
        for i, want_t in enumerate(exp_times):
            if want_t is not None:
                assert gen_times[i] == pytest.approx(want_t), (
                    f"{case.id}/{pid} {exp_labels[i].value}@idx{i}: {gen_times[i]} != {want_t}"
                )

    # interval-in-days assertions
    for pid_, a, b, days in case.durations:
        ts = [s.t for s in by_id[pid_].snapshots]
        assert (ts[b] - ts[a]) == pytest.approx(days), (
            f"{case.id}/{pid_} interval [{a}→{b}]: {ts[b] - ts[a]} != {days}d"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_snapshot_times_non_decreasing(case: Case):
    """Generic invariant across all scenarios (INIT-first/EndIteration-last are
    already pinned by the exact-sequence assertions above)."""
    profiles, _ = case.make()
    for p in profiles:
        ts = [s.t for s in p.snapshots]
        assert ts == sorted(ts)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_reach_decisions(case: Case):
    """Reach-scope decision stream — asserted once per scenario (not per profile)."""
    _profiles, sink = case.make()
    assert sink.decision_kinds == case.decisions, f"{case.id}: {sink.decisions}"


# --- timing invariants not captured by the label sequence ------------------


def test_multi_nourish_crew_is_serial():
    """Single crew: the second profile starts only after the first completes."""
    (p0, p1), _ = _nourish_multi()
    assert times_of(p0, L.ESN)[0] <= times_of(p1, L.SSN)[0]


def test_blackout_start_after_window():
    (p0,), _ = _blackout()
    assert times_of(p0, L.SSN)[0] >= 50.0


def test_interrupt_completes_once_after_resume():
    (p0,), _ = _interrupt_single()
    esns = times_of(p0, L.ESN)
    assert len(esns) == 1
    assert esns[0] > times_of(p0, L.PreStorm)[1]  # completed after the 2nd storm


def test_interrupt_multi_queued_profile_last():
    """p1 waits for the crew and finishes after p0."""
    (p0, p1), _ = _interrupt_multi()
    assert times_of(p1, L.ESN)[0] >= times_of(p0, L.ESN)[0]


# --- INUNDATION: interim skip-and-reuse handling ---------------------------


def test_inundation_no_poststorm_and_marks_snapshot():
    (p0,), _ = _inundation()
    inun = [s for s in p0.snapshots if s.label == L.INUNDATION]
    assert L.PostStorm not in labels(p0)  # storm skipped, no CSHORE response
    assert len(inun) == 1
    assert inun[0].storm_response_type == StormResponseType.INUNDATION


def test_inundation_skips_phase3_no_recovery():
    """HACK (interim): no storm ⇒ no recovery/nourishment for an inundated profile."""
    (p0,), _ = _inundation()
    assert L.REC not in labels(p0)
    assert L.SSN not in labels(p0)


def test_inundation_isolated_per_profile():
    """One profile's failure does not stop a neighbour from responding/nourishing."""
    runner = MockCSStorm({"p0": "inundation", "p1": 1.0})  # p0 fails; p1 scoops a chunk
    p0, p1 = run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20]),
        runner=runner,
        cfg=_nourish_cfg(production_rate=100.0),
    )[0]
    assert L.INUNDATION in labels(p0) and L.PostStorm not in labels(p0)
    assert L.ESN in labels(p1)


def test_inundation_records_output_warning():
    import os
    import tempfile

    import pandas as pd

    from erosion.results import ParquetResultsSink

    with tempfile.TemporaryDirectory() as root:
        sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
        run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            runner=MockCSStorm("inundation"),
            cfg=_nourish_cfg(),
            sink=sink,
        )
        df = pd.read_csv(os.path.join(sink.out_dir, "warnings.csv"))
        assert len(df) == 1
        assert "INUNDATION" in df.iloc[0]["message"]
