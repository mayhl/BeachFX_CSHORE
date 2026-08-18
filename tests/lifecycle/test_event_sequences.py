"""Reach-loop event-generation tests.

These exercise ``Reach.run`` end-to-end and assert the *sequence* of events
(snapshot labels) each profile accumulates — i.e. that the interval loop emits
the right ordering of PreStorm / PostStorm / REC / SEN / EEN across a storm
schedule.  They test event handling, not the nourishment/recovery numerics.

CSHORE is the only double: ``ScriptedRunner`` applies morphology-stated damage
(``MINOR``/``SEVERE``/``MASSIVE`` berm cuts from ``tests/doubles.py``), and the
*real* ``VolumeAssessor`` turns the deficit into a campaign (bigger cut / slower
``production_rate`` → longer, interruptible placement).  Blackout scenarios use
``blackout_windows``.

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
from erosion.types import DecisionKind as D
from erosion.types import SnapshotLabel as L
from erosion.types import StormResponseType
from tests.builders import SIM_START, RecordingSink, ncfg, run, storms_at, template_profile
from tests.doubles import MASSIVE, MINOR, NONE, SEVERE, Inundation, ScriptedRunner

TRIGGER = 30.0  # deficit trigger: MINOR ~22 m³ (recover) vs SEVERE ~42 m³ (nourish)


# --- trace helpers ---------------------------------------------------------


def labels(p) -> list[L]:
    return [s.label for s in p.snapshots]


def times_of(p, label: L) -> list[float]:
    return [s.t for s in p.snapshots if s.label == label]


# Mask recovery to the sub-berm face.  Without it (z_berm=None) recovery blends EVERY
# node toward the shoreline-shifted pre-storm bed — translating the dune landward by
# the cut distance.  The uniform scoop's ~2 m shift hid that; a 10 m berm cut doesn't.
_STORM = {"z_berm": 2.0}


def _nourish_cfg(production_rate=500.0) -> ReachConfig:
    return ReachConfig(
        storm=_STORM,
        nourishment=ncfg(
            volume_trigger=TRIGGER, production_rate=production_rate, assessor="volume"
        ),
    )


def _emergency_cfg(production_rate=100.0) -> ReachConfig:
    # No regular volume gate; an emergency_volume threshold forces mobilization instead
    # (the ≥1-active-trigger validator accepts emergency_volume standing alone).
    nc = ncfg(
        volume_trigger=None,
        production_rate=production_rate,
        assessor="volume",
        emergency_volume=TRIGGER,
    )
    return ReachConfig(storm=_STORM, nourishment=nc)


# --- scenario factories → (profiles, RecordingSink) after one lifecycle -----
# Profiles start on their as-built restore geometry (deficit ≈ 0); ScriptedRunner
# applies a stated berm cut → the real assessor sizes the campaign. Sink captures
# reach decisions.

Made = tuple[list[Profile], RecordingSink]


def _recovery() -> Made:
    # nourishment configured, but MINOR → deficit ~22 < trigger 30 → NOURISH_SKIP + recover
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(MINOR),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _nourish_single() -> Made:
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(SEVERE),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _nourish_multi() -> Made:
    return run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(SEVERE),
        cfg=_nourish_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _interrupt_single() -> Made:
    # Storms 14 days apart; MASSIVE (~62 m³) at 4 m³/day is a ~15.5-day placement —
    # longer than the 13.5-day gap → interrupt.  The 2nd storm passes through (NONE);
    # the crew resumes on its own account, not because of fresh damage.
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20, 34]),
        sim_end=100.0,
        runner=ScriptedRunner({"p0": [MASSIVE, NONE]}),
        cfg=_nourish_cfg(production_rate=4.0),
        sink=RecordingSink(),
    )


def _defer_single() -> Made:
    # Same setup as _interrupt_single, but the BeachFX "defer" policy delays the
    # storm-hit placement to the next storm-free gap instead of splitting it.
    cfg = _nourish_cfg(production_rate=4.0)
    cfg.nourishment.storm_conflict = "defer"
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20, 34]),
        sim_end=100.0,
        runner=ScriptedRunner({"p0": [MASSIVE, NONE]}),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _interrupt_multi() -> Made:
    # 4 m³/day keeps both timings: p0's ~15.5-day placement spans past storm 2
    # (interrupt), AND the resumed crew (p0's ~2-day remainder first) reaches p1
    # before p1's 21-day recovery completes at 55.5 (RECN, not REC).
    return run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20, 34]),
        sim_end=140.0,
        runner=ScriptedRunner({"p0": [MASSIVE, NONE], "p1": [MASSIVE, NONE]}),
        cfg=_nourish_cfg(production_rate=4.0),
        sink=RecordingSink(),
    )


def _periodic() -> Made:
    cfg = ReachConfig(storm=_STORM, erosion=UniformErosionConfig(rate=0.01, tick_days=10.0))
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(MINOR),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _inundation() -> Made:
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(Inundation()),
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
        runner=ScriptedRunner(SEVERE),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _emergency_only() -> Made:
    # volume_trigger OFF; the same cut that nourishes in _nourish_single here
    # mobilizes via the emergency_volume force path → NOURISH_EMERGENCY, not TRIGGER.
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(SEVERE),
        cfg=_emergency_cfg(production_rate=100.0),
        sink=RecordingSink(),
    )


def _mobilization() -> Made:
    # 5-day crew lead-time: placement can't start until storm_end + mobilization_days,
    # so the profile recovers over the lead-time (REC) before SEN — unlike _nourish_single
    # where the crew arrives at storm_end and no recovery precedes the placement.
    cfg = _nourish_cfg(production_rate=100.0)
    cfg.nourishment.mobilization_days = 5.0
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(SEVERE),
        cfg=cfg,
        sink=RecordingSink(),
    )


def _blackout_block() -> Made:
    # A blackout spanning the whole first inter-storm gap pushes the placement start
    # past the next storm → the campaign can't start (BLOCKED), carries forward, and
    # re-triggers in the second gap once the blackout has lifted.  Distinct from
    # _blackout (single storm: the start defers *within* the same window).
    cfg = _nourish_cfg(production_rate=100.0)
    cfg.nourishment.blackout_windows = [(20.0, 38.0)]
    return run(
        [template_profile("p0")],
        storms_df=storms_at([20, 34]),
        sim_end=100.0,
        runner=ScriptedRunner(SEVERE),
        cfg=cfg,
        sink=RecordingSink(),
    )


# --- expected full sequences per scenario ----------------------------------

_I, _PRE, _POST = L.INIT, L.PreStorm, L.PostStorm
_REC, _RECS, _SEN, _EEN, _END = L.REC, L.RECS, L.SEN, L.EEN, L.EndIteration
_RECN = L.RECN
_EENS = L.EENS  # storm cut the placement mid-fill
_INUN = L.INUNDATION
_PER = L.Periodic

# reach-scope decision aliases
_TRIG, _SKIP, _DEFER, _INTR, _SDEFER, _EMER = (
    D.NOURISH_TRIGGER,
    D.NOURISH_SKIP,
    D.BLACKOUT_DEFER,
    D.INTERRUPT,
    D.STORM_DEFER,
    D.NOURISH_EMERGENCY,
)


@dataclass
class Case:
    id: str
    desc: str
    make: Callable[[], tuple[list[Profile], RecordingSink]]
    expected: dict[str, list]  # per profile: a sequence of L or (L, t) time-pinned entries
    # Interval assertions in days: (profile_id, from, to, days), where each endpoint is
    # an (label, nth-occurrence) pair — NOT a position in the sequence.  Naming the
    # endpoints keeps the interval anchored to the events it measures, so inserting a
    # label elsewhere in the sequence can't silently re-point it at the wrong pair.
    durations: list = field(default_factory=list)
    # reach-scope decision kinds, in emission order (asserted once per scenario)
    decisions: list = field(default_factory=list)
    reach_id: str = "test"  # matches builders.run() default; shown in the events table


CASES = [
    Case(
        "1_recovery",
        "sub-trigger deficit → NOURISH_SKIP + recovery",
        _recovery,
        # REC lands at storm_end + campaign_offset + T_recover = 20.501 + 21 (true
        # completion), not the far window end.
        {"p0": [_I, (_PRE, 20.0), (_POST, 20.5), (_REC, 41.501), _END]},
        decisions=[_SKIP],
    ),
    Case(
        "2_nourish_single",
        "chunk deficit → single nourishment",
        _nourish_single,
        {"p0": [_I, _PRE, (_POST, 20.5), (_SEN, 20.501), _EEN, _END]},
        decisions=[_TRIG],
    ),
    Case(
        "3_nourish_multi",
        "multi-profile nourishment, serial crew",
        _nourish_multi,
        {
            "p0": [_I, _PRE, (_POST, 20.5), (_SEN, 20.501), _EEN, _END],
            "p1": [_I, _PRE, (_POST, 20.5), _RECN, _SEN, _EEN, _END],
        },
        decisions=[_TRIG],  # one reach-level trigger; crew serves both profiles
    ),
    Case(
        "4_interrupt_single",
        "nourishment interrupted 14 days in, then resumes",
        _interrupt_single,
        # SEN starts at the storm (t=20); the 2nd storm interrupts 14 days later
        # (t=34); the campaign resumes there (SEN t=34). Placement length is emergent.
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_SEN, 20.501),
                (_EENS, 34.0),  # 2nd storm cuts the placement short (partial fill)
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SEN, 34.501),
                _EEN,
                _END,
            ]
        },
        # the crew starts, is cut off, and resumes 14 days later at the 2nd storm
        durations=[("p0", (_SEN, 0), (_SEN, 1), 14.0)],
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
                (_SEN, 20.501),
                (_EENS, 34.0),  # 2nd storm cuts the placement short (partial fill)
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SEN, 34.501),
                _EEN,
                _END,
            ],
            "p1": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_RECS, 34.0),  # 2nd storm cuts recovery short (13.5d < T_recover 21d)
                _PRE,
                (_POST, 34.5),
                _RECN,  # recovery up to this profile's own nourishment start (crew, not storm)
                _SEN,
                _EEN,
                _END,
            ],
        },
        durations=[("p0", (_SEN, 0), (_SEN, 1), 14.0)],
        decisions=[_TRIG, _INTR, _TRIG],
    ),
    Case(
        "7_blackout",
        "nourishment deferred past a blackout window",
        _blackout,
        # recovery completes (41.501) before the blackout-deferred crew arrives (50),
        # so it's a full REC at completion, then SEN waits for the window.
        {"p0": [_I, _PRE, (_POST, 20.5), (_REC, 41.501), (_SEN, 50.0), _EEN, _END]},
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
        # Counterpart to 4_interrupt_single: rather than split SEN@20.5 → SEN@34.5,
        # the placement is deferred (recovery-only over gap 1), then placed atomically
        # in gap 2 — one SEN→EEN, no partial.
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_RECS, 34.0),  # placement deferred → recovery-only, cut short by the 2nd storm
                (_PRE, 34.0),
                (_POST, 34.5),
                (_SEN, 34.501),
                _EEN,
                _END,
            ]
        },
        decisions=[_TRIG, _SDEFER, _TRIG],  # launch → storm-defer → resume in clear window
    ),
    Case(
        "11_emergency_only",
        "no volume gate → emergency_volume force mobilizes the reach",
        _emergency_only,
        # Same sequence as 2_nourish_single, but reached through the force path: the
        # reach decision is NOURISH_EMERGENCY, not NOURISH_TRIGGER.
        {"p0": [_I, _PRE, (_POST, 20.5), (_SEN, 20.501), _EEN, _END]},
        decisions=[_EMER],
    ),
    Case(
        "12_mobilization",
        "crew lead-time delays SEN and recovers over the gap first",
        _mobilization,
        # mobilization_days=5 → crew arrives at storm_end(20.5)+5=25.5; the profile
        # recovers up to the crew's arrival (REC@25.5) before the placement starts.
        {"p0": [_I, _PRE, (_POST, 20.5), (_RECN, 25.501), (_SEN, 25.501), _EEN, _END]},
        # PostStorm → SEN = 5-day mobilization + the 0.001-day campaign-start offset
        durations=[("p0", (_POST, 0), (_SEN, 0), 5.001)],
        decisions=[_TRIG],
    ),
    Case(
        "13_blackout_block",
        "blackout spanning a gap blocks the campaign → re-triggers next gap",
        _blackout_block,
        # Gap 1: blackout (20,38) pushes the start past storm 2 (t=34) → BLOCKED, no
        # SEN, recovery cut short (RECS@34).  Gap 2: re-triggers, defers to the
        # blackout end (38), recovers up to it, then places.
        {
            "p0": [
                _I,
                _PRE,
                (_POST, 20.5),
                (_RECS, 34.0),
                (_PRE, 34.0),
                (_POST, 34.5),
                (_RECN, 38.0),
                (_SEN, 38.0),
                _EEN,
                _END,
            ]
        },
        decisions=[_TRIG, _DEFER, _TRIG, _DEFER],  # blocked launch → re-launch, both defer
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


def _nth_time(p, label: L, n: int) -> float:
    """Time of the ``n``th (0-based) occurrence of ``label`` on this profile."""
    return times_of(p, label)[n]


_MADE: dict[str, Made] = {}


def made(make: Callable[[], Made]) -> Made:
    """Run a scenario once and reuse it.

    Each scenario is a full lifecycle, and three parametrized tests plus the timing
    tests below all want the same one — re-running it per test bought nothing but
    wall-clock.  The tests only read the result, so one run serves them all.
    """
    key = make.__name__
    if key not in _MADE:
        _MADE[key] = make()
    return _MADE[key]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_event_sequence(case: Case, record_event):
    profiles, _sink = made(case.make)
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

    # interval-in-days assertions, endpoints named by (label, nth occurrence)
    for pid_, a, b, days in case.durations:
        p = by_id[pid_]
        elapsed = _nth_time(p, *b) - _nth_time(p, *a)
        assert elapsed == pytest.approx(days), (
            f"{case.id}/{pid_} interval [{a[0].value}#{a[1]}→{b[0].value}#{b[1]}]: "
            f"{elapsed} != {days}d"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_snapshot_times_non_decreasing(case: Case):
    """Generic invariant across all scenarios (INIT-first/EndIteration-last are
    already pinned by the exact-sequence assertions above)."""
    profiles, _ = made(case.make)
    for p in profiles:
        ts = [s.t for s in p.snapshots]
        assert ts == sorted(ts)


# --- timing invariants not captured by the label sequence ------------------


def test_multi_nourish_crew_is_serial():
    """Single crew: the second profile starts only after the first completes."""
    (p0, p1), _ = made(_nourish_multi)
    assert times_of(p0, L.EEN)[0] <= times_of(p1, L.SEN)[0]


def test_blackout_start_after_window():
    (p0,), _ = made(_blackout)
    assert times_of(p0, L.SEN)[0] >= 50.0


def test_interrupt_completes_once_after_resume():
    (p0,), _ = made(_interrupt_single)
    completions = times_of(p0, L.EEN)
    assert len(completions) == 1  # the partial closes as EENS; only the resume completes
    assert completions[0] > times_of(p0, L.PreStorm)[1]  # completed after the 2nd storm


def test_interrupt_multi_queued_profile_last():
    """p1 waits for the crew and finishes after p0."""
    (p0, p1), _ = made(_interrupt_multi)
    assert times_of(p1, L.EEN)[0] >= times_of(p0, L.EEN)[0]


# --- INUNDATION: interim skip-and-reuse handling ---------------------------


def test_inundation_no_poststorm_and_marks_snapshot():
    (p0,), _ = made(_inundation)
    inun = [s for s in p0.snapshots if s.label == L.INUNDATION]
    assert L.PostStorm not in labels(p0)  # storm skipped, no CSHORE response
    assert len(inun) == 1
    assert inun[0].storm_response_type == StormResponseType.INUNDATION


def test_inundation_skips_phase3_no_recovery():
    """HACK (interim): no storm ⇒ no recovery/nourishment for an inundated profile."""
    (p0,), _ = made(_inundation)
    assert L.REC not in labels(p0)
    assert L.SEN not in labels(p0)


def test_inundation_isolated_per_profile():
    """One profile's failure does not stop a neighbour from responding/nourishing."""
    runner = ScriptedRunner({"p0": Inundation(), "p1": SEVERE})  # p0 fails; p1 takes a cut
    p0, p1 = run(
        [template_profile("p0"), template_profile("p1")],
        storms_df=storms_at([20]),
        runner=runner,
        cfg=_nourish_cfg(production_rate=100.0),
    )[0]
    assert L.INUNDATION in labels(p0) and L.PostStorm not in labels(p0)
    assert L.EEN in labels(p1)


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
            runner=ScriptedRunner(Inundation()),
            cfg=_nourish_cfg(),
            sink=sink,
        )
        df = pd.read_csv(os.path.join(sink.out_dir, "warnings.csv"))
        assert len(df) == 1
        assert "INUNDATION" in df.iloc[0]["message"]
