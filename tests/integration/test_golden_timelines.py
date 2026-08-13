"""Golden decision/event timelines, read back from the output parquet.

The event-sequence suite asserts in-memory snapshot labels; these tests pin the
*persisted* record instead — the full ordered contents of ``decisions.parquet``, the
label stream in ``snapshots.parquet``, the applied-event stream in
``events.parquet``, and the nourishment rows of ``placements.csv`` — for four
canonical lifecycles.  They are the schema-and-ordering contract the decide/execute
split runs against: emission order (``decision_seq``) is the only ordering that holds
by design, so these are the tests that catch a refactor silently reordering it.

Every pinned number is arithmetic on the scenario's stated damage: a ``BermCut(c)``
deficit is ``c x 2 m`` berm elevation (+2 m3 of foreshore ramp on the volume
assessor), and a placement runs ``deficit / production_rate`` days.  Two deficits
(``_CYCLE_DEFICIT``, ``_BLACKOUT_DEFICIT``) are pinned from the run instead — see
the note at their declaration.

Authoring a new scenario: the label vocabulary is ``erosion.types.SnapshotLabel``
(docstrings there explain each), the decision kinds are ``erosion.types.DecisionKind``
with payload keys mirrored in ``erosion.decision.model``'s ``row()`` methods, and
``tests/lifecycle/test_nourishment_cycle.py`` is the prior art for cycle labels
(SSN/ESN vs the storm-triggered SEN/EEN).  Damage forms live in ``tests/doubles.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import pytest

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.nourishment import NourishmentConfig
from erosion.results import ParquetResultsSink
from tests.builders import SIM_START, ncfg, run, storms_at, template_profile
from tests.doubles import MASSIVE, NONE, SEVERE, ScriptedRunner

_STORM = {"z_berm": 2.0}  # mask recovery to the sub-berm face (see test_event_sequences)

# Pinned from the run, not head-arithmetic: an accrued-erosion deficit measured
# against the parametric restore template (and the recovery/re-cut interplay in
# blackout_block) has no clean closed form.  Regenerate by rerunning the scenario.
# The placement end-times ARE arithmetic on these: t_start + deficit/production.
_CYCLE_DEFICIT = 226.73  # stormless_cycle: 20 erosion ticks assessed at the day-200 cycle
_BLACKOUT_DEFICIT = 91.0  # blackout_block: doubled SEVERE damage less the cut-short recovery


def _nourish_cfg(production_rate: float) -> ReachConfig:
    return ReachConfig(
        storm=_STORM,
        nourishment=ncfg(volume_trigger=30.0, production_rate=production_rate, assessor="volume"),
    )


def _cycle_cfg() -> ReachConfig:
    from datetime import timedelta

    nc = NourishmentConfig.model_validate(
        {
            "volume_trigger": {"value": 30.0, "units": "m3"},
            "production_rate": {"value": 500.0, "units": "m3/day"},
            "assessor": "volume",
            "cycle_interval_years": 1.0,
            "cycle_start_date": SIM_START + timedelta(days=200.0),
        },
        context={"input_units": "m"},
    )
    return ReachConfig(
        storm=_STORM, nourishment=nc, erosion=UniformErosionConfig(rate=0.02, tick_days=10.0)
    )


def _blackout_cfg() -> ReachConfig:
    cfg = _nourish_cfg(production_rate=100.0)
    cfg.nourishment.blackout_windows = [(20.0, 38.0)]
    return cfg


@dataclass(frozen=True)
class Scenario:
    id: str
    runner_script: object
    cfg_make: object
    storm_days: list
    sim_end: float | None
    # decisions.parquet, in emission order: (kind, t, profile_id, {payload: value}).
    # Payload keys absent from a row must be null there; the column set is pinned too.
    decisions: list
    decision_columns: set
    labels: list  # snapshots.parquet label stream, time-ordered
    events: list  # events.parquet (event_type, t) stream, in event_seq order
    segments: list  # placements.csv (event_type, t_start, t_end, placed_m3)


_BASE_COLS = {"decision_seq", "kind", "t", "profile_id"}

SCENARIOS = [
    Scenario(
        # SEVERE (~42 m3) at 100 m3/day: trigger fires at the campaign start and the
        # fill takes 0.42 days.
        id="single_nourish",
        runner_script=SEVERE,
        cfg_make=lambda: _nourish_cfg(production_rate=100.0),
        storm_days=[20],
        sim_end=None,
        decisions=[("NOURISH_TRIGGER", 20.501, None, {"deficit": 42.0, "resume": 0, "forced": 0})],
        decision_columns=_BASE_COLS | {"deficit", "resume", "forced"},
        labels=["INIT", "PreStorm", "PostStorm", "SEN", "EEN", "EndIteration"],
        events=[
            ("StormResponse", 20.5),
            ("NourishmentStart", 20.501),
            ("FullNourishment", 20.921),
        ],
        segments=[("FullNourishment", 20.501, 20.921, 42.0)],
    ),
    Scenario(
        # MASSIVE (~62 m3) at 4 m3/day is a 15.5-day fill against a 13.5-day gap: the
        # 2nd storm interrupts at fraction 13.499/15.5, the crew resumes (gate
        # bypassed, resume=True) and places the ~10.6 m3 remainder in 2.65 days.
        id="interrupt_resume",
        runner_script={"p0": [MASSIVE, NONE]},
        cfg_make=lambda: _nourish_cfg(production_rate=4.0),
        storm_days=[20, 34],
        sim_end=100.0,
        decisions=[
            ("NOURISH_TRIGGER", 20.501, None, {"deficit": 62.0, "resume": 0, "forced": 0}),
            ("INTERRUPT", 34.0, "p0", {"placed_fraction": 13.499 / 15.5}),
            ("NOURISH_TRIGGER", 34.501, None, {"deficit": 10.61, "resume": 1, "forced": 0}),
        ],
        decision_columns=_BASE_COLS | {"deficit", "resume", "forced", "placed_fraction"},
        labels=[
            "INIT",
            "PreStorm",
            "PostStorm",
            "SEN",
            "EENS",
            "PreStorm",
            "PostStorm",
            "SEN",
            "EEN",
            "EndIteration",
        ],
        events=[
            ("StormResponse", 20.5),
            ("NourishmentStart", 20.501),
            ("PartialNourishment", 34.0),
            ("StormResponse", 34.5),
            ("NourishmentStart", 34.501),
            ("FullNourishment", 37.153),
        ],
        segments=[
            ("PartialNourishment", 20.501, 34.0, 54.0),
            ("FullNourishment", 34.501, 37.153, 10.61),
        ],
    ),
    Scenario(
        # No storms at all: the interval loop still ticks erosion every 10 days and the
        # calendar still fires its day-200 cycle on the accrued deficit (NOURISH_CYCLE
        # carries cycle=True on this path — asserted via the kind).
        id="stormless_cycle",
        runner_script=NONE,
        cfg_make=_cycle_cfg,
        storm_days=[],
        sim_end=300.0,
        decisions=[
            ("NOURISH_CYCLE", 200.0, None, {"deficit": _CYCLE_DEFICIT, "resume": 0, "forced": 0})
        ],
        decision_columns=_BASE_COLS | {"deficit", "resume", "forced"},
        labels=["INIT"] + ["Periodic"] * 20 + ["SSN", "ESN"] + ["Periodic"] * 10 + ["EndIteration"],
        events=[("ErosionTick", 10.0 * k) for k in range(1, 21)]
        + [("NourishmentStart", 200.0), ("FullNourishment", 200.0 + _CYCLE_DEFICIT / 500.0)]
        + [("ErosionTick", 10.0 * k) for k in range(21, 31)],
        segments=[("FullNourishment", 200.0, 200.0 + _CYCLE_DEFICIT / 500.0, _CYCLE_DEFICIT)],
    ),
    Scenario(
        # A blackout spanning the whole first gap pushes the start past storm 2: the
        # campaign is blocked (deferral emitted, nothing placed, recovery cut short),
        # re-triggers in gap 2 on the doubled damage, defers to the window end, places.
        id="blackout_block",
        runner_script=SEVERE,
        cfg_make=_blackout_cfg,
        storm_days=[20, 34],
        sim_end=100.0,
        decisions=[
            ("NOURISH_TRIGGER", 20.501, None, {"deficit": 42.0, "resume": 0, "forced": 0}),
            ("BLACKOUT_DEFER", 20.501, "p0", {"requested": 20.501, "deferred_to": 38.0}),
            (
                "NOURISH_TRIGGER",
                34.501,
                None,
                {"deficit": _BLACKOUT_DEFICIT, "resume": 0, "forced": 0},
            ),
            ("BLACKOUT_DEFER", 34.501, "p0", {"requested": 34.501, "deferred_to": 38.0}),
        ],
        decision_columns=_BASE_COLS | {"deficit", "resume", "forced", "requested", "deferred_to"},
        labels=[
            "INIT",
            "PreStorm",
            "PostStorm",
            "RECS",
            "PreStorm",
            "PostStorm",
            "RECN",
            "SEN",
            "EEN",
            "EndIteration",
        ],
        events=[
            ("StormResponse", 20.5),
            ("Recovery", 34.0),
            ("StormResponse", 34.5),
            ("Recovery", 38.0),
            ("NourishmentStart", 38.0),
            ("FullNourishment", 38.0 + _BLACKOUT_DEFICIT / 100.0),
        ],
        segments=[("FullNourishment", 38.0, 38.0 + _BLACKOUT_DEFICIT / 100.0, _BLACKOUT_DEFICIT)],
    ),
]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> dict[str, str]:
    """Run each scenario once into its own sink; every test reads the same output."""
    dirs = {}
    for sc in SCENARIOS:
        root = tmp_path_factory.mktemp(sc.id)
        sink = ParquetResultsSink(str(root), "R", "FWOP", lifecycle=0)
        run(
            [template_profile("p0")],
            storms_df=storms_at(sc.storm_days),
            sim_end=sc.sim_end,
            runner=ScriptedRunner(sc.runner_script),
            cfg=sc.cfg_make(),
            sink=sink,
        )
        dirs[sc.id] = sink.out_dir
    return dirs


def _ids(s):
    return s.id


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_decisions_parquet_full_ordered_contents(sc, outputs):
    df = pd.read_parquet(os.path.join(outputs[sc.id], "decisions.parquet"))
    assert set(df.columns) == sc.decision_columns, sc.id
    assert list(df["decision_seq"]) == list(range(len(sc.decisions))), sc.id
    # One whole-stream assert per column: a failure diffs the full ordered stream
    # instead of stopping at the first divergent cell
    assert list(df["kind"]) == [k for k, _, _, _ in sc.decisions], sc.id
    assert list(df["t"]) == pytest.approx([t for _, t, _, _ in sc.decisions], abs=1e-3), sc.id
    assert [None if pd.isna(v) else v for v in df["profile_id"]] == [
        pid for _, _, pid, _ in sc.decisions
    ], sc.id
    # Payload columns: a key a row does not claim must be null there — absent keys
    # landing as values would mean two decision kinds bled into each other
    for key in sorted(sc.decision_columns - _BASE_COLS):
        want = [payload.get(key, float("nan")) for _, _, _, payload in sc.decisions]
        got = list(pd.to_numeric(df[key]))  # nullable bools -> floats, None -> NaN
        assert got == pytest.approx(want, abs=0.05, nan_ok=True), f"{sc.id} {key}"


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_snapshots_label_stream(sc, outputs):
    df = pd.read_parquet(os.path.join(outputs[sc.id], "snapshots.parquet"))
    assert list(df.sort_values("t", kind="stable")["label"]) == sc.labels, sc.id


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_events_parquet_applied_stream(sc, outputs):
    df = pd.read_parquet(os.path.join(outputs[sc.id], "events.parquet")).sort_values("event_seq")
    assert list(df["event_type"]) == [e for e, _ in sc.events], sc.id
    assert list(df["t"]) == pytest.approx([t for _, t in sc.events], abs=1e-3), sc.id


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_placements_nourishment_rows(sc, outputs):
    df = pd.read_csv(os.path.join(outputs[sc.id], "placements.csv"))
    assert list(df["event_type"]) == [s[0] for s in sc.segments], sc.id
    assert list(df["t_start"]) == pytest.approx([s[1] for s in sc.segments], abs=1e-3), sc.id
    assert list(df["t_end"]) == pytest.approx([s[2] for s in sc.segments], abs=1e-3), sc.id
    assert list(df["placed_m3"]) == pytest.approx([s[3] for s in sc.segments], abs=0.05), sc.id
