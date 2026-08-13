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
assessor), and a placement runs ``deficit / production_rate`` days.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import pytest

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.nourishment import NourishmentConfig
from erosion.results import ParquetResultsSink
from tests.builders import SIM_START, ncfg, run, storms_at, template_profile
from tests.doubles import MASSIVE, NONE, SEVERE, ScriptedRunner

_STORM = {"z_berm": 2.0}  # mask recovery to the sub-berm face (see test_event_sequences)


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
    durations: dict = field(default_factory=dict)


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
        decisions=[("NOURISH_CYCLE", 200.0, None, {"deficit": 226.73, "resume": 0, "forced": 0})],
        decision_columns=_BASE_COLS | {"deficit", "resume", "forced"},
        labels=["INIT"] + ["Periodic"] * 20 + ["SSN", "ESN"] + ["Periodic"] * 10 + ["EndIteration"],
        events=[("ErosionTick", 10.0 * k) for k in range(1, 21)]
        + [("NourishmentStart", 200.0), ("FullNourishment", None)]
        + [("ErosionTick", 10.0 * k) for k in range(21, 31)],
        segments=[("FullNourishment", 200.0, None, None)],
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
            ("NOURISH_TRIGGER", 34.501, None, {"deficit": 91.0, "resume": 0, "forced": 0}),
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
            ("FullNourishment", None),
        ],
        segments=[("FullNourishment", 38.0, None, None)],
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
    assert len(df) == len(sc.decisions), sc.id
    for i, (kind, t, pid, payload) in enumerate(sc.decisions):
        row = df.iloc[i]
        assert row["kind"] == kind, f"{sc.id} seq {i}"
        assert row["t"] == pytest.approx(t, abs=1e-3), f"{sc.id} seq {i}"
        assert (row["profile_id"] == pid) if pid is not None else pd.isna(row["profile_id"])
        for key, want in payload.items():
            assert row[key] == pytest.approx(want, abs=0.05), f"{sc.id} seq {i} {key}"
        # every payload column this row does not claim must be null — absent keys
        # landing as values would mean two decision kinds bled into each other
        for key in sc.decision_columns - _BASE_COLS - set(payload):
            assert pd.isna(row[key]), f"{sc.id} seq {i} stray {key}"


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_snapshots_label_stream(sc, outputs):
    df = pd.read_parquet(os.path.join(outputs[sc.id], "snapshots.parquet"))
    assert list(df.sort_values("t", kind="stable")["label"]) == sc.labels, sc.id


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_events_parquet_applied_stream(sc, outputs):
    df = pd.read_parquet(os.path.join(outputs[sc.id], "events.parquet")).sort_values("event_seq")
    assert [r.event_type for r in df.itertuples()] == [e for e, _ in sc.events], sc.id
    for (want_type, want_t), got_t in zip(sc.events, df["t"]):
        if want_t is not None:
            assert got_t == pytest.approx(want_t, abs=1e-3), f"{sc.id} {want_type}"


@pytest.mark.parametrize("sc", SCENARIOS, ids=_ids)
def test_placements_nourishment_rows(sc, outputs):
    df = pd.read_csv(os.path.join(outputs[sc.id], "placements.csv"))
    assert len(df) == len(sc.segments), sc.id
    for i, (etype, t0, t1, vol) in enumerate(sc.segments):
        row = df.iloc[i]
        assert row["event_type"] == etype, f"{sc.id} row {i}"
        if t0 is not None:
            assert row["t_start"] == pytest.approx(t0, abs=1e-3)
        if t1 is not None:
            assert row["t_end"] == pytest.approx(t1, abs=1e-3)
        if vol is not None:
            assert row["placed_m3"] == pytest.approx(vol, abs=0.05)
