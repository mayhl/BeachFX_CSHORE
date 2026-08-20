"""The periodic (planned) nourishment cycle — calendar-driven renourishment.

Covers the three seams the cycle adds: the cycle dates it generates from the
interval, the tracker that decides which cycle the reach owes as the interval loop
walks the gaps, and the campaign itself — which labels its placements ``SSN``/``ESN``
(a planned cycle) rather than the post-storm ``SEN``/``EEN``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from erosion.config import ReachConfig
from erosion.decision import CycleTracker, cycle_times
from erosion.interstorm import UniformErosionConfig
from erosion.nourishment import NourishmentConfig
from erosion.types import DecisionKind as D
from erosion.types import SnapshotLabel as L
from tests.builders import SIM_START, RecordingSink, ncfg, run, storms_at, template_profile
from tests.doubles import MINOR, ScriptedRunner

YEAR = 365.0


def _cycle_cfg(
    interval_years: float,
    *,
    start_day: float | None = None,
    volume_trigger: float = 30.0,
    production_rate: float = 500.0,
    erosion_rate: float = 0.0,
) -> ReachConfig:
    """A reach on a planned cycle, optionally with background erosion accruing deficit."""
    nc = ncfg(
        volume_trigger=volume_trigger,
        production_rate=production_rate,
        assessor="volume",
        cycle_interval_years=interval_years,
        cycle_start_date=None if start_day is None else SIM_START + timedelta(days=start_day),
    )
    erosion = UniformErosionConfig(rate=erosion_rate, tick_days=10.0) if erosion_rate else None
    # z_berm masks recovery to the sub-berm face (see test_event_sequences._STORM)
    return ReachConfig(storm={"z_berm": 2.0}, nourishment=nc, erosion=erosion)


# --- cycle dates -----------------------------------------------------------


class TestCycleTimes:
    def test_none_interval_yields_no_cycles(self):
        cfg = ReachConfig()
        assert cycle_times(cfg.nourishment, SIM_START, 5 * YEAR) == []

    def test_defaults_to_the_sim_start_then_steps_by_the_interval(self):
        ncfg = _cycle_cfg(2.0).nourishment
        assert cycle_times(ncfg, SIM_START, 5 * YEAR) == [0.0, 2 * YEAR, 4 * YEAR]

    def test_explicit_start_date_anchors_the_first_cycle(self):
        ncfg = _cycle_cfg(1.0, start_day=100.0).nourishment
        assert cycle_times(ncfg, SIM_START, 2 * YEAR) == [100.0, 100.0 + YEAR]

    def test_stops_at_the_sim_end(self):
        ncfg = _cycle_cfg(1.0).nourishment
        assert cycle_times(ncfg, SIM_START, 1.5 * YEAR) == [0.0, YEAR]

    def test_a_start_date_before_the_sim_drops_the_cycles_outside_the_window(self):
        ncfg = _cycle_cfg(1.0, start_day=-200.0).nourishment
        # -200, +165, +530 … — only the in-window ones survive.
        assert cycle_times(ncfg, SIM_START, 2 * YEAR) == [165.0, 165.0 + YEAR]


# --- which cycle is owed ---------------------------------------------------


class TestCycleTracker:
    def test_owes_nothing_before_the_first_cycle_date(self):
        t = CycleTracker.build(_cycle_cfg(1.0, start_day=100.0).nourishment, SIM_START, 2 * YEAR)
        assert t.take_due(50.0) is None

    def test_hands_out_the_cycle_once_it_comes_due(self):
        t = CycleTracker.build(_cycle_cfg(1.0, start_day=100.0).nourishment, SIM_START, 2 * YEAR)
        assert t.take_due(150.0) == 100.0

    def test_an_unrun_cycle_stays_owed(self):
        """The gap closed without running it — it comes back in the next gap."""
        t = CycleTracker.build(_cycle_cfg(1.0, start_day=100.0).nourishment, SIM_START, 2 * YEAR)
        assert t.take_due(150.0) == 100.0
        assert t.take_due(150.0) == 100.0  # still owed; nothing cleared it

    def test_an_owed_cycle_blocks_the_ones_behind_it(self):
        t = CycleTracker.build(_cycle_cfg(1.0, start_day=100.0).nourishment, SIM_START, 3 * YEAR)
        assert t.take_due(3 * YEAR) == 100.0  # cycle 2 is also due, but waits its turn
        t.clear()
        assert t.take_due(3 * YEAR) == 100.0 + YEAR

    def test_clearing_the_last_cycle_leaves_nothing_owed(self):
        t = CycleTracker.build(_cycle_cfg(1.0, start_day=100.0).nourishment, SIM_START, 200.0)
        assert t.take_due(200.0) == 100.0  # the only cycle in the window
        t.clear()
        assert t.take_due(200.0) is None


# --- config ----------------------------------------------------------------


class TestCycleConfig:
    def test_a_cycle_without_a_volume_gate_is_rejected(self):
        """The calendar proposes and the volume gate disposes — without the gate the
        cycle would place fill every interval regardless of the beach's state."""
        with pytest.raises(ValidationError, match="cycle_interval_years requires volume_trigger"):
            NourishmentConfig.model_validate(
                {
                    "production_rate": {"value": 500.0, "units": "m3/day"},
                    "emergency_volume": {"value": 50.0, "units": "m3"},
                    "cycle_interval_years": 1.0,
                },
                context={"input_units": "m"},
            )

    def test_a_non_positive_interval_is_rejected(self):
        with pytest.raises(ValidationError, match="cycle_interval_years must be positive"):
            NourishmentConfig.model_validate(
                {
                    "volume_trigger": {"value": 30.0, "units": "m3"},
                    "production_rate": {"value": 500.0, "units": "m3/day"},
                    "cycle_interval_years": 0.0,
                },
                context={"input_units": "m"},
            )


# --- the campaign, end to end ----------------------------------------------


class TestPlannedCycleRun:
    def test_a_due_cycle_on_an_eroded_beach_places_ssn_esn(self):
        """Background erosion re-accrues a deficit after the storm crew leaves; the cycle
        date arrives and the calendar places its own fill.

        Both campaigns run here, which is the point: the storm response is labelled
        ``SEN``/``EEN`` and the planned cycle ``SSN``/``ESN``, so one run tells them apart.
        """
        cfg = _cycle_cfg(1.0, start_day=200.0, erosion_rate=0.02)
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            sim_end=300.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        seq = [s.label for s in profiles[0].snapshots]
        assert [s for s in seq if s in (L.SEN, L.EEN)] == [L.SEN, L.EEN]  # the storm's
        assert [s for s in seq if s in (L.SSN, L.ESN)] == [L.SSN, L.ESN]  # the calendar's

        kinds = [k for k, *_ in sink.decisions]
        assert D.NOURISH_TRIGGER in kinds  # post-storm
        assert D.NOURISH_CYCLE in kinds  # planned

    def test_the_cycle_fires_at_its_date_not_the_storm(self):
        cfg = _cycle_cfg(1.0, start_day=200.0, erosion_rate=0.02)
        profiles, _ = run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            sim_end=300.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        ssn = [s.t for s in profiles[0].snapshots if s.label == L.SSN]
        assert ssn == pytest.approx([200.0], abs=1e-6)

    def test_a_healthy_beach_skips_its_cycle(self):
        """Calendar proposes, volume gate disposes: no erosion, no storm damage left to
        fix, so the cycle date passes without placing anything."""
        cfg = _cycle_cfg(1.0, start_day=200.0)  # no background erosion
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            sim_end=300.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        seq = [s.label for s in profiles[0].snapshots]
        assert L.SSN not in seq
        kinds = [k for k, *_ in sink.decisions]
        assert D.NOURISH_CYCLE not in kinds

    def test_a_cycle_landing_mid_recovery_defers_past_it(self):
        """BeachFX's deferral code 1: the crew waits out the recovery rather than cutting
        it short, so the cycle is decided at the recovery completion, not on its own date
        — and no RECN is stamped.

        It finds a beach the storm crew restored but the recovery window's held ticks
        just landed on (the catch-up tick precedes the cycle assess), so it clears the
        volume gate and places; the deferral is what's asserted, and the decision
        carries the time it fired.
        """
        cfg = _cycle_cfg(1.0, start_day=25.0, erosion_rate=0.02)
        recovery_done = 20.5 + 0.001 + cfg.storm.T_recover  # storm end + offset + T_recover
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            sim_end=300.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        cycle_decisions = [
            (k, t) for k, t, _pid, p in sink.decisions if k is D.NOURISH_CYCLE or p.get("cycle")
        ]
        assert len(cycle_decisions) == 1
        _kind, t_fired = cycle_decisions[0]
        assert t_fired == pytest.approx(recovery_done, abs=1e-6)
        assert t_fired > 25.0  # pushed past its own cycle date

        seq = [s.label for s in profiles[0].snapshots]
        assert L.SEN in seq  # the storm crew was already on the beach
        assert L.RECN not in seq  # and the planned cycle cut no recovery short

    def test_a_blackout_window_delays_the_cycles_placement(self):
        """A planned cycle books the same crew as a storm response, so it defers around a
        blackout window the same way: the placement starts when the window lifts."""
        cfg = _cycle_cfg(1.0, start_day=200.0, erosion_rate=0.02, production_rate=500.0)
        cfg.nourishment.blackout_windows = [(190.0, 230.0)]
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([20]),
            sim_end=400.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        ssn = [s.t for s in profiles[0].snapshots if s.label == L.SSN]
        assert ssn and ssn[0] == pytest.approx(230.0, abs=1e-6)  # not its 200.0 date

        deferrals = [p for k, _t, _pid, p in sink.decisions if k is D.BLACKOUT_DEFER]
        assert deferrals and deferrals[0]["deferred_to"] == pytest.approx(230.0, abs=1e-6)

    def test_a_storm_cutting_a_cycle_mid_placement_stamps_esns(self):
        """A slow crew still placing when the next storm lands: the planned cycle's
        segment closes as ESNS (partial fill), not ESN — the storm-cut counterpart of the
        post-storm campaign's EENS."""
        cfg = _cycle_cfg(1.0, start_day=200.0, erosion_rate=0.02, production_rate=5.0)
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([20, 210]),
            sim_end=400.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        seq = [s.label for s in profiles[0].snapshots]
        assert L.SSN in seq
        assert L.ESNS in seq  # cut by the day-210 storm
        assert L.EENS not in seq  # and it was the CYCLE that was cut, not a storm campaign

        kinds = [k for k, *_ in sink.decisions]
        assert D.INTERRUPT in kinds

    def test_a_stormless_lifecycle_still_runs_its_cycle(self):
        """The interval loop is storm-driven; a reach with no storms in the window must
        still erode and still nourish on its calendar."""
        cfg = _cycle_cfg(1.0, start_day=200.0, erosion_rate=0.02)
        profiles, sink = run(
            [template_profile("p0")],
            storms_df=storms_at([], sim_start=datetime(2030, 1, 1)),
            sim_end=300.0,
            runner=ScriptedRunner(MINOR),
            cfg=cfg,
            sink=RecordingSink(),
        )
        seq = [s.label for s in profiles[0].snapshots]
        assert L.PostStorm not in seq  # no storms at all
        assert L.SSN in seq and L.ESN in seq
