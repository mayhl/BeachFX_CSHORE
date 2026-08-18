"""The interval planner in isolation: scalar metrics in, decision sequences out.

No beds, no mocks, no runner — every scenario here is a handful of floats and a
config, which is the point of the decide/execute split: the policy that used to
be reachable only by driving a full lifecycle is now a function call.  The
integration goldens pin the wiring; these pin the policy, including the float
boundaries (``>=`` vs ``>``) that a refactor could silently flip.
"""

from __future__ import annotations

from collections import deque

import pytest

from erosion.decision import (
    CalendarState,
    CampaignCarryover,
    CycleTracker,
    DeferBlackout,
    DeferCycle,
    DeferStorm,
    FireCycle,
    Interrupt,
    LaunchCampaign,
    Placement,
    ProfileDemand,
    SkipCampaign,
    decide_campaign,
    erosion_split,
    plan_next_cycle,
    plan_placements,
)
from erosion.decision.planner import _next_available
from erosion.types import CampaignKind, DecisionKind
from tests.builders import ncfg

STORM, CYCLE = CampaignKind.STORM, CampaignKind.PLANNED


def _m(pid: str, deficit: float, placement: float | None = None, force: bool = False):
    return ProfileDemand(pid, deficit, deficit if placement is None else placement, force)


def _cal(times=(), owed=None, campaign=None) -> CalendarState:
    return CalendarState(cycles=CycleTracker(deque(times), owed), campaign=campaign)


# --- the campaign gate ------------------------------------------------------


class TestDecideCampaign:
    """The one gate for both origins — go/no-go, kind, and placement order."""

    def test_below_trigger_skips(self):
        d = decide_campaign([_m("p0", 1.0)], None, ncfg(volume_trigger=1e9), False, STORM, t=20.5)
        assert isinstance(d, SkipCampaign)
        assert d.kind is DecisionKind.NOURISH_SKIP
        assert d.row() == {"deficit": 1.0, "trigger": 1e9}

    def test_deficit_exactly_at_trigger_launches(self):
        """The gate is ``>=``: a deficit meeting the trigger exactly mobilizes (the
        boundary the blackout suite pins for windows, pinned here for the gate)."""
        d = decide_campaign([_m("p0", 30.0)], None, ncfg(volume_trigger=30.0), False, STORM, t=20.5)
        assert isinstance(d, LaunchCampaign)
        assert d.kind is DecisionKind.NOURISH_TRIGGER

    def test_above_trigger_orders_by_priority_desc(self):
        ms = [_m("p0", 1.0), _m("p1", 5.0)]
        d = decide_campaign(ms, None, ncfg(volume_trigger=0.001), False, STORM, t=20.5)
        assert isinstance(d, LaunchCampaign)
        assert d.kind is DecisionKind.NOURISH_TRIGGER
        assert d.order == ("p1", "p0")  # higher deficit first

    def test_crew_on_site_bypasses_gate_with_prior_order_first(self):
        ms = [_m("p0", 1.0), _m("p1", 5.0)]
        prior = CampaignCarryover(crew_on_site=True, priority_order=["p0", "p1"])
        d = decide_campaign(ms, prior, ncfg(volume_trigger=1e9), False, STORM, t=20.5)
        assert isinstance(d, LaunchCampaign) and d.resume is True  # gate bypassed
        assert d.order == ("p0", "p1")  # prior rank wins over priority

    def test_resume_keeps_prior_rank_then_priority_for_newcomers(self):
        """The interrupted campaign's order is a promise: profiles the crew already
        owed keep their rank, and only the newcomers re-sort by deficit behind them."""
        ms = [_m("new_small", 1.0), _m("new_big", 5.0), _m("owed", 0.5)]
        prior = CampaignCarryover(crew_on_site=True, priority_order=["owed"])
        d = decide_campaign(ms, prior, ncfg(volume_trigger=1e9), False, STORM, t=20.5)
        assert d.order == ("owed", "new_big", "new_small")

    def test_force_override_mobilizes_below_gate(self):
        d = decide_campaign([_m("p0", 1.0)], None, ncfg(volume_trigger=1e9), True, STORM, t=20.5)
        assert isinstance(d, LaunchCampaign)
        assert d.forced is True  # emergency, not the volume gate
        assert d.kind is DecisionKind.NOURISH_EMERGENCY

    def test_emergency_only_has_no_volume_gate(self):
        """volume_trigger=None (emergency-only reach): no deficit ever mobilizes the
        regular gate; only an emergency force does."""
        # A real emergency-only reach: the validator needs one active trigger, and
        # emergency_volume is assessor-side -- the planner sees only `forced`
        nc = ncfg(volume_trigger=None, emergency_volume=1e12)
        cold = decide_campaign([_m("p0", 1e9)], None, nc, False, STORM, t=20.5)
        assert isinstance(cold, SkipCampaign)  # no gate, no force -> skip
        hot = decide_campaign([_m("p0", 1e9)], None, nc, True, STORM, t=20.5)
        assert isinstance(hot, LaunchCampaign)
        assert hot.kind is DecisionKind.NOURISH_EMERGENCY

    def test_planned_origin_records_the_cycle_kind(self):
        nc = ncfg(volume_trigger=30.0)
        d = decide_campaign([_m("p0", 50.0)], None, nc, False, CYCLE, t=200.0)
        assert d.kind is DecisionKind.NOURISH_CYCLE

    def test_planned_emergency_keeps_its_own_kind(self):
        """An emergency raised inside the cycle window would have mobilized the
        campaign either way, so it is not relabelled as a calendar action."""
        d = decide_campaign([_m("p0", 1.0)], None, ncfg(volume_trigger=1e9), True, CYCLE, t=200.0)
        assert d.kind is DecisionKind.NOURISH_EMERGENCY

    def test_planned_skip_is_marked_cycle(self):
        d = decide_campaign([_m("p0", 1.0)], None, ncfg(volume_trigger=1e9), False, CYCLE, t=200.0)
        assert isinstance(d, SkipCampaign)
        assert d.row() == {"cycle": True, "deficit": 1.0, "trigger": 1e9}


# --- blackout windows -------------------------------------------------------


class TestNextAvailable:
    def test_no_blackout_returns_t(self):
        assert _next_available(10.0, 5.0, []) == pytest.approx(10.0)

    def test_inside_blackout_defers_to_end(self):
        # [10, 15) overlaps [12, 20) → defer to 20
        assert _next_available(10.0, 5.0, [(12.0, 20.0)]) == pytest.approx(20.0)

    def test_entirely_after_blackout_no_defer(self):
        assert _next_available(25.0, 5.0, [(12.0, 20.0)]) == pytest.approx(25.0)

    def test_consecutive_blackouts_skips_both(self):
        # [0,6) hits [5,15) → defer to 15; [15,21) hits [15,25) → defer to 25
        assert _next_available(0.0, 6.0, [(5.0, 15.0), (15.0, 25.0)]) == pytest.approx(25.0)

    def test_start_exactly_at_blackout_end_ok(self):
        # t=20, duration=5: [20,25) — blackout ends at 20 → no overlap (half-open window)
        assert _next_available(20.0, 5.0, [(12.0, 20.0)]) == pytest.approx(20.0)

    def test_the_last_window_is_checked(self):
        """Pins a rejected BeachFX bug: the C++ overlap scan runs ``i < nCount-1`` and
        never checks the LAST blackout segment (BEACHFX_EVENT_SEMANTICS R4.3)."""
        assert _next_available(110.0, 5.0, [(5.0, 15.0), (100.0, 200.0)]) == pytest.approx(200.0)

    def test_the_first_window_is_honored(self):
        """Pins the second half of that bug: the C++ conflict test is ``nIndex > 0``,
        silently ignoring an overlap with window index 0."""
        assert _next_available(10.0, 5.0, [(8.0, 30.0)]) == pytest.approx(30.0)


# --- the crew-clock fold ----------------------------------------------------


def _nc(production_rate=4.0, **kw):
    return ncfg(volume_trigger=30.0, production_rate=production_rate, **kw)


class TestPlanPlacements:
    def test_cold_start_pays_the_mobilization_lead_time(self):
        nc = _nc(production_rate=100.0)
        nc.mobilization_days = 5.0
        plan, carry = plan_placements([_m("p0", 42.0)], 20.5, 1000.0, nc, resume=False)
        assert carry is None
        (p,) = plan
        assert isinstance(p, Placement)
        assert p.t_start == pytest.approx(25.5)

    def test_resume_puts_the_crew_straight_to_work(self):
        nc = _nc(production_rate=100.0)
        nc.mobilization_days = 5.0
        plan, _ = plan_placements([_m("p0", 42.0)], 20.5, 1000.0, nc, resume=True)
        assert plan[0].t_start == pytest.approx(20.5)

    def test_the_crew_is_serial(self):
        plan, carry = plan_placements(
            [_m("p0", 40.0), _m("p1", 20.0)], 0.0, 1000.0, _nc(), resume=True
        )
        assert carry is None
        p0, p1 = plan
        assert p1.t_start == pytest.approx(p0.t_end)

    def test_blackout_shift_defers_then_places(self):
        nc = _nc(production_rate=100.0)
        nc.blackout_windows = [(20.0, 30.0)]
        plan, carry = plan_placements([_m("p0", 42.0)], 20.5, 1000.0, nc, resume=True)
        assert carry is None
        defer, place = plan
        assert isinstance(defer, DeferBlackout)
        assert (defer.requested, defer.deferred_to) == (pytest.approx(20.5), pytest.approx(30.0))
        assert place.t_start == pytest.approx(30.0)

    def test_blocked_when_the_shift_lands_past_the_next_storm(self):
        nc = _nc(production_rate=100.0)
        nc.blackout_windows = [(20.0, 50.0)]
        plan, carry = plan_placements([_m("p0", 42.0)], 20.5, 34.0, nc, resume=True)
        assert [type(d) for d in plan] == [DeferBlackout]  # no placement started
        assert carry == CampaignCarryover(crew_on_site=False, priority_order=["p0"])

    def test_blocked_at_exactly_the_storm_instant(self):
        """Precedence rule: the storm closes the interval, so a start AT ``t_next``
        belongs to the next gap — the ``>=`` is load-bearing."""
        nc = _nc(production_rate=100.0)
        nc.mobilization_days = 13.5
        plan, carry = plan_placements([_m("p0", 42.0)], 20.5, 34.0, nc, resume=False)
        assert plan == []
        assert carry == CampaignCarryover(crew_on_site=False, priority_order=["p0"])

    def test_defer_policy_holds_the_whole_placement(self):
        nc = _nc(production_rate=4.0)
        nc.storm_conflict = "defer"
        plan, carry = plan_placements([_m("p0", 62.0)], 20.501, 34.0, nc, resume=True)
        (d,) = plan
        assert isinstance(d, DeferStorm)
        assert d.storm == pytest.approx(34.0)
        assert carry == CampaignCarryover(crew_on_site=False, priority_order=["p0"])

    def test_interrupt_places_the_fraction_that_fits(self):
        plan, carry = plan_placements(
            [_m("p0", 62.0), _m("p1", 62.0)], 20.501, 34.0, _nc(production_rate=4.0), resume=True
        )
        intr, place = plan
        assert isinstance(intr, Interrupt)
        assert intr.placed_fraction == pytest.approx(13.499 / 15.5)
        assert place.cut_by_storm is True
        assert place.placed_m3 == pytest.approx(62.0 * 13.499 / 15.5)
        # interrupted profile first in the carry-forward, the unreached one behind it
        assert carry == CampaignCarryover(crew_on_site=True, priority_order=["p0", "p1"])

    def test_a_placement_ending_exactly_at_the_storm_is_a_partial(self):
        """The boundary that motivated ``cut_by_storm``: fraction is exactly 1.0 here,
        yet the segment must still close as storm-cut (EENS), not completed (EEN)."""
        plan, carry = plan_placements(
            [_m("p0", 54.0)], 20.5, 34.0, _nc(production_rate=4.0), resume=True
        )
        intr, place = plan
        assert isinstance(intr, Interrupt)
        assert place.fraction == pytest.approx(1.0)
        assert place.cut_by_storm is True
        assert carry is not None and carry.crew_on_site is True

    def test_a_zero_volume_placement_cannot_wedge_the_crew(self):
        """Pins a rejected BeachFX bug: a computed duration ≤ 0 leaves the C++
        emergency queue permanently blocked (R4.8).  Here it is a trivial instant
        placement and the crew moves on."""
        plan, carry = plan_placements(
            [_m("p0", 0.0, placement=0.0), _m("p1", 40.0)], 0.0, 1000.0, _nc(), resume=True
        )
        assert carry is None
        z, p1 = plan
        assert z.t_start == z.t_end == pytest.approx(0.0)
        assert p1.profile_id == "p1"  # the crew was not wedged


# --- planned cycles ---------------------------------------------------------


class TestPlanNextCycle:
    def test_nothing_owed(self):
        assert plan_next_cycle(_cal(), 100.0, 0.0) is None

    def test_a_cycle_due_exactly_at_the_gap_end_is_not_owed_yet(self):
        """Precedence rule: the storm closes the interval — a cycle dated AT ``t_b``
        belongs to the next gap, not this one."""
        assert plan_next_cycle(_cal(times=[100.0]), 100.0, 0.0) is None

    def test_fires_on_its_date_when_recovery_is_done(self):
        d = plan_next_cycle(_cal(times=[50.0]), 100.0, 30.0)
        assert d == FireCycle(t_fire=50.0, erode_to=50.0)

    def test_waits_out_an_in_progress_recovery(self):
        """BeachFX deferral code 1: the fire time is ``max(owed, recovery_done)`` —
        the cycle assesses the fully recovered bed, at that exact instant."""
        d = plan_next_cycle(_cal(times=[25.0]), 100.0, 41.501)
        assert d == FireCycle(t_fire=41.501, erode_to=41.501)

    def test_defers_when_the_crew_is_busy(self):
        cal = _cal(
            times=[50.0], campaign=CampaignCarryover(crew_on_site=True, priority_order=["p0"])
        )
        d = plan_next_cycle(cal, 100.0, 0.0)
        assert isinstance(d, DeferCycle)
        assert d.t == pytest.approx(50.0)  # logged at the date it concerns
        assert d.crew_busy is True

    def test_defers_when_recovery_pushes_it_out_of_the_gap(self):
        d = plan_next_cycle(_cal(times=[50.0]), 100.0, 120.0)
        assert d == DeferCycle(t=50.0, would_fire=120.0, gap_end=100.0, crew_busy=False)

    def test_defers_when_the_fire_time_hits_the_gap_end_exactly(self):
        d = plan_next_cycle(_cal(times=[50.0]), 100.0, 100.0)
        assert isinstance(d, DeferCycle)  # >= — no room to start at the boundary

    def test_planning_commits_nothing(self):
        cal = _cal(times=[50.0])
        plan_next_cycle(cal, 100.0, 0.0)
        assert cal.cycles.owed is None and list(cal.cycles.times) == [50.0]


class TestErosionSplit:
    def test_no_cycle_due_erodes_the_whole_gap(self):
        assert erosion_split(_cal(), 20.5, 100.0, 41.5) == pytest.approx(100.0)

    def test_a_due_cycle_holds_erosion_at_the_recovery_completion(self):
        assert erosion_split(_cal(times=[50.0]), 20.5, 100.0, 41.5) == pytest.approx(41.5)

    def test_the_split_clamps_to_the_gap(self):
        assert erosion_split(_cal(times=[50.0]), 20.5, 100.0, 300.0) == pytest.approx(100.0)
        assert erosion_split(_cal(times=[50.0]), 20.5, 100.0, 5.0) == pytest.approx(20.5)
