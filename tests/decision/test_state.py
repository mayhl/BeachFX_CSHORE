"""Structural contracts of the decision layer: picklable state, physics-free imports.

Lifecycles parallelize as processes, so everything the decision layer carries
across an interval — and everything it returns — must survive a pickle round-trip.
And the layer's whole point is that it reasons in scalars and dates: the import
test is the seam, asserted structurally.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
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
)
from erosion.types import DecisionKind

STATES = [
    CalendarState(),
    CalendarState(
        cycles=CycleTracker(times=deque([100.0, 465.0]), owed=100.0),
        campaign=CampaignCarryover(crew_on_site=True, priority_order=["p1", "p0"]),
    ),
]

DECISIONS = [
    ProfileDemand("p0", 42.0, 42.0, force=True),
    Placement("p0", 20.501, 20.921, 42.0, 42.0),
    LaunchCampaign(
        kind=DecisionKind.NOURISH_TRIGGER,
        t=20.501,
        total_deficit=62.0,
        resume=False,
        forced=False,
        order=("p0", "p1"),
        placements=(Placement("p0", 20.501, 36.0, 62.0, 62.0, fraction=0.87),),
        carry_forward=CampaignCarryover(crew_on_site=True, priority_order=["p1"]),
    ),
    SkipCampaign(t=20.5, total_deficit=22.0, trigger=30.0, cycle=True),
    Interrupt(t=34.0, profile_id="p0", placed_fraction=0.87),
    DeferBlackout(t=20.501, profile_id="p0", requested=20.501, deferred_to=38.0),
    DeferStorm(t=20.501, profile_id="p0", would_start=20.501, would_end=36.0, storm=34.0),
    DeferCycle(t=100.0, would_fire=120.0, gap_end=110.0, crew_busy=True),
    FireCycle(t_fire=200.0, erode_to=200.0),
]


@pytest.mark.parametrize("obj", STATES + DECISIONS, ids=lambda o: type(o).__name__)
def test_round_trips_through_pickle(obj):
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_calendar_state_round_trip_preserves_the_backlog_mechanics():
    """Not just equality: the revived tracker still hands out cycles in order —
    the deque and the owed slot survive as working state, not as reprs."""
    cal = CalendarState(cycles=CycleTracker(times=deque([100.0, 465.0])))
    revived: CalendarState = pickle.loads(pickle.dumps(cal))
    assert revived.cycles.take_due(200.0) == 100.0
    revived.cycles.clear()
    assert revived.cycles.take_due(500.0) == 465.0


def test_importing_the_decision_layer_pulls_in_no_physics():
    """The seam, asserted structurally: `import erosion.decision` in a fresh
    interpreter must not drag in profiles, assessors, CSHORE, or numpy-heavy
    physics modules.  Config is allowed (the planner reads it); beds are not."""
    code = (
        "import sys; import erosion.decision; "
        "banned = ('erosion.profile', 'erosion.nourishment', 'erosion.storm', "
        "'erosion.metrics', 'erosion.runner', 'erosion.reach', 'erosion.interstorm'); "
        "bad = [m for m in sys.modules if m.startswith(banned)]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
