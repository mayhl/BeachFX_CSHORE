"""Reach-scope decision layer.

Owns decision emission, the management calendar (``CalendarState``), and the
decision vocabulary; the interval planner lands here as the decide/execute split
proceeds.  This package must never import physics at runtime (``profile`` /
``campaign`` / ``assess``) — its currency is scalars, dates, and config.
"""

from .calendar import CalendarState, CampaignCarryover, CycleTracker, cycle_times
from .model import (
    DeferBlackout,
    DeferCycle,
    DeferStorm,
    FireCycle,
    Interrupt,
    LaunchCampaign,
    Placement,
    ProfileDemand,
    SkipCampaign,
    emit,
)
from .planner import (
    ReachDecision,
    ReachNourishmentDecider,
    decide_campaign,
    erosion_split,
    plan_next_cycle,
    plan_placements,
)

__all__ = [
    "CampaignCarryover",
    "CalendarState",
    "CycleTracker",
    "DeferBlackout",
    "DeferCycle",
    "DeferStorm",
    "FireCycle",
    "Interrupt",
    "LaunchCampaign",
    "Placement",
    "ProfileDemand",
    "SkipCampaign",
    "ReachDecision",
    "ReachNourishmentDecider",
    "cycle_times",
    "decide_campaign",
    "emit",
    "erosion_split",
    "plan_next_cycle",
    "plan_placements",
]
