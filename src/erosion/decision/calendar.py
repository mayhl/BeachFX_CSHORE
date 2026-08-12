"""The management calendar — cycle backlog and campaign carry-forward.

This is the decision layer's persistent state: everything the reach remembers
about management time between storm intervals, gathered so it can cross a
process boundary in one picklable piece.  BeachFX lays every planned cycle down
up front: the first at ``gdatePlannedNourishmentStartDate``, then one every
``365 × gdwNourishmentTimeIncrement`` days to the end of the simulation
(cShoreResponseIteration.cpp:137-155).  We do the same, then track which cycle
the reach still owes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from ..nourishment.config import NourishmentConfig

_DAYS_PER_YEAR = 365.0


def cycle_times(
    ncfg: NourishmentConfig | None,
    sim_start: datetime,
    sim_end: float,
) -> list[float]:
    """Planned cycle start times (days from ``sim_start``), first through last in window."""
    if ncfg is None or ncfg.cycle_interval_years is None:
        return []
    step = ncfg.cycle_interval_years * _DAYS_PER_YEAR
    t = (
        0.0
        if ncfg.cycle_start_date is None
        else (ncfg.cycle_start_date - sim_start).total_seconds() / 86400.0
    )
    times = []
    while t <= sim_end:
        if t >= 0.0:
            times.append(t)
        t += step
    return times


@dataclass
class CycleTracker:
    """Which planned cycle the reach owes, as the interval loop walks the gaps.

    Cycles are handed out ONE at a time, oldest first, and an unrun cycle stays owed —
    it blocks the ones behind it until it either runs or the window closes.  A cycle
    deferred out of its gap is therefore retried in the next gap rather than skipped,
    and a backlog never runs out of order.  A long quiet gap can legitimately clear
    several cycles in sequence; the later ones find a restored beach and skip on the
    volume gate, which is the honest outcome rather than a silent collapse.
    """

    times: deque[float] = field(default_factory=deque)
    owed: float | None = None

    @classmethod
    def build(
        cls, ncfg: NourishmentConfig | None, sim_start: datetime, sim_end: float
    ) -> CycleTracker:
        return cls(deque(cycle_times(ncfg, sim_start, sim_end)))

    def peek_due(self, t_end: float) -> float | None:
        """``next_due`` without taking it — lets the interval loop see where to pause
        the gap's erosion before it commits to running the cycle."""
        if self.owed is not None:
            return self.owed
        return self.times[0] if self.times and self.times[0] < t_end else None

    def next_due(self, t_end: float) -> float | None:
        """The cycle the reach owes before ``t_end`` — the one still owed from an
        earlier gap, else the next one to come due.  None when nothing is owed yet."""
        if self.owed is None and self.times and self.times[0] < t_end:
            self.owed = self.times.popleft()
        return self.owed

    def clear(self) -> None:
        """The owed cycle ran — the reach owes nothing until the next one comes due."""
        self.owed = None


@dataclass
class CampaignCarryover:
    """Carry-forward state when a nourishment campaign is interrupted by a storm."""

    crew_on_site: bool
    priority_order: list[str]  # profile IDs in remaining campaign order


@dataclass
class CalendarState:
    """Everything the decision layer remembers between intervals — one picklable piece.

    The crew clock is deliberately NOT here: it lives only inside one planning call
    (a fold over that campaign's placements); its cross-interval residue is
    ``campaign.crew_on_site``.
    """

    cycles: CycleTracker = field(default_factory=CycleTracker)
    campaign: CampaignCarryover | None = None
