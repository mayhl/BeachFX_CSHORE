from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from .config import ReachConfig
from .decision import CalendarState, CycleTracker, DeferCycle, emit
from .decision.planner import erosion_split, plan_next_cycle
from .interstorm import run_interstorm
from .nourishment import recovery_duration, run_campaign, run_planned_campaign
from .profile import Profiles
from .results import ResultsSink, RunMeta
from .storm import build_storm_schedule, run_parallel_cshore
from .types import SnapshotLabel

if TYPE_CHECKING:
    from .runner.base import CSHORERunner

log = logging.getLogger(__name__)

# Nudge the Phase-3 campaign an instant past the storm response so its events
# (SEN / EEN, and any pre-placement recovery) sort *after* every profile's
# PostStorm at the same storm-end instant, rather than interleaving with them in
# the reach timeline.  Both the crew clock and the recovery baseline shift by this
# together, so no spurious recovery is emitted and placement intervals are preserved.
_CAMPAIGN_START_OFFSET_DAYS = 0.001


# ---------------------------------------------------------------------------
# Reach — one (reach × alternative × lifecycle) run
# ---------------------------------------------------------------------------


@dataclass
class Reach:
    """One (reach × alternative × lifecycle) run — the single object the interval
    loop operates on.

    Bundles the three 1:1 facets of a reach: its **transects** (``profiles``), its
    **identity + config + output** (``reach_id`` … ``results``, ``runner``), and the
    **interval-loop state** (``t``, and the management calendar in ``calendar``).
    """

    profiles: Profiles
    cfg: ReachConfig
    results: ResultsSink
    runner: CSHORERunner
    sim_start: datetime
    reach_id: str = "test"
    alternative_id: str = "FWOP"
    lifecycle: int = 0
    longshore_widths: list[float] = field(default_factory=list)
    # interval-loop state
    t: float = 0.0
    calendar: CalendarState = field(default_factory=CalendarState)

    def __post_init__(self) -> None:
        self.profiles = Profiles(self.profiles)  # accept a plain list too

    def _recovery_done(self, t_storm: float) -> float:
        """When the last profile finishes recovering from the storm at ``t_storm``."""
        return t_storm + max((recovery_duration(p, self.cfg) for p in self.profiles), default=0.0)

    def _erode_to_split(self, t_a: float, t_b: float, recovery_done: float = 0.0) -> float:
        """Erode the gap ``[t_a, t_b]`` up to the planner's split point, and say how far.

        The split policy (hold the erosion back when a planned cycle is due, so the
        cycle sees the gap's erosion accrue on its own date) lives in
        ``decision.planner.erosion_split``; this just applies the ticks.
        """
        t_split = erosion_split(self.calendar, t_a, t_b, recovery_done)
        run_interstorm(self.profiles, t_a, t_split, self.cfg)
        return t_split

    def _run_due_cycles(self, t_a: float, t_b: float, recovery_done: float = 0.0) -> float:
        """Fire every planned cycle owed before ``t_b``, oldest first, or leave it owed.

        The planner decides (``plan_next_cycle``: fire vs defer); this loop commits the
        tracker, erodes forward to each fired cycle's date, and launches its campaign.
        It stays a loop rather than one planned batch because each decision depends on
        the campaign the previous cycle launched — a crew left busy defers everything
        behind it.  Returns how far the gap has been eroded so the caller can finish
        the remainder.
        """
        t_eroded = t_a
        while (d := plan_next_cycle(self.calendar, t_b, recovery_done)) is not None:
            self.calendar.cycles.take_due(t_b)  # commit: this cycle is now the owed one
            if isinstance(d, DeferCycle):
                emit(self.results, d.kind, d.t, **d.row())
                break

            # Bring the bed forward to this cycle's date before the crew assesses it.
            run_interstorm(self.profiles, t_eroded, d.erode_to, self.cfg)
            t_eroded = d.erode_to

            self.calendar.campaign = run_planned_campaign(
                self.profiles,
                d.t_fire,
                t_b,
                self.cfg,
                self.results,
                longshore_widths=self.longshore_widths or None,
                prior=self.calendar.campaign,
            )
            self.calendar.cycles.clear()
        return t_eroded

    def run(self, storms_df: pd.DataFrame, sim_end: float) -> None:
        """Run the interval loop over the storm schedule.

        Phases per storm:
          Phase 1   — pre-storm erosion / SLC   (run_interstorm)
          Phase 1.5 — planned cycle due in the pre-storm gap
          Phase 2   — parallel CSHORE           (run_parallel_cshore)
          Phase 2.5 — inter-storm gap erosion   (run_interstorm over [storm, next])
          Phase 3   — campaign: recovery + nourishment (run_campaign)
          Phase 3.5 — planned cycle due in the post-storm gap

        Gap erosion PAUSES at a planned cycle (Phase 2.5 stops at the cycle's date, the
        remainder is eroded after Phase 3.5), so the cycle assesses the beach as of its
        own date rather than one already eroded to the gap's end.  With no cycle due, the
        gap erodes in one pass as before.
        """
        widths = self.longshore_widths or None
        self.profiles.snapshot_all(SnapshotLabel.INIT, 0.0)

        schedule = build_storm_schedule(
            storms_df, self.sim_start, self.cfg, lifecycle=self.lifecycle
        )
        n = len(schedule)
        self.calendar = CalendarState(
            cycles=CycleTracker.build(self.cfg.nourishment, self.sim_start, sim_end)
        )

        # A stormless lifecycle still erodes and still nourishes on its calendar; the
        # storm loop below would skip both, so run the whole window as one quiet gap.
        if n == 0:
            t_split = self._erode_to_split(self.t, sim_end)
            t_eroded = self._run_due_cycles(t_split, sim_end)
            run_interstorm(self.profiles, t_eroded, sim_end, self.cfg)
            self.t = sim_end

        for i, storm in enumerate(schedule):
            t_next = schedule[i + 1].t if i + 1 < n else sim_end
            storm_end = storm.t + storm.duration  # PostStorm lands here
            campaign_start = storm_end + _CAMPAIGN_START_OFFSET_DAYS  # recovery/nourishment begin

            # Phase 1 / 1.5 — pre-storm erosion / SLC, pausing at any planned cycle that
            # falls in the gap.  Only the LEADING gap is non-empty (later gaps close at
            # the next storm, so their cycles are already spent in Phase 3.5), and there
            # is no storm behind it to recover from — so no recovery deferral applies.
            t_split = self._erode_to_split(self.t, storm.t)
            t_eroded = self._run_due_cycles(t_split, storm.t)
            run_interstorm(self.profiles, t_eroded, storm.t, self.cfg)

            # Phase 2 — CSHORE (all profiles in parallel); PostStorm at storm end.
            outcomes = run_parallel_cshore(
                self.profiles, storm.t, storm.forcing, self.runner, self.cfg, t_post=storm_end
            )
            for o in outcomes:
                if not o.inundated:
                    self.results.record_storm_hazard(o.profile.id, storm.t, o.result)
                else:
                    # INUNDATION: CSHORE failed. Interim handling (undecided by the
                    # group) — skip the storm, reuse the profile, surface a warning,
                    # and skip Phase 3 for this profile (no storm ⇒ no recovery).
                    self.results.record_warning(
                        o.profile.id,
                        storm.t,
                        f"CSHORE failed for storm {storm.storm_id!r} — "
                        "storm skipped, profile reused (INUNDATION)",
                    )

            # Phase 2.5 — inter-storm erosion / SLC over the gap [storm end, next storm].
            # Applied BEFORE the campaign so Periodic ticks stay ≤ the recovery time
            # (monotonic snapshots); recovery/nourishment then act on the eroded bed.
            # Stops at a planned cycle, if one is due before the next storm.
            recovery_done = self._recovery_done(campaign_start)
            t_split = self._erode_to_split(storm_end, t_next, recovery_done)

            # Phase 3 — campaign (recovery + nourishment) begins just after storm end.
            # storm_at_next distinguishes a recovery cut short by the next storm
            # (RECS) from one that runs to the sim end on the last storm (REC).
            self.t, self.calendar.campaign = run_campaign(
                outcomes,
                campaign_start,
                t_next,
                self.cfg,
                self.results,
                longshore_widths=widths,
                prior=self.calendar.campaign,
                storm_at_next=i + 1 < n,
            )

            # Phase 3.5 — a planned cycle falling in this gap, once the storm's
            # recoveries have run out and the crew is free of storm work.  The rest of
            # the gap's erosion then lands on whatever bed the cycle left behind.
            t_eroded = self._run_due_cycles(t_split, t_next, recovery_done)
            run_interstorm(self.profiles, t_eroded, t_next, self.cfg)

        self.profiles.snapshot_all(SnapshotLabel.EndIteration, self.t)
        self.results.flush(
            self.profiles,
            RunMeta(self.reach_id, self.alternative_id, self.sim_start, self.lifecycle),
        )
        log.info(
            "reach run complete — reach=%s alt=%s lc=%d",
            self.reach_id,
            self.alternative_id,
            self.lifecycle,
        )
