from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from .config import ReachConfig
from .interstorm import run_interstorm
from .nourishment import (
    ActiveCampaign,
    CycleTracker,
    recovery_duration,
    run_campaign,
    run_scheduled_campaign,
)
from .profile import Profiles
from .results import ResultsSink, RunMeta
from .storm import build_storm_schedule, run_parallel_cshore
from .types import DecisionKind, SnapshotLabel

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
    **interval-loop state** (``t``, ``active_campaign``).
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
    active_campaign: ActiveCampaign | None = None

    def __post_init__(self) -> None:
        self.profiles = Profiles(self.profiles)  # accept a plain list too

    def _recovery_done(self, t_storm: float) -> float:
        """When the last profile finishes recovering from the storm at ``t_storm``."""
        return t_storm + max((recovery_duration(p, self.cfg) for p in self.profiles), default=0.0)

    def _fire_time(self, cycles: CycleTracker, t_b: float, recovery_done: float) -> float | None:
        """When the next planned cycle owed before ``t_b`` would actually place, or None.

        Not its calendar date: a cycle landing while the reach is still recovering waits
        out the recovery (BeachFX's deferral code 1) rather than cutting it short.
        """
        owed = cycles.peek_due(t_b)
        return None if owed is None else max(owed, recovery_done)

    def _erode_to_cycle(
        self, cycles: CycleTracker, t_a: float, t_b: float, recovery_done: float = 0.0
    ) -> float:
        """Erode the gap ``[t_a, t_b]`` up to where the storm campaign acts, and say how far.

        A gap is normally eroded in ONE pass, before the campaign — so the campaign
        assesses a bed already eroded to the gap's end.  A planned cycle can't live with
        that: the storm campaign would restore the beach to the template and swallow the
        whole gap's erosion on the way, leaving the cycle nothing to find on its own date.

        So when a cycle is due, the erosion is held back at the campaign's own span (the
        last recovery completion) and the remainder is applied *after* the campaign, in
        ``_run_due_cycles``, where the cycle can actually see it accrue.  With no cycle
        due the gap erodes in one pass exactly as before.
        """
        if self._fire_time(cycles, t_b, recovery_done) is None:
            run_interstorm(self.profiles, t_a, t_b, self.cfg)
            return t_b

        t_split = max(t_a, min(t_b, recovery_done))
        run_interstorm(self.profiles, t_a, t_split, self.cfg)
        return t_split

    def _run_due_cycles(
        self, cycles: CycleTracker, t_a: float, t_b: float, recovery_done: float = 0.0
    ) -> float:
        """Fire every planned cycle owed before ``t_b``, oldest first, or leave it owed.

        Erodes forward to each cycle's date as it goes (the bed arrives at ``t_a`` already
        eroded to the first one), and returns how far the gap has been eroded so the
        caller can finish the remainder.

        Two things can push a cycle out of its gap.  It can land while the reach is still
        recovering from the storm that opened the gap, so it waits for ``recovery_done``.
        And a crew already carrying unfinished storm work (``active_campaign``) owns the
        window, so the cycle yields to it.  Either way, a cycle that can no longer start
        before ``t_b`` stays owed and is retried in the next gap, blocking those behind it.
        """
        t_eroded = t_a
        while (owed := cycles.next_due(t_b)) is not None:
            t_fire = max(owed, recovery_done)

            if self.active_campaign is not None or t_fire >= t_b:
                self.results.record_decision(
                    DecisionKind.CYCLE_DEFER,
                    owed,
                    would_fire=t_fire,
                    gap_end=t_b,
                    crew_busy=self.active_campaign is not None,
                )
                break

            # Bring the bed forward to this cycle's date before the crew assesses it.
            run_interstorm(self.profiles, t_eroded, t_fire, self.cfg)
            t_eroded = t_fire

            self.active_campaign = run_scheduled_campaign(
                self.profiles,
                t_fire,
                t_b,
                self.cfg,
                self.results,
                longshore_widths=self.longshore_widths or None,
                prior=self.active_campaign,
            )
            cycles.clear()
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
        cycles = CycleTracker.build(self.cfg.nourishment, self.sim_start, sim_end)

        # A stormless lifecycle still erodes and still nourishes on its calendar; the
        # storm loop below would skip both, so run the whole window as one quiet gap.
        if n == 0:
            t_split = self._erode_to_cycle(cycles, self.t, sim_end)
            t_eroded = self._run_due_cycles(cycles, t_split, sim_end)
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
            t_split = self._erode_to_cycle(cycles, self.t, storm.t)
            t_eroded = self._run_due_cycles(cycles, t_split, storm.t)
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
            t_split = self._erode_to_cycle(cycles, storm_end, t_next, recovery_done)

            # Phase 3 — campaign (recovery + nourishment) begins just after storm end.
            # storm_at_next distinguishes a recovery cut short by the next storm
            # (RECS) from one that runs to the sim end on the last storm (REC).
            self.t, self.active_campaign = run_campaign(
                outcomes,
                campaign_start,
                t_next,
                self.cfg,
                self.results,
                longshore_widths=widths,
                prior=self.active_campaign,
                storm_at_next=i + 1 < n,
            )

            # Phase 3.5 — a planned cycle falling in this gap, once the storm's
            # recoveries have run out and the crew is free of storm work.  The rest of
            # the gap's erosion then lands on whatever bed the cycle left behind.
            t_eroded = self._run_due_cycles(cycles, t_split, t_next, recovery_done)
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
