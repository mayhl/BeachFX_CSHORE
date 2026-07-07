from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from .config import ReachConfig
from .nourishment import ActiveCampaign
from .profile import Profiles
from .results import ResultsSink, RunMeta
from .types import SnapshotLabel

if TYPE_CHECKING:
    from .runner.base import CSHORERunner

log = logging.getLogger(__name__)


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

    def run(self, storms_df: pd.DataFrame, sim_end: float) -> None:
        """Run the interval loop over the storm schedule.

        Three phases per storm:
          Phase 1   — pre-storm erosion / SLC   (run_interstorm)
          Phase 2   — parallel CSHORE           (run_parallel_cshore)
          Phase 2.5 — inter-storm gap erosion   (run_interstorm over [storm, next])
          Phase 3   — campaign: recovery + nourishment (run_campaign)
        """
        from .interstorm import run_interstorm
        from .nourishment import run_campaign
        from .storm import build_storm_schedule, run_parallel_cshore

        widths = self.longshore_widths or None
        self.profiles.snapshot_all(SnapshotLabel.INIT, 0.0)

        schedule = build_storm_schedule(
            storms_df, self.sim_start, self.cfg, lifecycle=self.lifecycle
        )
        n = len(schedule)

        for i, storm in enumerate(schedule):
            t_next = schedule[i + 1].t if i + 1 < n else sim_end

            # Phase 1 — pre-storm erosion / SLC (only non-empty before the FIRST
            # storm; later gaps are eroded in Phase 2.5 below).
            run_interstorm(self.profiles, self.t, storm.t, self.cfg)

            # Phase 2 — CSHORE (all profiles in parallel)
            results, zb_pre_new = run_parallel_cshore(
                self.profiles, storm.t, storm.forcing, self.runner, self.cfg
            )
            inundated: set[str] = set()
            for p, r in zip(self.profiles, results):
                if r is not None:
                    self.results.record_storm_hazard(p.id, storm.t, r)
                else:
                    # INUNDATION: CSHORE failed. Interim handling (undecided by the
                    # group) — skip the storm, reuse the profile, surface a warning,
                    # and skip Phase 3 for this profile (no storm ⇒ no recovery).
                    inundated.add(p.id)
                    self.results.record_warning(
                        p.id,
                        storm.t,
                        f"CSHORE failed for storm {storm.storm_id!r} — "
                        "storm skipped, profile reused (INUNDATION)",
                    )

            # Phase 2.5 — inter-storm erosion / SLC over the gap [storm, next storm].
            # Applied BEFORE the campaign so Periodic ticks stay ≤ the recovery time
            # (monotonic snapshots); recovery/nourishment then act on the eroded bed.
            run_interstorm(self.profiles, storm.t, t_next, self.cfg)

            # Phase 3 — campaign
            self.t, self.active_campaign = run_campaign(
                self.profiles,
                zb_pre_new,
                storm.t,
                t_next,
                self.cfg,
                self.results,
                longshore_widths=widths,
                prior=self.active_campaign,
                inundated=inundated,
            )

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
