from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from .config import ReachConfig
from .nourishment import ActiveCampaign
from .results import NullResultsSink, ResultsSink, RunMeta
from .types import SnapshotLabel

if TYPE_CHECKING:
    from .profile import Profile
    from .runner.base import CSHORERunner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy context (retained for ParquetResultsSink.flush() signature)
# ---------------------------------------------------------------------------

@dataclass
class ReachContext:
    reach_id:         str
    alternative_id:   str
    results:          ResultsSink
    sim_start:        datetime
    cfg:              ReachConfig    = field(default_factory=ReachConfig)
    longshore_widths: list[float]   = field(default_factory=list)

    @classmethod
    def minimal(cls, cfg: ReachConfig | None = None) -> ReachContext:
        return cls(
            reach_id="test",
            alternative_id="FWOP",
            results=NullResultsSink(),
            sim_start=datetime(2025, 1, 1),
            cfg=cfg if cfg is not None else ReachConfig(),
        )


# ---------------------------------------------------------------------------
# Interval-loop state
# ---------------------------------------------------------------------------

@dataclass
class ReachState:
    t:                float                  = 0.0
    active_campaign:  ActiveCampaign | None  = None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_lifecycle(
    profiles: list[Profile],
    storms_df: pd.DataFrame,
    sim_start: datetime,
    sim_end: float,
    cfg: ReachConfig,
    runner: CSHORERunner,
    ctx: ReachContext,
    lifecycle: int = 0,
) -> None:
    """Run one (reach, alternative, lifecycle) via the interval-loop architecture.

    Three phases per storm:
      Phase 1 — inter-storm erosion / SLC  (run_interstorm)
      Phase 2 — parallel CSHORE             (run_parallel_cshore)
      Phase 3 — campaign: recovery + nourishment (run_campaign)
    """
    from .interstorm import run_interstorm
    from .nourishment import run_campaign
    from .storm import build_storm_schedule, run_parallel_cshore

    state    = ReachState()
    widths   = ctx.longshore_widths or None

    # INIT snapshot
    for p in profiles:
        p.snapshot(SnapshotLabel.INIT, 0.0)

    schedule = build_storm_schedule(storms_df, sim_start, cfg, lifecycle=lifecycle)
    n        = len(schedule)

    for i, storm in enumerate(schedule):
        t_next = schedule[i + 1].t if i + 1 < n else sim_end

        # Phase 1 — erosion / SLC
        run_interstorm(profiles, state.t, storm.t, cfg)

        # Phase 2 — CSHORE (all profiles in parallel)
        results, zb_pre_new = run_parallel_cshore(
            profiles, storm.t, storm.forcing, runner, cfg,
        )
        for p, r in zip(profiles, results):
            if r is not None:
                ctx.results.record_storm_hazard(p.id, storm.t, r)

        # Phase 3 — campaign
        state.t, state.active_campaign = run_campaign(
            profiles, zb_pre_new, storm.t, t_next, cfg,
            ctx.results,
            longshore_widths=widths,
            prior=state.active_campaign,
        )

    # EndIteration snapshot
    for p in profiles:
        p.snapshot(SnapshotLabel.EndIteration, state.t)

    meta = RunMeta(
        reach_id=ctx.reach_id,
        alternative_id=ctx.alternative_id,
        sim_start=sim_start,
        lifecycle=lifecycle,
    )
    ctx.results.flush(profiles, meta)
    log.info("run_lifecycle complete — reach=%s alt=%s lc=%d",
             ctx.reach_id, ctx.alternative_id, lifecycle)
