"""Campaign execution: one decide-then-execute path for both campaign origins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..decision import Placement, SkipCampaign, emit
from ..decision.calendar import CampaignCarryover
from ..decision.model import ProfileDemand
from ..decision.planner import decide_campaign, plan_placements
from ..profile import FullNourishment, PartialNourishment
from ..types import CampaignKind
from .campaign import (
    _recover_profile,
    _Work,
    _Works,
)

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..profile import Profile
    from ..results import ResultsSink
    from ..storm import StormOutcome


@dataclass
class _CampaignScheduler:
    """Tier-3 executor: applies a planned campaign — recovery to each placement
    start, the fill itself, and the audit emission of the planner's decisions.

    All policy (blackout shift, storm-conflict defer/interrupt, crew clock) lives
    in ``erosion.decision.planner.plan_placements``; what remains here is bed
    mutation and bookkeeping.  ``kind`` selects the morphology label pair
    (``SEN``/``EEN`` post-storm, ``SSN``/``ESN`` for a planned cycle) and whether
    a placement is preceded by recovery — a planned cycle only fires in a quiet
    window, so it has none left to run.
    """

    t_storm: float  # campaign start — storm end, or the cycle date for a planned cycle
    t_next: float
    cfg: ReachConfig
    sink: ResultsSink
    storm_at_next: bool = False  # t_next is a following storm (vs sim/window end) → RECS on cutoff
    kind: CampaignKind = CampaignKind.STORM

    def _apply(self, p: Placement, w: _Work) -> None:
        """Execute one planned placement on its work item."""
        # recovery up to the placement start; cut short here → RECN (crew, not storm).
        # A planned cycle has none to run — it only fires past the recovery completion.
        if self.kind is CampaignKind.STORM:
            _recover_profile(w, self.t_storm, p.t_start, self.cfg, nourish_at_end=True)
        label = self.kind.start_label
        w.profile.snapshot(label, p.t_start)
        w.profile.record_event("NourishmentStart", p.t_start, label)

        if p.cut_by_storm:
            PartialNourishment(
                t=p.t_end,
                template_zb=w.plan.template_zb,
                fraction=p.fraction,
                label=self.kind.partial_label,
            ).apply(w.profile)
            self.sink.record_nourishment(
                w.profile.id,
                p.t_start,
                p.t_end,
                p.placed_m3,
                "PartialNourishment",
                borrow_m3=p.borrow_m3,
            )
        else:
            FullNourishment(
                t=p.t_end, template_zb=w.plan.template_zb, label=self.kind.end_label
            ).apply(w.profile)
            self.sink.record_nourishment(
                w.profile.id,
                p.t_start,
                p.t_end,
                p.placed_m3,
                "FullNourishment",
                borrow_m3=p.borrow_m3,
            )
        w.recovered = True

    def run(self, order: list[_Work], works: _Works, resume: bool) -> CampaignCarryover | None:
        """Plan the placements, then play the plan: audit decisions are emitted in
        emission order, placements are applied, and every profile the crew never
        reached is recovered.  ``works`` is the full (non-inundated) set for that
        trailing sweep.
        """
        metrics = [
            ProfileDemand(w.profile.id, w.plan.volume_m3, w.plan.placement_m3, w.force)
            for w in order
        ]
        plan, campaign = plan_placements(
            metrics, self.t_storm, self.t_next, self.cfg.nourishment, resume
        )
        by_id = {w.profile.id: w for w in order}
        for d in plan:
            if isinstance(d, Placement):
                self._apply(d, by_id[d.profile_id])
            else:
                emit(self.sink, d.kind, d.t, profile_id=d.profile_id, **d.row())

        works.recover_unreached(self.t_storm, self.t_next, self.cfg, self.storm_at_next)
        return campaign


def _run_decided(
    works: _Works,
    t_base: float,
    t_next: float,
    cfg: ReachConfig,
    sink: ResultsSink,
    prior: CampaignCarryover | None,
    origin: CampaignKind,
    storm_at_next: bool = False,
) -> CampaignCarryover | None:
    """Assess → decide → execute, for either origin.

    The two origins share the whole pipeline; their differences ride on ``origin``
    (the NOURISH_CYCLE remap and the label pair) and on the works themselves (a
    scheduled cycle's items arrive pre-recovered, so its recover sweeps are no-ops).
    """
    works.assess(cfg)
    decision = decide_campaign(works.metrics, prior, cfg.nourishment, works.forced, origin, t_base)
    emit(sink, decision.kind, decision.t, **decision.row())
    if isinstance(decision, SkipCampaign):
        works.recover_unreached(t_base, t_next, cfg, storm_at_next)
        return None

    scheduler = _CampaignScheduler(
        t_storm=t_base,
        t_next=t_next,
        cfg=cfg,
        sink=sink,
        storm_at_next=storm_at_next,
        kind=origin,
    )
    return scheduler.run(works.in_order(list(decision.order)), works, decision.resume)


def run_campaign(
    outcomes: list[StormOutcome],
    t_storm: float,
    t_next: float,
    cfg: ReachConfig,
    sink: ResultsSink,
    longshore_widths: list[float] | None = None,
    prior: CampaignCarryover | None = None,
    storm_at_next: bool = False,
) -> tuple[float, CampaignCarryover | None]:
    """Run the post-storm campaign: recovery + (optionally) nourishment.

    Applies one Recovery event per profile at the appropriate time, then
    schedules nourishment sequentially by priority order.  ``storm_at_next`` marks
    ``t_next`` as a following storm (vs the sim/window end), so a recovery it cuts
    short before ``T_recover`` is labelled ``RECS`` instead of ``REC``.

    Inundated outcomes (``o.inundated`` — CSHORE failed) are skipped entirely.
    HACK (interim, group decision pending): the storm outcome is unknown for
    these, so Phase 3 is skipped for them — no recovery and no nourishment — and
    the profile carries forward unchanged.

    Returns (t_next, active_campaign):
        active_campaign is non-None only if the campaign was interrupted by
        the next storm or couldn't finish within [t_storm, t_next].
    """
    widths = longshore_widths or [1.0] * len(outcomes)
    works = _Works.build(outcomes, widths)

    # No nourishment configured: pure recovery
    if cfg.nourishment is None:
        works.recover_unreached(t_storm, t_next, cfg, storm_at_next)
        return t_next, None

    return t_next, _run_decided(
        works, t_storm, t_next, cfg, sink, prior, CampaignKind.STORM, storm_at_next
    )


def run_scheduled_campaign(
    profiles: list[Profile],
    t_cycle: float,
    t_next: float,
    cfg: ReachConfig,
    sink: ResultsSink,
    longshore_widths: list[float] | None = None,
    prior: CampaignCarryover | None = None,
) -> CampaignCarryover | None:
    """Run a periodic planned nourishment cycle in the quiet window ``[t_cycle, t_next]``.

    The calendar proposes and the volume gate disposes: every profile is assessed as in
    a post-storm campaign, but the reach mobilizes only if the deficit clears
    ``volume_trigger`` — so a cycle that finds a healthy beach places nothing.  Beyond
    that it is the same crew: same priority order, same blackout windows, same
    ``storm_conflict`` policy against the following storm.  The one difference is that
    there is no recovery to run (the caller only fires a cycle past the last recovery
    completion), so placements are ``SSN``/``ESN`` with no ``RECN`` ahead of them.

    Returns the carry-forward campaign if the crew was interrupted or blocked, else None.
    """
    if cfg.nourishment is None:
        return None

    widths = longshore_widths or [1.0] * len(profiles)
    works = _Works.build_scheduled(list(profiles), widths)
    return _run_decided(works, t_cycle, t_next, cfg, sink, prior, CampaignKind.SCHEDULED)
