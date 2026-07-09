"""Tier-3 serial placement scheduler and the run_campaign entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..profile import FullNourishment, PartialNourishment
from ..types import DecisionKind, SnapshotLabel
from .campaign import (
    ActiveCampaign,
    _Outcome,
    _recover_profile,
    _Work,
    _Works,
)
from .decide import _DEFAULT_DECIDER

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..results import ResultsSink
    from ..storm import StormOutcome

log = logging.getLogger(__name__)


def _next_available(
    t: float,
    duration: float,
    blackouts: list[tuple[float, float]],
) -> float:
    """Return earliest start ≥ t such that [start, start+duration) doesn't overlap any blackout."""
    while True:
        blocking = next(
            (bw_end for bw_start, bw_end in blackouts if t < bw_end and t + duration > bw_start),
            None,
        )
        if blocking is None:
            return t
        t = blocking


def _log_decision(sink: ResultsSink, kind: DecisionKind, t: float, **payload) -> None:
    """Record a decision to the structured audit AND the operational console."""
    sink.record_decision(kind, t, **payload)
    log.info("campaign t=%.1fd — %s %s", t, kind.value, payload)


def _remaining(order: list[_Work], interrupted: _Work | None = None) -> list[str]:
    """Profile IDs still owed placement, interrupted one first (for carry-forward)."""
    ids = [w.profile.id for w in order if not w.recovered]
    if interrupted is not None and interrupted.profile.id not in ids:
        ids.insert(0, interrupted.profile.id)
    return ids


@dataclass
class _CampaignScheduler:
    """Tier-3: a serial crew placing nourishment plans within ``[t_storm, t_next]``.

    Owns the crew clock (``t_crew``) and emits the placement events — full/partial
    nourishment, plus BLACKOUT_DEFER / STORM_DEFER / INTERRUPT decisions — so
    ``run_campaign`` stays a flat assess->decide->place->recover pipeline.
    """

    t_storm: float
    t_next: float
    cfg: ReachConfig
    sink: ResultsSink
    t_crew: float
    storm_at_next: bool = False  # t_next is a following storm (vs sim/window end) → RECS on cutoff

    def _place(self, w: _Work) -> _Outcome:
        """Schedule the placement, then dispatch: recover to the start and place
        full, or partial/deferred on storm conflict.  Advances ``self.t_crew`` and
        returns the placement outcome.
        """
        ncfg = self.cfg.nourishment
        borrow = w.plan.placement_m3 * ncfg.borrow_to_placement_ratio
        duration = borrow / ncfg.production_rate  # days — dredged borrow, not restored geometry
        t_start = _next_available(self.t_crew, duration, list(ncfg.blackout_windows))
        if t_start > self.t_crew:
            _log_decision(
                self.sink,
                DecisionKind.BLACKOUT_DEFER,
                self.t_storm,
                profile_id=w.profile.id,
                requested=self.t_crew,
                deferred_to=t_start,
            )
        if t_start >= self.t_next:  # can't start before the next storm
            return _Outcome.BLOCKED

        t_end = t_start + duration
        storm_conflict = t_end >= self.t_next  # next storm would land during placement

        # DEFER policy (BeachFX): a placement the storm would hit is not started at all —
        # delay it to a storm-free window (the campaign carries to the next gap). No SSN,
        # no partial fill; the profile just recovers over this gap.
        if storm_conflict and ncfg.storm_conflict == "defer":
            _log_decision(
                self.sink,
                DecisionKind.STORM_DEFER,
                self.t_storm,
                profile_id=w.profile.id,
                would_start=t_start,
                would_end=t_end,
                storm=self.t_next,
            )
            return _Outcome.BLOCKED

        _recover_profile(w, self.t_storm, t_start, self.cfg)  # recovery up to the start
        w.profile.snapshot(SnapshotLabel.SSN, t_start)

        if storm_conflict:  # INTERRUPT policy: place what fits, resume after the storm
            return self._place_partial(w, t_start, duration, borrow)
        return self._place_full(w, t_start, t_end, borrow)

    def _place_partial(self, w: _Work, t_start: float, duration: float, borrow: float) -> _Outcome:
        """INTERRUPT policy: place what fits before the next storm, resume after."""
        fraction = (self.t_next - t_start) / duration
        placed = w.plan.placement_m3 * fraction  # geometry-effective volume on the beach
        _log_decision(
            self.sink,
            DecisionKind.INTERRUPT,
            self.t_next,
            profile_id=w.profile.id,
            placed_fraction=fraction,
        )
        PartialNourishment(t=self.t_next, template_zb=w.plan.template_zb, fraction=fraction).apply(
            w.profile
        )
        self.sink.record_nourishment(
            w.profile.id,
            t_start,
            self.t_next,
            placed,
            "PartialNourishment",
            borrow_m3=borrow * fraction,
        )
        w.recovered = True
        self.t_crew = self.t_next
        return _Outcome.INTERRUPTED

    def _place_full(self, w: _Work, t_start: float, t_end: float, borrow: float) -> _Outcome:
        """Full placement completes before the next storm."""
        FullNourishment(t=t_end, template_zb=w.plan.template_zb).apply(w.profile)
        self.sink.record_nourishment(
            w.profile.id, t_start, t_end, w.plan.placement_m3, "FullNourishment", borrow_m3=borrow
        )
        w.recovered = True
        self.t_crew = t_end
        return _Outcome.COMPLETED

    def run(self, order: list[_Work], works: _Works) -> ActiveCampaign | None:
        """Place each plan in ``order`` serially; on the first block/interrupt, stop
        and return the carry-forward campaign.  Recovers any profile the crew never
        reached.  ``works`` is the full (non-inundated) set for that trailing sweep.
        """
        campaign: ActiveCampaign | None = None
        for w in order:
            outcome = self._place(w)
            if outcome is _Outcome.BLOCKED:
                campaign = ActiveCampaign(crew_on_site=False, priority_order=_remaining(order))
                break
            if outcome is _Outcome.INTERRUPTED:
                campaign = ActiveCampaign(
                    crew_on_site=True,
                    priority_order=_remaining(order, interrupted=w),
                )
                break

        works.recover_unreached(self.t_storm, self.t_next, self.cfg, self.storm_at_next)
        return campaign


def run_campaign(
    outcomes: list[StormOutcome],
    t_storm: float,
    t_next: float,
    cfg: ReachConfig,
    sink: ResultsSink,
    longshore_widths: list[float] | None = None,
    prior: ActiveCampaign | None = None,
    storm_at_next: bool = False,
) -> tuple[float, ActiveCampaign | None]:
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
    ncfg = cfg.nourishment
    widths = longshore_widths or [1.0] * len(outcomes)
    works = _Works.build(outcomes, widths)

    # --- No nourishment configured: pure recovery ---
    if ncfg is None:
        works.recover_unreached(t_storm, t_next, cfg, storm_at_next)
        return t_next, None

    # --- Per-profile assessment (Tier 1) → plans ---
    works.assess(cfg)

    # --- Reach decision (Tier 2): trigger gate + placement order ---
    decision = _DEFAULT_DECIDER.decide(works.plans, prior, ncfg, forced=works.forced)
    if not decision.mobilize:
        _log_decision(
            sink,
            decision.kind,
            t_storm,
            deficit=decision.total_deficit,
            trigger=float(ncfg.volume_trigger) if ncfg.volume_trigger is not None else None,
        )
        works.recover_unreached(t_storm, t_next, cfg, storm_at_next)
        return t_next, None
    _log_decision(
        sink,
        decision.kind,
        t_storm,
        deficit=decision.total_deficit,
        resume=decision.resume,
        forced=decision.forced,
    )

    # --- Serial placement (Tier 3): crew works the priority order, recovers the rest ---
    t_crew = t_storm if decision.resume else t_storm + ncfg.mobilization_days
    scheduler = _CampaignScheduler(
        t_storm=t_storm,
        t_next=t_next,
        cfg=cfg,
        sink=sink,
        t_crew=t_crew,
        storm_at_next=storm_at_next,
    )
    return t_next, scheduler.run(decision.order, works)
