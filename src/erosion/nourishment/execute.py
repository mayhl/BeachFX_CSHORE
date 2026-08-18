"""Campaign execution: run-local work bundles, profile recovery, and the one
decide-then-execute path for both campaign origins."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..decision import Placement, SkipCampaign, emit
from ..decision.calendar import CampaignCarryover
from ..decision.model import ProfileDemand
from ..decision.planner import decide_campaign, plan_placements
from ..metrics import volume_above_datum
from ..profile import FullNourishment, PartialNourishment, Recovery
from ..storm import StormOutcome, _recovery_fraction
from ..types import CampaignKind
from .assess import _select_assessor

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..profile import Profile
    from ..results import ResultsSink

log = logging.getLogger(__name__)


@dataclass
class FillSpec:
    profile_id: str
    deficit_m3: float  # subaerial deficit (m³) — the trigger volume
    template_zb: np.ndarray  # template interpolated onto profile grid
    placement_m3: float = 0.0  # full active-height volume placed (borrow = ×ratio)


@dataclass
class WorkItem:
    """Per-profile run-local bundle for one campaign — bundles the profile with its
    captured post-storm bed, recovery target, width, plan, and recovered flag so the
    campaign loop iterates objects instead of parallel arrays indexed by position.
    """

    profile: Profile
    zb_post: np.ndarray
    zb_pre: np.ndarray  # zb_pre_new (recovery target on the fixed grid)
    width: float
    plan: FillSpec | None = None
    recovered: bool = False
    force: bool = False  # emergency trigger (volume or dune geometry) fired (Tier-1)


class Workset(list):
    """The campaign's active (non-inundated) work items — a ``list[WorkItem]``, one per
    profile, that owns the loops over that set.

    It *is* a list (so ``for w in works`` / ``len(works)`` / ``zip`` all work); it
    only adds named bulk steps (``assess``, ``recover_unreached``) so ``run_campaign``
    and ``CampaignExecutor`` compose them instead of re-opening a ``for w in active``
    each time — notably the recover-the-rest sweep, which the interval loop would
    otherwise repeat at every early return and after placement.
    """

    @classmethod
    def build(cls, outcomes: list[StormOutcome], widths: list[float]) -> Workset:
        """Bundle each non-inundated profile with its run-local campaign data.

        ``widths`` is aligned to ``outcomes`` (one per profile, including inundated
        ones), so the two zip positionally and inundated profiles drop out."""
        return cls(
            WorkItem(o.profile, o.profile.zb.copy(), o.zb_pre, w)
            for o, w in zip(outcomes, widths)
            if not o.inundated
        )

    @classmethod
    def build_planned(cls, profiles: list[Profile], widths: list[float]) -> Workset:
        """Bundle each profile for a periodic cycle — no storm, so no recovery.

        A planned cycle only ever fires in a quiet window (past the last recovery,
        clear of the next storm), so there is nothing to blend: every item starts
        ``recovered`` and the recover-the-rest sweep is a no-op.  The recovery beds
        are the current one, kept only so a ``WorkItem`` stays one shape.
        """
        return cls(
            WorkItem(p, p.zb.copy(), p.zb.copy(), w, recovered=True)
            for p, w in zip(profiles, widths)
        )

    def assess(self, cfg: ReachConfig) -> None:
        """Tier-1: assess each profile, attaching a plan when it needs fill or its
        dune tripped the emergency, and recording the force flag."""
        for w in self:
            assessor = _select_assessor(w.profile, cfg.nourishment)
            a = assessor.assess(w.profile, cfg, w.width)
            w.force = a.force
            if a.needs_fill or a.force:
                w.plan = FillSpec(
                    profile_id=w.profile.id,
                    deficit_m3=a.deficit_m3,
                    template_zb=assessor.restore_template(w.profile, cfg),
                    placement_m3=a.placement_m3,
                )

    @property
    def plans(self) -> list[WorkItem]:
        return [w for w in self if w.plan is not None]

    @property
    def metrics(self) -> list[ProfileDemand]:
        """The scalar boundary to Tier-2: one ``ProfileDemand`` per planned profile.
        The decider never sees a ``WorkItem`` — beds and templates stay on this side."""
        return [
            ProfileDemand(w.profile.id, w.plan.deficit_m3, w.plan.placement_m3, w.force)
            for w in self.plans
        ]

    def in_order(self, order: list[str]) -> list[WorkItem]:
        """Resolve the decider's ID order back onto the work items for placement."""
        by_id = {w.profile.id: w for w in self}
        return [by_id[pid] for pid in order]

    @property
    def forced(self) -> bool:
        return any(w.force for w in self)

    def recover_unreached(
        self, t_storm: float, t_apply: float, cfg: ReachConfig, storm_at_end: bool = False
    ) -> None:
        """Recover every profile the crew didn't nourish, up to ``t_apply``.

        ``storm_at_end`` marks ``t_apply`` as a following storm (vs the sim/window
        end), so a recovery it cuts short is labelled ``RECS`` rather than ``REC``.
        """
        for w in self:
            if not w.recovered:
                _recover_profile(w, t_storm, t_apply, cfg, storm_at_end)


def recovery_duration(profile: Profile, cfg: ReachConfig) -> float:
    """``T_recover`` for one profile — its own override, else the reach default."""
    if profile.geometry is not None and profile.geometry.recovery_duration is not None:
        return profile.geometry.recovery_duration
    return cfg.storm.T_recover


def _resolve_z_berm(profile: Profile, cfg: ReachConfig) -> float | None:
    """Below-berm mask elevation for recovery blending, per profile.

    An explicit reach-wide ``storm.z_berm`` wins; otherwise derive from the
    as-built berm (``ref_metrics.berm_elevation``), falling back to the design
    value (``geometry.berm_elevation``).  ``None`` (blend every node) survives
    only when no berm elevation is known at all — the blend then translates the
    dune landward with the shifted pre-storm target, so warn.
    """
    if cfg.storm.z_berm is not None:
        return float(cfg.storm.z_berm)
    ref = profile.ref_metrics
    if ref is not None and np.isfinite(ref.berm_elevation):
        return float(ref.berm_elevation)
    if profile.geometry is not None:
        return float(profile.geometry.berm_elevation)
    log.warning(
        "recovery: profile %s has no berm elevation (config, ref fit, or geometry); "
        "blending all nodes — the dune will track the shifted pre-storm bed",
        profile.id,
    )
    return None


def _apply_recovery_one(
    profile: Profile,
    zb_post: np.ndarray,
    zb_pre: np.ndarray,
    t_storm: float,
    t_apply: float,
    cfg: ReachConfig,
    storm_at_end: bool = False,
    nourish_at_end: bool = False,
) -> None:
    """Apply one Recovery event to profile, evaluated over ``[t_storm, t_apply]``.

    Recovery has a fixed duration ``T_recover`` (same for both models): it *ends*
    at ``t_storm + T_recover``, where the bed is frozen — linear reaches 100%,
    exponential is left at its T90 (90%), with no further creep.  So a recovery
    that reaches completion snapshots ``REC`` at that completion time.  It ends
    *earlier* only when cut short before ``T_recover``: by a following storm
    (``storm_at_end`` → ``RECS``) or by the crew arriving to nourish
    (``nourish_at_end`` → ``RECN``), stamped at ``t_apply`` with the partial
    fraction.  The two cut-short flags are set by disjoint call sites
    (recover-the-rest vs pre-placement), never both.
    """
    if t_apply <= t_storm:
        return

    T_recover = recovery_duration(profile, cfg)
    completion = t_storm + T_recover
    completed = t_apply >= completion
    # Freeze the fraction at the T_recover value once complete (exponential stops at
    # its T90 90% rather than creeping toward 1.0; linear is already capped at 1.0).
    eff_dt = T_recover if completed else (t_apply - t_storm)
    fraction = _recovery_fraction(eff_dt, T_recover, cfg.storm.recovery_model)
    if completed:
        # A completed recovery ends at its true completion time.  When background
        # erosion/SLC shares the gap, ticks and recovery are applied in phase order
        # (not time order), so stamping REC mid-gap there would mis-sequence it —
        # keep the gap-end stamp until the loop applies events in true time order.
        interstorm = cfg.erosion is not None or cfg.slc is not None
        t_rec = t_apply if interstorm else completion
        interrupted = False
    else:
        t_rec = t_apply
        interrupted = storm_at_end or nourish_at_end
    Recovery(
        t=t_rec,
        fraction=fraction,
        zb_post_storm=zb_post,
        zb_pre_storm=zb_pre,
        z_berm=_resolve_z_berm(profile, cfg),
        interrupted=interrupted,
        by_nourishment=(not completed) and nourish_at_end,
    ).apply(profile)


def _recover_profile(
    w: WorkItem,
    t_storm: float,
    t_apply: float,
    cfg: ReachConfig,
    storm_at_end: bool = False,
    nourish_at_end: bool = False,
) -> None:
    """Recover one work item's profile up to ``t_apply`` and mark it recovered."""
    _apply_recovery_one(
        w.profile, w.zb_post, w.zb_pre, t_storm, t_apply, cfg, storm_at_end, nourish_at_end
    )
    w.recovered = True


@dataclass
class CampaignExecutor:
    """Tier-3 executor: applies a planned campaign — recovery to each placement
    start, the fill itself, and the audit emission of the planner's decisions.

    All policy (blackout shift, storm-conflict defer/interrupt, crew clock) lives
    in ``erosion.decision.planner.plan_placements``; what remains here is bed
    mutation and bookkeeping.  ``kind`` selects the morphology label pair
    (``SEN``/``EEN`` post-storm, ``SSN``/``ESN`` for a planned cycle) and whether
    a placement is preceded by recovery — a planned cycle only fires in a quiet
    window, so it has none left to run.
    """

    t_base: float  # campaign start — storm end, or the cycle date for a planned cycle
    t_next: float
    cfg: ReachConfig
    sink: ResultsSink
    storm_at_next: bool = False  # t_next is a following storm (vs sim/window end) → RECS on cutoff
    kind: CampaignKind = CampaignKind.STORM

    def _apply(self, p: Placement, w: WorkItem) -> None:
        """Execute one planned placement on its work item."""
        # recovery up to the placement start; cut short here → RECN (crew, not storm).
        # A planned cycle has none to run — it only fires past the recovery completion.
        if self.kind is CampaignKind.STORM:
            _recover_profile(w, self.t_base, p.t_start, self.cfg, nourish_at_end=True)
        label = self.kind.start_label
        w.profile.snapshot(label, p.t_start)
        w.profile.record_event("NourishmentStart", p.t_start, label)

        if p.cut_by_storm:
            zb_before = w.profile.zb.copy()
            PartialNourishment(
                t=p.t_end,
                template_zb=w.plan.template_zb,
                fraction=p.fraction,
                label=self.kind.partial_label,
            ).apply(w.profile)
            billed_m3, borrow_m3 = p.placed_m3, p.borrow_m3
            ncfg = self.cfg.nourishment
            if ncfg is not None and ncfg.partial_billing == "beach_volume":
                # Bill the bed change the blend delivered instead of crew rate x
                # time -- the two differ by the assessed-volume vs template-gap
                # mismatch (see partial_billing in NourishmentConfig)
                dv = volume_above_datum(
                    w.profile.x, w.profile.zb, self.cfg.msl
                ) - volume_above_datum(w.profile.x, zb_before, self.cfg.msl)
                billed_m3 = max(dv, 0.0) * w.width
                borrow_m3 = billed_m3 * ncfg.borrow_to_placement_ratio
            self.sink.record_nourishment(
                w.profile.id,
                p.t_start,
                p.t_end,
                billed_m3,
                "PartialNourishment",
                borrow_m3=borrow_m3,
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

    def run(self, order: list[WorkItem], works: Workset, resume: bool) -> CampaignCarryover | None:
        """Plan the placements, then play the plan: audit decisions are emitted in
        emission order, placements are applied, and every profile the crew never
        reached is recovered.  ``works`` is the full (non-inundated) set for that
        trailing sweep.
        """
        metrics = [
            ProfileDemand(w.profile.id, w.plan.deficit_m3, w.plan.placement_m3, w.force)
            for w in order
        ]
        plan, campaign = plan_placements(
            metrics, self.t_base, self.t_next, self.cfg.nourishment, resume
        )
        by_id = {w.profile.id: w for w in order}
        for d in plan:
            if isinstance(d, Placement):
                self._apply(d, by_id[d.profile_id])
            else:
                emit(self.sink, d.kind, d.t, profile_id=d.profile_id, **d.row())

        works.recover_unreached(self.t_base, self.t_next, self.cfg, self.storm_at_next)
        return campaign


def _run_decided(
    works: Workset,
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
    planned cycle's items arrive pre-recovered, so its recover sweeps are no-ops).
    """
    works.assess(cfg)
    decision = decide_campaign(works.metrics, prior, cfg.nourishment, works.forced, origin, t_base)
    emit(sink, decision.kind, decision.t, **decision.row())
    if isinstance(decision, SkipCampaign):
        works.recover_unreached(t_base, t_next, cfg, storm_at_next)
        return None

    scheduler = CampaignExecutor(
        t_base=t_base,
        t_next=t_next,
        cfg=cfg,
        sink=sink,
        storm_at_next=storm_at_next,
        kind=origin,
    )
    return scheduler.run(works.in_order(list(decision.order)), works, decision.resume)


def run_campaign(
    outcomes: list[StormOutcome],
    t_base: float,
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
        the next storm or couldn't finish within [t_base, t_next].
    """
    widths = longshore_widths or [1.0] * len(outcomes)
    works = Workset.build(outcomes, widths)

    # No nourishment configured: pure recovery
    if cfg.nourishment is None:
        works.recover_unreached(t_base, t_next, cfg, storm_at_next)
        return t_next, None

    return t_next, _run_decided(
        works, t_base, t_next, cfg, sink, prior, CampaignKind.STORM, storm_at_next
    )


def run_planned_campaign(
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
    works = Workset.build_planned(list(profiles), widths)
    return _run_decided(works, t_cycle, t_next, cfg, sink, prior, CampaignKind.PLANNED)
