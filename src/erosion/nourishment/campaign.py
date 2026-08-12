"""Run-local campaign state: the nourishment plan, per-profile work items, the
active-campaign carry-forward collection, and profile recovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..decision.model import PlanMetrics
from ..profile import Recovery
from ..storm import StormOutcome, _recovery_fraction
from .assess import _select_assessor

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..profile import Profile

log = logging.getLogger(__name__)


@dataclass
class ProfileNourishmentPlan:
    profile_id: str
    volume_m3: float  # subaerial deficit (m³) — the trigger volume
    template_zb: np.ndarray  # template interpolated onto profile grid
    placement_m3: float = 0.0  # full active-height volume placed (borrow = ×ratio)


@dataclass
class _Work:
    """Per-profile run-local bundle for one campaign — bundles the profile with its
    captured post-storm bed, recovery target, width, plan, and recovered flag so the
    campaign loop iterates objects instead of parallel arrays indexed by position.
    """

    profile: Profile
    zb_post: np.ndarray
    zb_pre: np.ndarray  # zb_pre_new (recovery target on the fixed grid)
    width: float
    plan: ProfileNourishmentPlan | None = None
    recovered: bool = False
    force: bool = False  # emergency geometric trigger fired (Tier-1)


class _Works(list):
    """The campaign's active (non-inundated) work items — a ``list[_Work]``, one per
    profile, that owns the loops over that set.

    It *is* a list (so ``for w in works`` / ``len(works)`` / ``zip`` all work); it
    only adds named bulk steps (``assess``, ``recover_unreached``) so ``run_campaign``
    and ``_CampaignScheduler`` compose them instead of re-opening a ``for w in active``
    each time — notably the recover-the-rest sweep, which the orchestrator would
    otherwise repeat at every early return and after placement.
    """

    @classmethod
    def build(cls, outcomes: list[StormOutcome], widths: list[float]) -> _Works:
        """Bundle each non-inundated profile with its run-local campaign data.

        ``widths`` is aligned to ``outcomes`` (one per profile, including inundated
        ones), so the two zip positionally and inundated profiles drop out."""
        return cls(
            _Work(o.profile, o.profile.zb.copy(), o.zb_pre, w)
            for o, w in zip(outcomes, widths)
            if not o.inundated
        )

    @classmethod
    def build_scheduled(cls, profiles: list[Profile], widths: list[float]) -> _Works:
        """Bundle each profile for a periodic cycle — no storm, so no recovery.

        A planned cycle only ever fires in a quiet window (past the last recovery,
        clear of the next storm), so there is nothing to blend: every item starts
        ``recovered`` and the recover-the-rest sweep is a no-op.  The recovery beds
        are the current one, kept only so a ``_Work`` stays one shape.
        """
        return cls(
            _Work(p, p.zb.copy(), p.zb.copy(), w, recovered=True) for p, w in zip(profiles, widths)
        )

    def assess(self, cfg: ReachConfig) -> None:
        """Tier-1: assess each profile, attaching a plan when it needs fill or its
        dune tripped the emergency, and recording the force flag."""
        for w in self:
            assessor = _select_assessor(w.profile, cfg.nourishment)
            a = assessor.assess(w.profile, cfg, w.width)
            w.force = a.force
            if a.needs_fill or a.force:
                w.plan = ProfileNourishmentPlan(
                    profile_id=w.profile.id,
                    volume_m3=a.volume_m3,
                    template_zb=assessor.restore_template(w.profile, cfg),
                    placement_m3=a.placement_m3,
                )

    @property
    def plans(self) -> list[_Work]:
        return [w for w in self if w.plan is not None]

    @property
    def metrics(self) -> list[PlanMetrics]:
        """The scalar boundary to Tier-2: one ``PlanMetrics`` per planned profile.
        The decider never sees a ``_Work`` — beds and templates stay on this side."""
        return [
            PlanMetrics(w.profile.id, w.plan.volume_m3, w.plan.placement_m3, w.force)
            for w in self.plans
        ]

    def in_order(self, order: list[str]) -> list[_Work]:
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
    w: _Work,
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
