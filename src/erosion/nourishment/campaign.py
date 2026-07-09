"""Run-local campaign state: the nourishment plan, per-profile work items, the
active-campaign carry-forward collection, and profile recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..profile import Recovery
from ..storm import StormOutcome, _recovery_fraction
from .assess import _select_assessor

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..profile import Profile


@dataclass
class ProfileNourishmentPlan:
    profile_id: str
    volume_m3: float  # subaerial deficit (m³) — the trigger volume
    template_zb: np.ndarray  # template interpolated onto profile grid
    priority_score: float  # higher = nourish first
    placement_m3: float = 0.0  # full active-height volume placed (borrow = ×ratio)


@dataclass
class ActiveCampaign:
    """Carry-forward state when a nourishment campaign is interrupted by a storm."""

    crew_on_site: bool
    priority_order: list[str]  # profile IDs in remaining campaign order


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
                    priority_score=a.volume_m3,  # equal-spacing: priority = deficit
                    placement_m3=a.placement_m3,
                )

    @property
    def plans(self) -> list[_Work]:
        return [w for w in self if w.plan is not None]

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


class _Outcome(Enum):
    """Result of placing one profile within a campaign window."""

    COMPLETED = "completed"  # placed fully, crew moves on
    BLOCKED = "blocked"  # couldn't start before the next storm
    INTERRUPTED = "interrupted"  # next storm cut it mid-placement


def _apply_recovery_one(
    profile: Profile,
    zb_post: np.ndarray,
    zb_pre: np.ndarray,
    t_storm: float,
    t_apply: float,
    cfg: ReachConfig,
    storm_at_end: bool = False,
) -> None:
    """Apply one Recovery event to profile evaluated at t_apply.

    When ``storm_at_end`` (``t_apply`` is a following storm) and less than the full
    ``T_recover`` has elapsed, the storm forced the recovery short → ``RECS``.  A
    recovery that ran its full period, or ended at the sim/window boundary, → ``REC``.
    """
    if t_apply <= t_storm:
        return

    T_recover = (
        profile.geometry.recovery_duration
        if profile.geometry and profile.geometry.recovery_duration is not None
        else cfg.storm.T_recover
    )
    dt = t_apply - t_storm
    fraction = _recovery_fraction(dt, T_recover, cfg.storm.recovery_model)
    Recovery(
        t=t_apply,
        fraction=fraction,
        zb_post_storm=zb_post,
        zb_pre_storm=zb_pre,
        z_berm=cfg.storm.z_berm,
        interrupted=storm_at_end and dt < T_recover,
    ).apply(profile)


def _recover_profile(
    w: _Work, t_storm: float, t_apply: float, cfg: ReachConfig, storm_at_end: bool = False
) -> None:
    """Recover one work item's profile up to ``t_apply`` and mark it recovered."""
    _apply_recovery_one(w.profile, w.zb_post, w.zb_pre, t_storm, t_apply, cfg, storm_at_end)
    w.recovered = True
