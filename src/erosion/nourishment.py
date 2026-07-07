from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .metrics import volume_above_datum
from .types import DecisionKind, SnapshotLabel
from .units import ufloat

if TYPE_CHECKING:
    from .config import ReachConfig
    from .profile import Profile
    from .results import ResultsSink

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class NourishmentConfig(BaseModel):
    """Reach-level nourishment policy parameters."""

    # Template profile
    template_x: list[
        ufloat("m", "ft")
    ]  # cross-shore positions (CSHORE convention, landward-positive)
    template_z: list[ufloat("m", "ft")]  # bed elevations (m NAVD internally)

    # Trigger + production
    volume_trigger: ufloat("m3", "cy")  # reach-level volume deficit (m³) to launch a campaign
    production_rate: ufloat("m3/day", "cy/yr")  # dredge/pump output rate (m³/day)

    # Cost accounting
    cost_per_cy: float = 0.0  # unit material cost ($/cy placed)
    mobilization_cost: float = 0.0  # fixed contractor mobilization cost ($)
    mobilization_threshold: float = 0.0  # minimum total campaign cost ($) to proceed
    mobilization_days: float = 0.0  # lead-time days before crew is on-site

    # Campaign scheduling
    strategy: Literal["equal_spacing", "zone_priority", "emergency"] = "equal_spacing"
    blackout_windows: list[tuple[float, float]] = Field(
        default_factory=list
    )  # (t_start_days, t_end_days)

    # Output tagging
    alternative_id: str = "FWP"

    @property
    def mobilization_volume_cy(self) -> float:
        """Minimum deficit (cy) for material + mobilization cost to meet threshold."""
        if self.cost_per_cy <= 0:
            return 0.0
        net = self.mobilization_threshold - self.mobilization_cost
        return max(0.0, net / self.cost_per_cy)

    @model_validator(mode="after")
    def _check_positive_rates(self) -> NourishmentConfig:
        if self.volume_trigger <= 0:
            raise ValueError("volume_trigger must be positive")
        if self.production_rate <= 0:
            raise ValueError("production_rate must be positive")
        return self


# ---------------------------------------------------------------------------
# Nourishment plan + campaign state
# ---------------------------------------------------------------------------


@dataclass
class ProfileNourishmentPlan:
    profile_id: str
    volume_m3: float  # volume deficit (m³/m × width)
    template_zb: np.ndarray  # template interpolated onto profile grid
    priority_score: float  # higher = nourish first


@dataclass
class ActiveCampaign:
    """Carry-forward state when a nourishment campaign is interrupted by a storm."""

    crew_on_site: bool
    priority_order: list[str]  # profile IDs in remaining campaign order
    placed: dict[str, float] = field(default_factory=dict)  # volume placed so far (m³)


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


class _Outcome(Enum):
    """Result of placing one profile within a campaign window."""

    COMPLETED = "completed"  # placed fully, crew moves on
    BLOCKED = "blocked"  # couldn't start before the next storm
    INTERRUPTED = "interrupted"  # next storm cut it mid-placement


# ---------------------------------------------------------------------------
# Nourishment strategies
# ---------------------------------------------------------------------------


class NourishmentStrategy(ABC):
    @abstractmethod
    def plan(
        self,
        profile: Profile,
        ncfg: NourishmentConfig,
        width_m: float,
    ) -> ProfileNourishmentPlan | None: ...


def _interp_template(profile: Profile, ncfg: NourishmentConfig) -> np.ndarray:
    tx = np.asarray(ncfg.template_x)
    tz = np.asarray(ncfg.template_z)
    return np.interp(profile.x, tx, tz, left=float(tz[0]), right=float(tz[-1]))


def _volume_above_datum(x: np.ndarray, z: np.ndarray, datum: float, width_m: float) -> float:
    return volume_above_datum(x, z, datum) * width_m


class EqualSpacingStrategy(NourishmentStrategy):
    """All profiles receive the full template; priority by volume deficit."""

    def plan(self, profile, ncfg, width_m):
        template_zb = _interp_template(profile, ncfg)
        datum = profile.geometry.datum if profile.geometry else 0.0
        v_now = _volume_above_datum(profile.x, profile.zb, datum, width_m)
        v_tpl = _volume_above_datum(profile.x, template_zb, datum, width_m)
        deficit = max(0.0, v_tpl - v_now)
        if deficit <= 0:
            return None
        return ProfileNourishmentPlan(
            profile_id=profile.id,
            volume_m3=deficit,
            template_zb=template_zb,
            priority_score=deficit,
        )


class ZonePriorityStrategy(NourishmentStrategy):
    """Prioritise by dune deficit over berm deficit over nearshore deficit."""

    def plan(self, profile, ncfg, width_m):
        template_zb = _interp_template(profile, ncfg)
        datum = profile.geometry.datum if profile.geometry else 0.0
        berm_elev = profile.geometry.berm_elevation if profile.geometry else 0.0
        v_now = _volume_above_datum(profile.x, profile.zb, datum, width_m)
        v_tpl = _volume_above_datum(profile.x, template_zb, datum, width_m)
        deficit = max(0.0, v_tpl - v_now)
        if deficit <= 0:
            return None

        # Dune score: deficit of bed above berm_elevation
        v_dune_now = _volume_above_datum(profile.x, profile.zb, berm_elev, width_m)
        v_dune_tpl = _volume_above_datum(profile.x, template_zb, berm_elev, width_m)
        dune_deficit = max(0.0, v_dune_tpl - v_dune_now)

        # Dune deficit dominates priority; overall deficit is tiebreaker
        priority_score = dune_deficit * 1e6 + deficit

        return ProfileNourishmentPlan(
            profile_id=profile.id,
            volume_m3=deficit,
            template_zb=template_zb,
            priority_score=priority_score,
        )


class EmergencyStrategy(NourishmentStrategy):
    """Emergency: prioritise profiles with dune loss regardless of berm width."""

    def plan(self, profile, ncfg, width_m):
        # Same placement as EqualSpacing but doubled priority score to
        # signal urgency; caller can layer additional triggers on top.
        base = EqualSpacingStrategy().plan(profile, ncfg, width_m)
        if base is None:
            return None
        return ProfileNourishmentPlan(
            profile_id=base.profile_id,
            volume_m3=base.volume_m3,
            template_zb=base.template_zb,
            priority_score=base.priority_score * 2.0,
        )


_STRATEGIES: dict[str, NourishmentStrategy] = {
    "equal_spacing": EqualSpacingStrategy(),
    "zone_priority": ZonePriorityStrategy(),
    "emergency": EmergencyStrategy(),
}


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


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


def _apply_recovery_one(
    profile: Profile,
    zb_post: np.ndarray,
    zb_pre: np.ndarray,
    t_storm: float,
    t_apply: float,
    cfg: ReachConfig,
) -> None:
    """Apply one Recovery event to profile evaluated at t_apply."""
    from .profile import Recovery
    from .storm import _recovery_fraction

    if t_apply <= t_storm:
        return

    T_recover = (
        profile.geometry.recovery_duration
        if profile.geometry and profile.geometry.recovery_duration is not None
        else cfg.storm.T_recover
    )
    fraction = _recovery_fraction(t_apply - t_storm, T_recover, cfg.storm.recovery_model)
    Recovery(
        t=t_apply,
        fraction=fraction,
        zb_post_storm=zb_post,
        zb_pre_storm=zb_pre,
        z_berm=cfg.storm.z_berm,
    ).apply(profile)


# ---------------------------------------------------------------------------
# Phase 3 runner
# ---------------------------------------------------------------------------


def run_campaign(
    profiles: list[Profile],
    zb_pre_new: list[np.ndarray],
    t_storm: float,
    t_next: float,
    cfg: ReachConfig,
    sink: ResultsSink,
    longshore_widths: list[float] | None = None,
    prior: ActiveCampaign | None = None,
    inundated: set[str] | None = None,
) -> tuple[float, ActiveCampaign | None]:
    """Run the post-storm campaign: recovery + (optionally) nourishment.

    Applies one Recovery event per profile at the appropriate time, then
    schedules nourishment sequentially by priority order.

    ``inundated`` is the set of profile IDs whose storm response was INUNDATION
    (CSHORE failed).  HACK (interim, group decision pending): the storm outcome
    is unknown for these, so we skip Phase 3 entirely for them — no recovery and
    no nourishment — and let the profile carry forward unchanged.

    Returns (t_next, active_campaign):
        active_campaign is non-None only if the campaign was interrupted by
        the next storm or couldn't finish within [t_storm, t_next].
    """
    from .profile import FullNourishment, PartialNourishment

    ncfg = cfg.nourishment
    widths = longshore_widths or [1.0] * len(profiles)
    skip = inundated or set()

    def _decide(kind: DecisionKind, t: float, **payload) -> None:
        """Log (operational console) AND record (structured audit) a decision."""
        sink.record_decision(kind, t, **payload)
        log.info("campaign t=%.1fd — %s %s", t, kind.value, payload)

    # Active (non-inundated) work items — bundle each profile's run-local data so the
    # rest of the campaign iterates objects, not parallel arrays indexed by position.
    active = [
        _Work(p, p.zb.copy(), zb_pre_new[i], widths[i])
        for i, p in enumerate(profiles)
        if p.id not in skip
    ]

    def _recover(w: _Work, t_apply: float) -> None:
        _apply_recovery_one(w.profile, w.zb_post, w.zb_pre, t_storm, t_apply, cfg)
        w.recovered = True

    # --- No nourishment configured: pure recovery ---
    if ncfg is None:
        for w in active:
            _recover(w, t_next)
        return t_next, None

    # --- Per-profile plans, then the reach-wide trigger gate ---
    strategy = _STRATEGIES[ncfg.strategy]
    for w in active:
        w.plan = strategy.plan(w.profile, ncfg, w.width)
    plans = [w for w in active if w.plan is not None]
    total_deficit = sum(w.plan.volume_m3 for w in plans)

    crew_on_site = prior is not None and prior.crew_on_site
    if not crew_on_site and total_deficit < ncfg.volume_trigger:
        _decide(
            DecisionKind.NOURISH_SKIP,
            t_storm,
            deficit=total_deficit,
            trigger=float(ncfg.volume_trigger),
        )
        for w in active:
            _recover(w, t_next)
        return t_next, None

    _decide(DecisionKind.NOURISH_TRIGGER, t_storm, deficit=total_deficit, resume=crew_on_site)

    # --- Order: priority desc; on resume, prior-campaign profiles first (stable sort) ---
    plans.sort(key=lambda w: -w.plan.priority_score)
    if prior is not None:
        rank = {pid: pos for pos, pid in enumerate(prior.priority_order)}
        plans.sort(key=lambda w: rank.get(w.profile.id, 999))  # stable → priority kept per rank
        t_crew = t_storm  # crew already on site
    else:
        t_crew = t_storm + ncfg.mobilization_days

    active_campaign: ActiveCampaign | None = None

    def _remaining(interrupted: _Work | None = None) -> list[str]:
        ids = [w.profile.id for w in plans if not w.recovered]
        if interrupted is not None and interrupted.profile.id not in ids:
            ids.insert(0, interrupted.profile.id)
        return ids

    def _place(w: _Work, t_crew: float) -> tuple[float, _Outcome, float]:
        """Recover up to the start, then place (full, or partial on interrupt).

        Returns (new t_crew, outcome, volume placed). Emits BLACKOUT_DEFER/INTERRUPT.
        """
        duration = w.plan.volume_m3 / ncfg.production_rate  # days
        t_start = _next_available(t_crew, duration, list(ncfg.blackout_windows))
        if t_start > t_crew:
            _decide(
                DecisionKind.BLACKOUT_DEFER,
                t_storm,
                profile_id=w.profile.id,
                requested=t_crew,
                deferred_to=t_start,
            )
        if t_start >= t_next:  # can't start before the next storm
            return t_crew, _Outcome.BLOCKED, 0.0

        _recover(w, t_start)  # recovery up to the nourishment start
        w.profile.snapshot(SnapshotLabel.SSN, t_start)
        t_end = t_start + duration

        if t_end >= t_next:  # storm interrupts — partial placement
            fraction = (t_next - t_start) / duration
            placed = w.plan.volume_m3 * fraction
            _decide(
                DecisionKind.INTERRUPT, t_next, profile_id=w.profile.id, placed_fraction=fraction
            )
            PartialNourishment(
                t=t_next, template_zb=w.plan.template_zb, fraction=fraction
            ).apply(w.profile)
            sink.record_nourishment(w.profile.id, t_start, t_next, placed, "PartialNourishment")
            w.recovered = True
            return t_next, _Outcome.INTERRUPTED, placed

        FullNourishment(t=t_end, template_zb=w.plan.template_zb).apply(w.profile)
        sink.record_nourishment(w.profile.id, t_start, t_end, w.plan.volume_m3, "FullNourishment")
        w.recovered = True
        return t_end, _Outcome.COMPLETED, w.plan.volume_m3

    for w in plans:
        t_crew, outcome, placed = _place(w, t_crew)
        if outcome is _Outcome.BLOCKED:
            active_campaign = ActiveCampaign(crew_on_site=False, priority_order=_remaining())
            break
        if outcome is _Outcome.INTERRUPTED:
            active_campaign = ActiveCampaign(
                crew_on_site=True,
                priority_order=_remaining(interrupted=w),
                placed={w.profile.id: placed},
            )
            break

    # Recovery for any active profile the crew didn't reach
    for w in active:
        if not w.recovered:
            _recover(w, t_next)

    return t_next, active_campaign
