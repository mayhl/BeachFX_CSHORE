from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .metrics import volume_above_datum
from .types import SnapshotLabel
from .units import ufloat

if TYPE_CHECKING:
    from .profile import Profile
    from .config import ReachConfig
    from .results import ResultsSink

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class NourishmentConfig(BaseModel):
    """Reach-level nourishment policy parameters."""

    # Template profile
    template_x: list[ufloat("m", "ft")]        # cross-shore positions (CSHORE convention, landward-positive)
    template_z: list[ufloat("m", "ft")]        # bed elevations (m NAVD internally)

    # Trigger + production
    volume_trigger: ufloat("m3", "cy")         # reach-level volume deficit (m³) to launch a campaign
    production_rate: ufloat("m3/day", "cy/yr") # dredge/pump output rate (m³/day)

    # Cost accounting
    cost_per_cy: float = 0.0                   # unit material cost ($/cy placed)
    mobilization_cost: float = 0.0             # fixed contractor mobilization cost ($)
    mobilization_threshold: float = 0.0        # minimum total campaign cost ($) to proceed
    mobilization_days: float = 0.0             # lead-time days before crew is on-site

    # Campaign scheduling
    strategy: Literal["equal_spacing", "zone_priority", "emergency"] = "equal_spacing"
    blackout_windows: list[tuple[float, float]] = Field(default_factory=list)  # (t_start_days, t_end_days)

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
    profile_id:     str
    volume_m3:      float           # volume deficit (m³/m × width)
    template_zb:    np.ndarray      # template interpolated onto profile grid
    priority_score: float           # higher = nourish first


@dataclass
class ActiveCampaign:
    """Carry-forward state when a nourishment campaign is interrupted by a storm."""
    crew_on_site:   bool
    priority_order: list[str]       # profile IDs in remaining campaign order
    placed:         dict[str, float] = field(default_factory=dict)  # volume placed so far (m³)


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
        v_now = _volume_above_datum(profile.x, profile.zb,   datum, width_m)
        v_tpl = _volume_above_datum(profile.x, template_zb,  datum, width_m)
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
        v_now = _volume_above_datum(profile.x, profile.zb,  datum, width_m)
        v_tpl = _volume_above_datum(profile.x, template_zb, datum, width_m)
        deficit = max(0.0, v_tpl - v_now)
        if deficit <= 0:
            return None

        # Dune score: deficit of bed above berm_elevation
        v_dune_now = _volume_above_datum(profile.x, profile.zb,  berm_elev, width_m)
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
    "zone_priority":  ZonePriorityStrategy(),
    "emergency":      EmergencyStrategy(),
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
            (bw_end for bw_start, bw_end in blackouts
             if t < bw_end and t + duration > bw_start),
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
) -> tuple[float, ActiveCampaign | None]:
    """Run the post-storm campaign: recovery + (optionally) nourishment.

    Applies one Recovery event per profile at the appropriate time, then
    schedules nourishment sequentially by priority order.

    Returns (t_next, active_campaign):
        active_campaign is non-None only if the campaign was interrupted by
        the next storm or couldn't finish within [t_storm, t_next].
    """
    from .profile import FullNourishment, PartialNourishment

    ncfg   = cfg.nourishment
    widths = longshore_widths or [1.0] * len(profiles)

    # Capture post-storm zb BEFORE any recovery is applied
    zb_post = [p.zb.copy() for p in profiles]

    # --- No nourishment configured: pure recovery ---
    if ncfg is None:
        for i, p in enumerate(profiles):
            _apply_recovery_one(p, zb_post[i], zb_pre_new[i], t_storm, t_next, cfg)
        return t_next, None

    # --- Build per-profile plans ---
    strategy = _STRATEGIES[ncfg.strategy]
    plans: dict[int, ProfileNourishmentPlan] = {}
    for i, p in enumerate(profiles):
        plan = strategy.plan(p, ncfg, widths[i])
        if plan is not None:
            plans[i] = plan

    # --- Mobilization gate (skip if crew already on site) ---
    crew_on_site = prior is not None and prior.crew_on_site
    if not crew_on_site:
        total_deficit = sum(pl.volume_m3 for pl in plans.values())
        if total_deficit < ncfg.volume_trigger:
            log.info(
                "Campaign at t=%.1fd: deficit %.0f m³ < trigger %.0f m³ — skipping",
                t_storm, total_deficit, ncfg.volume_trigger,
            )
            for i, p in enumerate(profiles):
                _apply_recovery_one(p, zb_post[i], zb_pre_new[i], t_storm, t_next, cfg)
            return t_next, None

    # --- Sort by priority (highest first) ---
    sorted_plans: list[tuple[int, ProfileNourishmentPlan]] = sorted(
        plans.items(), key=lambda x: -x[1].priority_score,
    )

    # --- If resuming a prior campaign, prepend remaining profiles in previous order ---
    if prior is not None:
        id_to_idx   = {p.id: i for i, p in enumerate(profiles)}
        prev_ids    = [pid for pid in prior.priority_order if pid in id_to_idx]
        prev_set    = set(prev_ids)
        rank        = {pid: pos for pos, pid in enumerate(prev_ids)}
        remaining   = sorted(
            [(i, pl) for i, pl in sorted_plans if profiles[i].id in prev_set],
            key=lambda x: rank.get(profiles[x[0]].id, 999),
        )
        new_ones    = [(i, pl) for i, pl in sorted_plans if profiles[i].id not in prev_set]
        sorted_plans = remaining + new_ones
        t_crew = t_storm   # crew already on site
    else:
        t_crew = t_storm + ncfg.mobilization_days

    recovery_done: dict[int, float] = {}   # idx → time through which recovery was applied
    active_campaign: ActiveCampaign | None = None

    for i, plan in sorted_plans:
        p        = profiles[i]
        duration = plan.volume_m3 / ncfg.production_rate   # days
        t_start  = _next_available(t_crew, duration, list(ncfg.blackout_windows))

        if t_start >= t_next:
            # Can't start before next storm — queue this and all subsequent profiles
            remaining_ids = [profiles[j].id for j, _ in sorted_plans
                             if j not in recovery_done]
            if active_campaign is None:
                active_campaign = ActiveCampaign(
                    crew_on_site=False,
                    priority_order=remaining_ids,
                )
            break

        # Recovery up to nourishment start
        _apply_recovery_one(p, zb_post[i], zb_pre_new[i], t_storm, t_start, cfg)
        recovery_done[i] = t_start

        # SSN: start of nourishment
        p.snapshot(SnapshotLabel.SSN, t_start)

        t_end = t_start + duration

        if t_end >= t_next:
            # Storm interrupts nourishment — partial placement
            fraction = (t_next - t_start) / duration
            PartialNourishment(
                t=t_next, template_zb=plan.template_zb, fraction=fraction,
            ).apply(p)
            sink.record_nourishment(p.id, t_start, t_next, plan.volume_m3 * fraction, "PartialNourishment")
            recovery_done[i] = t_next
            t_crew = t_next

            remaining_ids = [
                profiles[j].id for j, _ in sorted_plans
                if j not in recovery_done or j == i
            ]
            active_campaign = ActiveCampaign(
                crew_on_site=True,
                priority_order=remaining_ids,
                placed={profiles[i].id: plan.volume_m3 * fraction},
            )
            break
        else:
            FullNourishment(t=t_end, template_zb=plan.template_zb).apply(p)
            sink.record_nourishment(p.id, t_start, t_end, plan.volume_m3, "FullNourishment")
            recovery_done[i] = t_end
            t_crew = t_end

    # Apply recovery to all profiles not yet processed
    for i, p in enumerate(profiles):
        if i not in recovery_done:
            _apply_recovery_one(p, zb_post[i], zb_pre_new[i], t_storm, t_next, cfg)

    return t_next, active_campaign
