"""Reach-level nourishment configuration models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..units import ufloat


class GeometryThresholds(BaseModel):
    """Restore-target geometry for the synthesized template, and the emergency
    trigger thresholds (Phase 4).  Each field is a target; ``None`` falls back to
    the measured as-built value (``ref_metrics``) — the default->override idiom.
    BeachFX-grid numeric defaults are populated per reach in config; unset here
    means restore-to-as-built.

    As the sole template definition (the ``template_x/template_z`` array is
    retired), these parametric metrics drive the whole restore shape; ``berm_height``
    and ``foreshore_slope`` fix the beach-face wedge, the dune fields the dune. The
    template is synthesized on the profile grid — already CSHORE-frame (landward-
    positive, x=0 offshore) — so there is no coordinate array to mis-orient.
    """

    model_config = ConfigDict(extra="forbid")

    berm_width: ufloat("m", "ft") | None = None
    berm_height: ufloat("m", "ft") | None = None  # BE: berm crest elevation above datum
    foreshore_slope: float | None = None  # beach-face |dz/dx| (m/m)
    dune_height: ufloat("m", "ft") | None = None  # crest above berm (front relief)
    dune_width: ufloat("m", "ft") | None = None
    # Synthesized dune crest shape (mirrors the fitter's three forms): a sharp
    # ``triangle`` apex, a flat-topped ``trapezoid`` plateau, or a rounded
    # ``gaussian`` bell.  ``None`` auto-selects: trapezoid when the idealized dune
    # has a measured plateau (``ref.dune_top_width > 0``), else triangle.
    dune_form: Literal["triangle", "trapezoid", "gaussian"] | None = None


class NourishmentConfig(BaseModel):
    """Reach-level nourishment policy parameters."""

    model_config = ConfigDict(extra="forbid")

    # Template profile — parametric restore geometry, synthesized on the profile
    # grid (CSHORE-frame by construction).  The former ``template_x/template_z``
    # coordinate array is retired: it lived in a foreign frame that had to be
    # declared per-config and was one data-entry slip from being placed backwards.
    template_geometry: GeometryThresholds = Field(default_factory=GeometryThresholds)

    # Trigger + production
    # Regular (scheduled) trigger: reach-level volume deficit (m³) that launches a
    # campaign.  Optional — omit it for an emergency-only reach (the ≥1-active-trigger
    # validator then requires emergency_volume or emergency_geometry instead).
    volume_trigger: ufloat("m3", "cy") | None = None
    production_rate: ufloat("m3/day", "cy/yr")  # dredge/pump output rate (m³/day)
    # Borrow = placement × ratio (cReach.cpp): extra material lost in placement.
    # Drives duration and cost off the borrow volume, not the restored geometry.
    # Default 1.0 (BeachFX default since 4/8/2015); emergency EN is always 1.0.
    borrow_to_placement_ratio: float = 1.0

    # Cost accounting
    cost_per_cy: float = 0.0  # unit material cost ($/cy placed)
    mobilization_cost: float = 0.0  # fixed contractor mobilization cost ($)
    mobilization_threshold: float = 0.0  # minimum total campaign cost ($) to proceed
    mobilization_days: float = 0.0  # lead-time days before crew is on-site

    # Periodic (planned) nourishment cycle — BeachFX's calendar-driven renourishment
    # (gbPlannedNourishmentFlag / gdatePlannedNourishmentStartDate / the 365 ×
    # gdwNourishmentTimeIncrement cycle in cShoreResponseIteration.cpp:137-155).
    # Distinct from the post-storm campaign: the calendar makes the reach *eligible*,
    # the volume gate still decides (so a healthy beach skips its cycle).  A cycle
    # landing while the reach is still recovering defers to the recovery completion.
    # None = no periodic nourishment; the reach nourishes only in response to storms.
    cycle_interval_years: float | None = None
    cycle_start_date: datetime | None = None  # first cycle; None = the sim start date

    # Campaign scheduling
    blackout_windows: list[tuple[float, float]] = Field(
        default_factory=list
    )  # (t_start_days, t_end_days)
    # Storm-during-placement policy — the divergence between the parent frameworks.
    # "interrupt" (Python lineage): place what fits, partial-fill, resume after the storm.
    # "defer"    (BeachFX lineage): don't start a placement a storm would hit; delay the
    #             start to a storm-free window (retried in the next inter-storm gap).
    storm_conflict: Literal["interrupt", "defer"] = "interrupt"

    # Assessment selection (Tier 1).  ``assessor`` = reach default; a profile in
    # ``per_profile_assessor`` overrides it; when neither is set the assessor is
    # auto-classified per profile (dune present -> geometric, else volume).
    assessor: Literal["volume", "fitted", "geometric"] | None = None
    per_profile_assessor: dict[str, Literal["volume", "fitted", "geometric"]] = Field(
        default_factory=dict
    )
    # Emergency geometric trigger thresholds (cReach.cpp:509): a profile forces
    # mobilization when its measured dune_height < dune_height OR dune_width <
    # dune_width.  None = that criterion is off; berm_width is NOT a trigger.
    # Dune-aware assessors (fitted, geometric) only.
    emergency_geometry: GeometryThresholds = Field(default_factory=GeometryThresholds)
    # Assessor-agnostic emergency trigger: a profile forces mobilization when its
    # measured subaerial deficit meets this volume threshold (m³).  The base
    # ``ProfileAssessor`` path — dune-aware assessors add ``emergency_geometry`` on
    # top and fall back to this.  None = the volume emergency path is off.
    emergency_volume: ufloat("m3", "cy") | None = None

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
        if self.volume_trigger is not None and self.volume_trigger <= 0:
            raise ValueError("volume_trigger must be positive when set (omit for emergency-only)")
        if self.emergency_volume is not None and self.emergency_volume <= 0:
            raise ValueError("emergency_volume must be positive when set")
        if self.production_rate <= 0:
            raise ValueError("production_rate must be positive")
        if self.cycle_interval_years is not None and self.cycle_interval_years <= 0:
            raise ValueError("cycle_interval_years must be positive when set")
        return self

    @model_validator(mode="after")
    def _cycle_needs_volume_gate(self) -> NourishmentConfig:
        """A periodic cycle proposes; ``volume_trigger`` disposes.  Without the gate the
        cycle date has nothing to clear, so the calendar would place fill every cycle
        regardless of the beach's state — configure the gate, or drop the cycle and let
        the reach nourish off its storm triggers."""
        if self.cycle_interval_years is not None and self.volume_trigger is None:
            raise ValueError(
                "cycle_interval_years requires volume_trigger — the calendar makes the "
                "reach eligible but the volume gate decides whether to mobilize"
            )
        return self

    @model_validator(mode="after")
    def _require_active_trigger(self) -> NourishmentConfig:
        """A present nourishment block must actually do something: at least one
        trigger active (regular volume, emergency volume, or a geometric dune
        threshold).  A block with none is a silent no-op — express no-action
        explicitly with ``nourishment: null`` instead."""
        tg = self.emergency_geometry
        has_trigger = (
            self.volume_trigger is not None
            or self.emergency_volume is not None
            or tg.dune_height is not None
            or tg.dune_width is not None
        )
        if not has_trigger:
            raise ValueError(
                "nourishment block has no active trigger; set volume_trigger, "
                "emergency_volume, or emergency_geometry.dune_height/dune_width — or use "
                "`null` (no nourishment) for a recovery-only reach"
            )
        return self

    @model_validator(mode="after")
    def _restore_clears_trigger(self) -> NourishmentConfig:
        """A configured restore geometry must be able to clear its own emergency
        trigger.  If the restore target sits below the trigger threshold, even a full
        restore leaves the dune tripping the trigger, so the profile re-fires an
        emergency nourishment every interval.  Only checkable when both the restore
        target and the trigger are set explicitly — an unset restore target falls back
        to the measured as-built geometry, unknowable at config time."""
        tmpl, trig = self.template_geometry, self.emergency_geometry
        for attr in ("dune_height", "dune_width"):
            target, threshold = getattr(tmpl, attr), getattr(trig, attr)
            if target is not None and threshold is not None and target < threshold:
                raise ValueError(
                    f"template_geometry.{attr} ({target}) is below emergency_geometry.{attr} "
                    f"({threshold}); a full restore can't clear the emergency trigger and "
                    "would re-fire it every interval"
                )
        return self
