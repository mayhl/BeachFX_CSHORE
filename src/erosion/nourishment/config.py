"""Reach-level nourishment configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..units import ufloat


class GeometryThresholds(BaseModel):
    """Target restore geometry for the synthesized template (``GeometricAssessor``)
    and, later, the emergency trigger (Phase 4).  Each field is a target; ``None``
    falls back to the measured as-built value (``ref_metrics``) — the
    default->override idiom.  BeachFX-grid numeric defaults are populated per reach
    in config; unset here means restore-to-as-built.
    """

    berm_width: ufloat("m", "ft") | None = None
    dune_height: ufloat("m", "ft") | None = None  # crest above berm (front relief)
    dune_width: ufloat("m", "ft") | None = None


class NourishmentConfig(BaseModel):
    """Reach-level nourishment policy parameters."""

    # Template profile
    template_x: list[
        ufloat("m", "ft")
    ]  # cross-shore positions (CSHORE convention, landward-positive)
    template_z: list[ufloat("m", "ft")]  # bed elevations (m NAVD internally)
    # Restore-target geometry for GeometricAssessor's synthesized template.
    template_geometry: GeometryThresholds = Field(default_factory=GeometryThresholds)

    # Trigger + production
    volume_trigger: ufloat("m3", "cy")  # reach-level volume deficit (m³) to launch a campaign
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
    trigger_geometry: GeometryThresholds = Field(default_factory=GeometryThresholds)

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
