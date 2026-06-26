from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from .metrics import ProfileMetrics, fit_profile, last_wet_dry_crossing
from .types import SnapshotLabel, StormResponseType
from .units import ufloat

if TYPE_CHECKING:
    from .runner.base import CSHOREResult


def _shoreline_x(x: np.ndarray, z: np.ndarray, datum: float = 0.0) -> float:
    """Return x coordinate of the last (most landward) wet→dry zero-crossing.

    Falls back to the offshore node when the whole profile is submerged and to
    the landward node when it is entirely dry.
    """
    x_shore, _ = last_wet_dry_crossing(x, z, datum)
    if x_shore is None:
        return float(x[0]) if np.all(z <= datum) else float(x[-1])
    return x_shore


def _shoreline_shift(
    x_old: np.ndarray, z_old: np.ndarray,
    x_new: np.ndarray, z_new: np.ndarray,
    datum: float = 0.0,
) -> float:
    """Landward shift in shoreline position between old and new grids (m)."""
    return _shoreline_x(x_new, z_new, datum) - _shoreline_x(x_old, z_old, datum)


# ---------------------------------------------------------------------------
# Per-profile geometry reference (immutable throughout the simulation)
# ---------------------------------------------------------------------------

class ProfileGeometryConfig(BaseModel):
    """Immutable geometry reference for one cross-shore profile transect.

    ``berm_elevation`` (BE) is the design/target berm crest elevation used for
    BeachFX morphology classification and metric extraction.  It remains
    constant throughout the simulation regardless of storm response.
    ``recovery_duration`` overrides ``StormConfig.T_recover`` for this profile;
    ``None`` means use the reach-wide default.
    """
    berm_elevation: ufloat("m", "ft")
    datum: float = 0.0
    recovery_duration: ufloat("days") | None = None


# ---------------------------------------------------------------------------
# Profile: pure data + snapshot()
# ---------------------------------------------------------------------------

@dataclass
class ProfileSnapshot:
    label: SnapshotLabel
    x: np.ndarray
    zb: np.ndarray
    t: float = 0.0
    metrics: ProfileMetrics | None = None
    storm_response_type: StormResponseType | None = None


@dataclass
class Profile:
    id: str
    x: np.ndarray
    zb: np.ndarray
    d50: float
    snapshots: list[ProfileSnapshot] = field(default_factory=list)
    geometry: ProfileGeometryConfig | None = field(default=None)
    ref_metrics: ProfileMetrics | None = field(default=None)

    def snapshot(self, label: SnapshotLabel, t: float = 0.0) -> None:
        metrics = None
        if self.geometry is not None:
            metrics, _ideal = fit_profile(
                self.x, self.zb,
                self.geometry.berm_elevation,
                self.geometry.datum,
                ref=self.ref_metrics,
            )
        snap = ProfileSnapshot(
            label=label, x=self.x.copy(), zb=self.zb.copy(), t=t, metrics=metrics,
        )
        self.snapshots.append(snap)
        # Pin ref_metrics to the INIT snapshot — constrains dune search in all subsequent fits
        if label == SnapshotLabel.INIT and metrics is not None:
            self.ref_metrics = metrics


# ---------------------------------------------------------------------------
# ProfileEvent ABC
# ---------------------------------------------------------------------------

@dataclass
class ProfileEvent(ABC):
    t: float

    @abstractmethod
    def apply(self, profile: Profile) -> None: ...


# ---------------------------------------------------------------------------
# Concrete events
# ---------------------------------------------------------------------------

@dataclass
class ErosionTick(ProfileEvent):
    """Lower bed by erosion + SLC during the inter-storm interval."""
    dz_erosion: float | np.ndarray = 0.0
    dz_slc:     float | np.ndarray = 0.0

    def apply(self, profile: Profile) -> None:
        profile.zb = profile.zb - (self.dz_erosion + self.dz_slc)
        profile.snapshot(SnapshotLabel.Periodic, self.t)


@dataclass
class StormResponse(ProfileEvent):
    """Apply CSHORE result to profile — zb interpolated onto the fixed profile grid.

    ``profile.x`` is never changed; CSHORE output is re-sampled onto the
    original fixed grid so every snapshot shares the same x-axis.
    """
    result: CSHOREResult

    def apply(self, profile: Profile) -> None:
        profile.zb = np.interp(
            profile.x, self.result.x, self.result.zb,
            left=self.result.zb[0], right=self.result.zb[-1],
        )
        profile.snapshot(SnapshotLabel.PostStorm, self.t)


@dataclass
class Recovery(ProfileEvent):
    """Blend profile from post-storm shape toward pre-storm shape by fraction.

    ``zb_pre_storm`` must already be re-interpolated onto the post-storm grid
    (shifted by shoreline offset) by the phase runner before constructing this event.
    Only nodes below ``z_berm`` are blended; nodes at or above are left unchanged.
    """
    fraction:      float
    zb_post_storm: np.ndarray
    zb_pre_storm:  np.ndarray
    z_berm:        float | None = None

    def apply(self, profile: Profile) -> None:
        blend = self.zb_post_storm + self.fraction * (self.zb_pre_storm - self.zb_post_storm)
        if self.z_berm is not None:
            new_zb = profile.zb.copy()
            mask = self.zb_post_storm < self.z_berm
            new_zb[mask] = blend[mask]
            profile.zb = new_zb
        else:
            profile.zb = blend
        profile.snapshot(SnapshotLabel.REC, self.t)


@dataclass
class FullNourishment(ProfileEvent):
    """Place complete nourishment template (campaign reaches this profile fully)."""
    template_zb: np.ndarray

    def apply(self, profile: Profile) -> None:
        profile.zb = self.template_zb.copy()
        profile.snapshot(SnapshotLabel.ESN, self.t)
        # Reset ref_metrics to the post-nourishment shape so subsequent storm
        # fits are constrained to the new equilibrium dune position.
        if profile.snapshots and profile.snapshots[-1].metrics is not None:
            profile.ref_metrics = profile.snapshots[-1].metrics


@dataclass
class PartialNourishment(ProfileEvent):
    """Place partial nourishment toward template (campaign interrupted before completion)."""
    template_zb: np.ndarray
    fraction:    float

    def apply(self, profile: Profile) -> None:
        profile.zb = profile.zb + self.fraction * (self.template_zb - profile.zb)
        profile.snapshot(SnapshotLabel.SSN, self.t)
