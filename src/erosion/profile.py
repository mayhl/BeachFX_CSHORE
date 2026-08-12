from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

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
    x_old: np.ndarray,
    z_old: np.ndarray,
    x_new: np.ndarray,
    z_new: np.ndarray,
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
    ``depth_of_closure`` overrides ``ReachConfig.depth_of_closure``; ``None``
    means use the reach-wide default.
    """

    model_config = ConfigDict(extra="forbid")

    berm_elevation: ufloat("m", "ft")
    datum: float = 0.0
    recovery_duration: ufloat("days") | None = None
    depth_of_closure: ufloat("m", "ft") | None = None


class Georef(BaseModel):
    """Real-world georeference for a cross-shore transect (the GeoParquet hook).

    ``origin`` is the transect's x=0 node (seaward) in lon/lat; ``azimuth_deg`` is the
    compass bearing (deg from north) toward increasing x (landward).  Optional — when
    unset, outputs stay in local cross-shore coordinates and the geo products are skipped.
    No data carries this yet; it's the seam that lets postprocess emit real geometry when
    a georeferenced dataset arrives.
    """

    model_config = ConfigDict(extra="forbid")

    origin_lon: float
    origin_lat: float
    azimuth_deg: float


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
class EventRecord:
    """One applied event, serialized for the append-only event log (Phase A output spine).

    Lightweight by design: scalars + a small scalar payload only.  The bed the event
    produced lives in the referenced snapshot (``label`` at ``t``), so the log never
    duplicates node arrays.  ``ref_pos`` is a SEAM for the Phase-C ref/recovery policy —
    recorded now, inert until then.
    """

    event_type: str
    t: float
    label: str | None = None
    ref_pos: float | None = None
    payload: dict = field(default_factory=dict)


@dataclass
class Profile:
    id: str
    x: np.ndarray
    zb: np.ndarray
    d50: float
    snapshots: list[ProfileSnapshot] = field(default_factory=list)
    geometry: ProfileGeometryConfig | None = field(default=None)
    ref_metrics: ProfileMetrics | None = field(default=None)
    events: list[EventRecord] = field(default_factory=list)  # append-only event log
    georef: Georef | None = field(default=None)  # transect lon/lat + azimuth (GeoParquet hook)

    def snapshot(self, label: SnapshotLabel, t: float = 0.0) -> None:
        metrics = None
        if self.geometry is not None:
            metrics, _ideal = fit_profile(
                self.x,
                self.zb,
                self.geometry.berm_elevation,
                self.geometry.datum,
                ref=self.ref_metrics,
            )
        snap = ProfileSnapshot(
            label=label,
            x=self.x.copy(),
            zb=self.zb.copy(),
            t=t,
            metrics=metrics,
        )
        self.snapshots.append(snap)
        # Pin ref_metrics to the INIT snapshot — constrains dune search in all subsequent fits
        if label == SnapshotLabel.INIT and metrics is not None:
            self.ref_metrics = metrics

    def last_snapshot(self, label: SnapshotLabel) -> ProfileSnapshot | None:
        """Most recent snapshot carrying ``label``, or ``None`` if there is none."""
        return next((s for s in reversed(self.snapshots) if s.label == label), None)

    def record_event(
        self,
        event_type: str,
        t: float,
        label: SnapshotLabel | None = None,
        ref_pos: float | None = None,
        **payload,
    ) -> None:
        """Append an entry to the profile's append-only event log (Phase A output spine)."""
        lbl = label.value if isinstance(label, SnapshotLabel) else label
        self.events.append(EventRecord(event_type, float(t), lbl, ref_pos, payload))


class Profiles(list):
    """A ``list[Profile]`` with orchestration-level collection helpers.

    It *is* a list (so every ``for p in profiles`` / ``profiles[i]`` /
    ``zip(profiles, …)`` still works); it only adds a couple of bulk operations.
    """

    def snapshot_all(self, label: SnapshotLabel, t: float = 0.0) -> None:
        for p in self:
            p.snapshot(label, t)

    def by_id(self, pid: str) -> Profile:
        return next(p for p in self if p.id == pid)


# ---------------------------------------------------------------------------
# ProfileEvent ABC
# ---------------------------------------------------------------------------


@dataclass
class ProfileEvent(ABC):
    t: float
    event_type: ClassVar[str] = "ProfileEvent"

    @abstractmethod
    def apply(self, profile: Profile) -> None: ...

    def _payload(self) -> dict:
        """Scalar provenance for the event-log entry (arrays live in the snapshot)."""
        return {}

    def _emit(self, profile: Profile, label: SnapshotLabel | None) -> None:
        """Append this event to the profile's log after it has been applied."""
        profile.record_event(self.event_type, self.t, label, **self._payload())


# ---------------------------------------------------------------------------
# Concrete events
# ---------------------------------------------------------------------------


@dataclass
class ErosionTick(ProfileEvent):
    """Lower bed by erosion + SLC during the inter-storm interval."""

    event_type: ClassVar[str] = "ErosionTick"
    dz_erosion: float | np.ndarray = 0.0
    dz_slc: float | np.ndarray = 0.0

    def _payload(self) -> dict:
        return {
            "dz_erosion": float(np.mean(self.dz_erosion)),
            "dz_slc": float(np.mean(self.dz_slc)),
        }

    def apply(self, profile: Profile) -> None:
        profile.zb = profile.zb - (self.dz_erosion + self.dz_slc)
        profile.snapshot(SnapshotLabel.Periodic, self.t)
        self._emit(profile, SnapshotLabel.Periodic)


@dataclass
class StormResponse(ProfileEvent):
    """Apply CSHORE result to profile — zb interpolated onto the fixed profile grid.

    ``profile.x`` is never changed; CSHORE output is re-sampled onto the
    original fixed grid so every snapshot shares the same x-axis.
    """

    event_type: ClassVar[str] = "StormResponse"
    result: CSHOREResult

    def apply(self, profile: Profile) -> None:
        r = self.result
        profile.zb = np.interp(profile.x, r.x, r.zb, left=r.zb[0], right=r.zb[-1])
        # Bed nodes outside CSHORE's returned grid are constant-extrapolated (the JMAX
        # landward-boundary chop); count them so consumers can flag non-physical nodes.
        # jr = landward wet-computation limit (hydro valid over nodes < jr).
        n_extrapolated = int(np.count_nonzero((profile.x < r.x[0]) | (profile.x > r.x[-1])))
        profile.snapshot(SnapshotLabel.PostStorm, self.t)
        profile.record_event(
            "StormResponse",
            self.t,
            SnapshotLabel.PostStorm,
            runup_m=float(r.runup_m),
            jr=int(r.jr),
            n_extrapolated=n_extrapolated,
        )


def recovered_bed(
    zb_post: np.ndarray,
    zb_pre: np.ndarray,
    fraction: float,
    z_berm: float | None = None,
    base_zb: np.ndarray | None = None,
) -> np.ndarray:
    """Bed after applying ``fraction`` of recovery from post- toward pre-storm.

    ``blend = zb_post + fraction·(zb_pre − zb_post)``.  With ``z_berm`` set, only
    nodes where ``zb_post < z_berm`` are blended; the rest keep ``base_zb`` (the
    bed the recovery is applied on top of, defaulting to ``zb_post``).  Pure
    function shared by ``Recovery.apply`` and the recovery gallery.
    """
    blend = zb_post + fraction * (zb_pre - zb_post)
    if z_berm is None:
        return blend
    base = zb_post if base_zb is None else base_zb
    out = np.asarray(base, dtype=float).copy()
    mask = zb_post < z_berm
    out[mask] = blend[mask]
    return out


@dataclass
class Recovery(ProfileEvent):
    """Blend profile from post-storm shape toward pre-storm shape by fraction.

    ``zb_pre_storm`` must already be re-interpolated onto the post-storm grid
    (shifted by shoreline offset) by the phase runner before constructing this event.
    Only nodes below ``z_berm`` are blended; nodes at or above are left unchanged.

    ``interrupted`` marks a recovery cut short before ``T_recover`` elapsed; ``by_nourishment``
    then says which agent cut it: the crew arriving to nourish (``RECN``) vs a following
    storm (BeachFX's forced-recovery ``RECS``).  A recovery that ran its full period (or to
    the sim/window end) snapshots ``REC``.
    """

    event_type: ClassVar[str] = "Recovery"
    fraction: float
    zb_post_storm: np.ndarray
    zb_pre_storm: np.ndarray
    z_berm: float | None = None
    interrupted: bool = False
    by_nourishment: bool = False  # cut short by the crew, not a storm → RECN vs RECS

    def _payload(self) -> dict:
        return {"fraction": float(self.fraction), "interrupted": bool(self.interrupted)}

    def apply(self, profile: Profile) -> None:
        profile.zb = recovered_bed(
            self.zb_post_storm, self.zb_pre_storm, self.fraction, self.z_berm, base_zb=profile.zb
        )
        if self.interrupted:
            label = SnapshotLabel.RECN if self.by_nourishment else SnapshotLabel.RECS
        else:
            label = SnapshotLabel.REC
        profile.snapshot(label, self.t)
        self._emit(profile, label)


@dataclass
class FullNourishment(ProfileEvent):
    """Place complete nourishment template (campaign reaches this profile fully).

    ``label`` is the campaign's end marker — ``EEN`` for a post-storm response,
    ``ESN`` for a periodic planned cycle (``CampaignKind.end_label``).
    """

    event_type: ClassVar[str] = "FullNourishment"
    template_zb: np.ndarray
    label: SnapshotLabel = SnapshotLabel.EEN

    def apply(self, profile: Profile) -> None:
        profile.zb = self.template_zb.copy()
        profile.snapshot(self.label, self.t)
        self._emit(profile, self.label)
        # Reset ref_metrics to the post-nourishment shape so subsequent storm
        # fits are constrained to the new equilibrium dune position.
        if profile.snapshots and profile.snapshots[-1].metrics is not None:
            profile.ref_metrics = profile.snapshots[-1].metrics


@dataclass
class PartialNourishment(ProfileEvent):
    """Place partial nourishment toward template (campaign interrupted before completion).

    Closes the segment the campaign's start marker (``SEN``/``SSN``) opened, with the
    storm-cut end marker (``EENS``/``ESNS`` — ``CampaignKind.partial_label``).  The
    full-fill end markers (``EEN``/``ESN``) are emitted only on resume-to-completion, so
    every placement segment is a matched pair and the label alone says whether the crew
    finished.  A resumed campaign opens a fresh segment with its own start marker.
    """

    event_type: ClassVar[str] = "PartialNourishment"
    template_zb: np.ndarray
    fraction: float
    label: SnapshotLabel = SnapshotLabel.EENS

    def _payload(self) -> dict:
        return {"fraction": float(self.fraction)}

    def apply(self, profile: Profile) -> None:
        profile.zb = profile.zb + self.fraction * (self.template_zb - profile.zb)
        profile.snapshot(self.label, self.t)
        self._emit(profile, self.label)
