"""Tier-1 profile assessment: fill-deficit assessors and template synthesis."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..metrics import erosion_volume_above_msl, fit_profile
from .config import GeometryThresholds, NourishmentConfig

if TYPE_CHECKING:
    from ..config import ReachConfig
    from ..profile import Profile

log = logging.getLogger(__name__)


def _interp_template(profile: Profile, ncfg: NourishmentConfig) -> np.ndarray:
    tx = np.asarray(ncfg.template_x)
    tz = np.asarray(ncfg.template_z)
    return np.interp(profile.x, tx, tz, left=float(tz[0]), right=float(tz[-1]))


def _depth_of_closure(profile: Profile, cfg: ReachConfig) -> float:
    """Resolve DoC: per-profile override (``geometry.depth_of_closure``) else reach default."""
    if profile.geometry is not None and profile.geometry.depth_of_closure is not None:
        return float(profile.geometry.depth_of_closure)
    return float(cfg.depth_of_closure)


def _trigger_geometry(cfg: ReachConfig) -> GeometryThresholds:
    """Emergency trigger thresholds, or an all-``None`` (inactive) set when a bare
    reach config carries no nourishment policy."""
    return cfg.nourishment.trigger_geometry if cfg.nourishment else GeometryThresholds()


def _dune_emergency_force(metrics: dict, tg: GeometryThresholds) -> bool:
    """Geometric emergency trigger (cReach.cpp:509): forced when the measured dune
    height (front relief) OR width falls below a configured threshold.  Each
    criterion is active only when its threshold is set; berm width is not a
    trigger.  Reads the fitted ``metrics`` dict, so a profile with no measured dune
    never fires.  Shared by the dune-aware assessors (Fitted, Geometric)."""
    relief = metrics.get("dune_front_relief", np.nan)
    width = metrics.get("dune_width", np.nan)
    if tg.dune_height is not None and np.isfinite(relief) and relief < float(tg.dune_height):
        return True
    if tg.dune_width is not None and np.isfinite(width) and 0.0 < width < float(tg.dune_width):
        return True
    return False


@dataclass
class ProfileAssessment:
    """Tier-1 physical read of one profile: whether it needs fill, and how much.

    ``volume_m3`` is the subaerial deficit (m³) that drives the reach trigger
    gate; ``placement_m3`` is the full active-height volume actually placed
    (subaerial deficit extended down to depth of closure), which Tier 2 scales
    by the borrow ratio to drive duration and cost. ``force`` lets a profile
    override the reach economic default (the emergency trigger — set by the
    assessor's ``emergency_force``: a volume threshold on the base, plus the
    geometric dune criterion for dune-aware assessors). ``metrics`` is free-form
    json that varies by assessor and rides along for output.
    """

    needs_fill: bool
    volume_m3: float
    placement_m3: float = 0.0
    force: bool = False
    metrics: dict = field(default_factory=dict)


class ProfileAssessor(ABC):
    """Reads a profile and returns its fill deficit (Tier 1 — physical)."""

    @abstractmethod
    def assess(self, profile: Profile, cfg: ReachConfig, width_m: float) -> ProfileAssessment: ...

    def emergency_force(self, a: ProfileAssessment, cfg: ReachConfig) -> bool:
        """Does this profile force reach mobilization on its own (emergency trigger)?

        Base rule (assessor-agnostic): the measured subaerial deficit meets the
        configured ``emergency_volume`` threshold.  Dune-aware assessors override to
        add the geometric dune trigger, falling back to this volume rule.  A ``None``
        threshold (or a bare reach config) leaves the volume path off.
        """
        ncfg = cfg.nourishment
        ev = ncfg.emergency_volume if ncfg is not None else None
        return ev is not None and a.volume_m3 >= float(ev)

    def restore_template(self, profile: Profile, cfg: ReachConfig) -> np.ndarray:
        """The bed shape (on the profile grid) this assessor restores to.

        Default: the reach config template array (``template_x/template_z``).
        ``GeometricAssessor`` overrides it to synthesize an idealized profile, so
        for that assessor the deficit target and the applied shape coincide. Kept
        polymorphic (not a field on ``ProfileAssessment``) because template
        synthesis is assessor-specific — most assessors just use the config array.
        """
        return _interp_template(profile, cfg.nourishment)


class VolumeAssessor(ProfileAssessor):
    """Deficit = subaerial erosion below the template, above MSL (erosion-only).

    The native array readout (≈ the former ``EqualSpacingStrategy``) and the
    reach-wide default: the per-station positive difference (template − current)
    clipped at ``cfg.msl``, so accreted sections never offset a deficit.  Only
    dry-beach loss counts — nourishment is placed on land (see
    ``ARCHITECTURE_nourishment``); depth-of-closure scaling to a borrow volume
    is a Tier-2 / FittedAssessor concern.
    """

    def assess(self, profile, cfg, width_m):
        template_zb = self.restore_template(profile, cfg)
        deficit = erosion_volume_above_msl(profile.x, template_zb, profile.zb, cfg.msl) * width_m
        dclose = _depth_of_closure(profile, cfg)
        # Extend the subaerial deficit (a dry wedge of height BE above MSL) down to
        # the depth of closure: same berm advance, full active height BE+DClose, so
        # placement = deficit × (BE+DClose)/BE.  (Assumes MSL ≈ datum, the default;
        # BE is the measured berm height above MSL.)  BE comes from the INIT fit
        # (ref_metrics); without it the deficit can't be inflated — place as-is.
        placement = deficit
        if deficit > 0.0 and dclose > 0.0:
            ref = profile.ref_metrics
            be = ref.berm_elevation - cfg.msl if ref is not None else np.nan
            if np.isfinite(be) and be > 0.0:
                placement = deficit * (be + dclose) / be
            else:
                log.warning(
                    "VolumeAssessor: profile %s has no measured BE; "
                    "placement not extended to depth of closure",
                    profile.id,
                )
        a = ProfileAssessment(
            needs_fill=deficit > 0.0,
            volume_m3=deficit,
            placement_m3=placement,
            metrics={"msl": cfg.msl, "depth_of_closure": dclose},
        )
        a.force = self.emergency_force(a, cfg)
        return a


class FittedAssessor(ProfileAssessor):
    """Deficit from the FITTED geometry: measured berm-width shortfall vs the
    INIT (as-built) berm — the "hybrid" that reads the chained array *through*
    the fitter (``metrics.py``), so BeachFX's geometric workflow runs on the real
    post-storm profile instead of raw array volume.

    ``BE`` and the target berm width come from the INIT fit (``ref_metrics``,
    Phase-3 decision 2a: measured, not a design-BE config).  Requires
    ``ref_metrics``; a profile without it isn't a `FittedAssessor` candidate and
    reports no fill (per-profile assessor selection lands in Phase 4).

    NOTE: trusts the fitter's measured berm — solid on clean profiles, but berm
    detection on noisy real profiles is a known gap (e.g. ``reach1_p0``); fixing
    that is a parallel track (Phase-3 decision 1b).  The ``(BE+DClose)`` placement
    volume and the emergency ``force`` trigger arrive in 3.2 / Phase 4.
    """

    _basis = "fitted"

    def _target_berm_width(self, cfg: ReachConfig, ref) -> float:
        """Restore-target berm width: the as-built (``ref``) berm (Phase-3 decision 2a)."""
        return ref.berm_width

    def _extra_placement(self, profile, cfg, ref, m, width_m) -> float:
        """Placement volume beyond the berm wedge (assessor-specific). Default: none;
        ``GeometricAssessor`` overrides it to add the rebuilt dune's fill."""
        return 0.0

    def emergency_force(self, a: ProfileAssessment, cfg: ReachConfig) -> bool:
        """Dune-aware trigger: the geometric dune criterion (shared
        ``_dune_emergency_force``) OR the base volume fallback."""
        if _dune_emergency_force(a.metrics, _trigger_geometry(cfg)):
            return True
        return super().emergency_force(a, cfg)

    def assess(self, profile, cfg, width_m):
        ref = profile.ref_metrics
        if ref is None or not np.isfinite(ref.berm_elevation):
            return ProfileAssessment(
                needs_fill=False, volume_m3=0.0, metrics={"basis": self._basis}
            )
        be = ref.berm_elevation
        m, _ideal = fit_profile(profile.x, profile.zb, be, cfg.msl, ref=ref)
        target_berm_width = self._target_berm_width(cfg, ref)
        berm_deficit = max(0.0, target_berm_width - m.berm_width)
        dclose = _depth_of_closure(profile, cfg)
        volume_m3 = berm_deficit * be * width_m  # subaerial dry-wedge trigger deficit
        # Placement fills the full active wedge, crest (BE) down to closure (cReach.cpp:1649).
        placement_m3 = berm_deficit * (be + dclose) * width_m
        metrics = {
            "basis": self._basis,
            "berm_elevation": be,
            "berm_width": m.berm_width,
            "target_berm_width": target_berm_width,
            "berm_width_deficit": berm_deficit,
            "depth_of_closure": dclose,
            "dune_crest_elevation": m.dune_crest_elevation,
            "dune_front_relief": m.dune_front_relief,
            "dune_width": m.dune_width,
        }
        dune_fill = self._extra_placement(profile, cfg, ref, m, width_m)
        if dune_fill > 0.0:
            placement_m3 += dune_fill
            metrics["dune_fill_m3"] = dune_fill
        a = ProfileAssessment(
            needs_fill=berm_deficit > 0.0,
            volume_m3=volume_m3,
            placement_m3=placement_m3,
            metrics=metrics,
        )
        a.force = self.emergency_force(a, cfg)
        return a


def _dune_target(tg: GeometryThresholds, ref, m) -> tuple[float, float, float] | None:
    """Target ``(front_relief, width, front_fraction)`` for the synthesized dune,
    or ``None`` when there's no dune to restore.

    Height (crest above the berm) and total footprint width fall back to the
    as-built ``ref`` when unset (the default->override idiom, matching the berm).
    The crest splits the footprint by the measured front:back width ratio (``ref``,
    else symmetric).  Requires a measured dune (``m.dune_crest_x``): the restore
    anchors on the measured landward toe, so a profile with no dune keeps the
    berm-only shape and preserves its back (Phase-4b bound).
    """
    if not np.isfinite(m.dune_crest_x):
        return None
    height = float(tg.dune_height) if tg.dune_height is not None else float(ref.dune_front_relief)
    width = float(tg.dune_width) if tg.dune_width is not None else float(ref.dune_width)
    if not (np.isfinite(height) and height > 0.0 and np.isfinite(width) and width > 0.0):
        return None
    fw, bw = m.dune_front_width, m.dune_back_width
    if not (np.isfinite(fw) and np.isfinite(bw) and fw + bw > 0.0):
        fw, bw = ref.dune_front_width, ref.dune_back_width
    front_frac = fw / (fw + bw) if np.isfinite(fw) and np.isfinite(bw) and fw + bw > 0.0 else 0.5
    return height, width, float(front_frac)


def _synthesize_berm_template(
    profile: Profile, cfg: ReachConfig, ref, m
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Synthesize a berm(+dune)-restore template on the profile grid.

    Advances the shoreline seaward (nourishment adds material seaward) while the
    landward structure is fixed: the berm is rebuilt flat at the as-built elevation
    ``BE`` to the target width, the foreshore ramps down at the measured slope, and
    the seaward end ties to the first raw data point (``x[0]``).

    When the profile has a dune to restore (``_dune_target``), the dune is rebuilt
    too: its **landward toe stays fixed** at the measured position and upland
    elevation ``UE`` (freezing the back, as the berm does), the crest rises to
    ``BE + target_relief``, and the footprint grows *seaward* to the target width —
    so the seaward dune toe becomes the berm's landward edge.  The measured back
    landward of that toe is preserved.  Fill-only — the result never dips below the
    existing bed (``np.maximum``), so restoration cannot carve.

    Returns ``(template, dune_span)`` where ``dune_span`` is the ``(seaward_toe,
    landward_toe)`` x-range of the rebuilt dune (for metering its fill volume), or
    ``None`` when no dune was synthesized.
    """
    x = profile.x
    zb = profile.zb
    datum = float(cfg.msl)
    be = float(ref.berm_elevation)
    tg = cfg.nourishment.template_geometry
    target_bw = float(tg.berm_width) if tg.berm_width is not None else float(ref.berm_width)
    slope = m.foreshore_slope if np.isfinite(m.foreshore_slope) and m.foreshore_slope > 0 else None
    if slope is None:
        slope = ref.foreshore_slope if np.isfinite(ref.foreshore_slope) else 0.1

    dune = _dune_target(tg, ref, m)
    if dune is not None:
        relief, width, front_frac = dune
        de = be + relief  # target crest elevation
        ue = m.upland_elevation if np.isfinite(m.upland_elevation) else ref.upland_elevation
        ue = float(ue) if np.isfinite(ue) else be
        x_lt = float(m.dune_crest_x) + float(m.dune_back_width)  # fixed measured landward toe
        x_lt = min(max(x_lt, float(x[0])), float(x[-1]))
        x_crest = x_lt - width * (1.0 - front_frac)  # back_width landward of the crest
        x_bl = x_lt - width  # seaward dune toe = berm landward edge
    else:
        # No dune to restore: berm ends at the measured dune toe / upland rise.
        if np.isfinite(m.dune_crest_x):
            x_bl = float(m.dune_crest_x) - float(m.dune_front_width)  # seaward dune toe
        else:
            x_bl = float(m.shoreline_x) + float(m.berm_width)
    x_bl = min(max(x_bl, float(x[0])), float(x[-1]))

    x_bs = x_bl - target_bw  # seaward berm edge (foreshore crest)
    x_sh = x_bs - (be - datum) / slope  # new shoreline (foreshore foot)
    # Clamp to the grid, keep knots non-decreasing in x.
    x_sh = max(x_sh, float(x[0]))
    x_bs = max(x_bs, x_sh)
    x_bl = max(x_bl, x_bs)

    kx = [float(x[0]), x_sh, x_bs, x_bl]
    kz = [float(zb[0]), datum, be, be]
    x_far = x_bl  # landward extent of the synthesized front; back preserved beyond
    if dune is not None:
        x_crest = max(x_crest, x_bl)
        x_lt = max(x_lt, x_crest)
        kx += [x_crest, x_lt]
        kz += [de, ue]
        x_far = x_lt
    front = np.interp(x, kx, kz, left=float(zb[0]), right=kz[-1])

    tmpl = zb.copy()  # preserve the measured back
    seaward = x <= x_far
    tmpl[seaward] = front[seaward]
    span = (x_bl, x_lt) if dune is not None else None
    return np.maximum(tmpl, zb), span


class GeometricAssessor(FittedAssessor):
    """``FittedAssessor``'s decision (measured berm-width shortfall via the fitter),
    but the restore template is SYNTHESIZED from target geometry
    (``NourishmentConfig.template_geometry``, as-built fallback) rather than the
    reach's ``template_x/template_z`` array — BeachFX's parametric restore, run on
    the real post-storm profile.  The target berm width comes from the same config,
    so the deficit target and the synthesized applied shape coincide.

    The synthesized shape rebuilds the berm + foreshore and, when the profile has a
    dune, the dune too — crest to ``BE + template_geometry.dune_height`` over a
    ``dune_width`` footprint (as-built ``ref`` fallback), landward toe fixed on the
    measured back (see ``_synthesize_berm_template``).  So an emergency force
    restores the geometry that tripped it.  ``assess`` meters the subaerial dune
    fill into ``placement_m3`` on top of the berm wedge (``_extra_placement``).
    """

    _basis = "geometric"

    def _target_berm_width(self, cfg: ReachConfig, ref) -> float:
        tg = cfg.nourishment.template_geometry
        return float(tg.berm_width) if tg.berm_width is not None else ref.berm_width

    def _extra_placement(self, profile, cfg, ref, m, width_m) -> float:
        """Subaerial dune fill the synthesized template adds over the current bed,
        integrated across the rebuilt dune footprint.  The dune is a subaerial
        structure built on the berm crest, so — unlike the berm wedge — it is NOT
        extended to the depth of closure (no ``(BE+DClose)`` inflation)."""
        _tmpl, span = _synthesize_berm_template(profile, cfg, ref, m)
        if span is None:
            return 0.0
        x_bl, x_lt = span
        region = (profile.x >= x_bl) & (profile.x <= x_lt)
        if np.count_nonzero(region) < 2:
            return 0.0
        return (
            erosion_volume_above_msl(profile.x[region], _tmpl[region], profile.zb[region], cfg.msl)
            * width_m
        )

    def restore_template(self, profile, cfg):
        ref = profile.ref_metrics
        if ref is None or not np.isfinite(ref.berm_elevation):
            return super().restore_template(profile, cfg)  # no fit basis → config array
        m, _ideal = fit_profile(profile.x, profile.zb, ref.berm_elevation, cfg.msl, ref=ref)
        tmpl, _span = _synthesize_berm_template(profile, cfg, ref, m)
        return tmpl


_ASSESSORS: dict[str, ProfileAssessor] = {
    "volume": VolumeAssessor(),
    "fitted": FittedAssessor(),
    "geometric": GeometricAssessor(),
}
_DEFAULT_ASSESSOR: ProfileAssessor = _ASSESSORS["volume"]


def _warn_geometric_deprecated(profile_id: str) -> None:
    """Signal that ``GeometricAssessor`` is legacy — kept working, but users should
    migrate to ``fitted`` (successor) or ``volume``.  Emitted on both a log warning
    (visible in pipeline output) and a ``DeprecationWarning`` (for tooling/tests)."""
    msg = (
        f"GeometricAssessor (profile {profile_id}) is a legacy assessor; prefer "
        "'fitted' or 'volume'. It stays available but may be removed in future."
    )
    log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


def _select_assessor(profile: Profile, ncfg: NourishmentConfig) -> ProfileAssessor:
    """Pick a profile's assessor: per-profile override, else reach default, else
    auto-classify (dune present -> geometric, else volume).  Explicit wins.  Any
    path that lands on the legacy ``GeometricAssessor`` emits a deprecation warning.
    """
    name = ncfg.per_profile_assessor.get(profile.id) or ncfg.assessor
    if name is None:
        ref = profile.ref_metrics
        has_dune = ref is not None and np.isfinite(ref.dune_crest_elevation)
        name = "geometric" if has_dune else "volume"
    if name == "geometric":
        _warn_geometric_deprecated(profile.id)
    return _ASSESSORS[name]
