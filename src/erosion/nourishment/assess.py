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

# Trapezoid plateau width as a fraction of the dune footprint, used only when an
# explicit ``dune_form="trapezoid"`` has no measured plateau to size it from.
_DUNE_TOP_FRAC = 0.3


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
    trigger.  Reads the fitted ``metrics`` dict: a profile that never had a dune
    never fires, while a reference dune the storm erased (``dune_lost``) counts
    as below every active threshold.  Shared by the dune-aware assessors
    (Fitted, Geometric)."""
    # An erased dune measures as NaN relief / zero width -- below every active
    # threshold by definition, not unmeasurable
    if metrics.get("dune_lost", False):
        return tg.dune_height is not None or tg.dune_width is not None
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

    ``deficit_m3`` is the subaerial deficit (m³) that drives the reach trigger
    gate; ``placement_m3`` is the full active-height volume actually placed
    (subaerial deficit extended down to depth of closure), which Tier 2 scales
    by the borrow ratio to drive duration and cost. ``force`` lets a profile
    override the reach economic default (the emergency trigger — set by the
    assessor's ``emergency_force``: a volume threshold on the base, plus the
    geometric dune criterion for dune-aware assessors). ``metrics`` is free-form
    json that varies by assessor and rides along for output.
    """

    needs_fill: bool
    deficit_m3: float
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
        return ev is not None and a.deficit_m3 >= float(ev)

    def restore_template(self, profile: Profile, cfg: ReachConfig) -> np.ndarray:
        """The restore bed shape (on the profile grid) this assessor builds to.

        A parametric template synthesized from ``template_geometry`` at the measured
        shoreline — so it is CSHORE-frame by construction (landward-positive, x=0
        offshore) and fill-only.  The template metrics drive the target shape (berm
        width/height, foreshore slope, dune); the fit supplies only the horizontal
        anchor (where the shoreline/dune sit on this profile).  Shared by every
        assessor — ``GeometricAssessor`` no longer needs its own override.

        Requires ``ref_metrics`` (the as-built INIT fit) as the fit basis; a profile
        without one places no fill.
        """
        ref = profile.ref_metrics
        if ref is None or not np.isfinite(ref.berm_elevation):
            log.warning(
                "restore_template: profile %s has no as-built fit basis; placing no fill",
                profile.id,
            )
            return profile.zb.copy()
        tmpl, _span = _synthesize_berm_template(profile, cfg, ref)
        return tmpl


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
            deficit_m3=deficit,
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
                needs_fill=False, deficit_m3=0.0, metrics={"basis": self._basis}
            )
        be = ref.berm_elevation
        m, _ideal = fit_profile(profile.x, profile.zb, be, cfg.msl, ref=ref)
        target_berm_width = self._target_berm_width(cfg, ref)
        berm_deficit = max(0.0, target_berm_width - m.berm_width)
        dclose = _depth_of_closure(profile, cfg)
        deficit_m3 = berm_deficit * be * width_m  # subaerial dry-wedge trigger deficit
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
            # The ref had a dune but the post-storm fit measures none (the
            # HIGH_UPLAND flip past the prominence floor) -- storm-erased,
            # not never-there
            "dune_lost": bool(
                np.isfinite(ref.dune_crest_elevation) and not np.isfinite(m.dune_crest_elevation)
            ),
        }
        dune_fill = self._extra_placement(profile, cfg, ref, m, width_m)
        if dune_fill > 0.0:
            placement_m3 += dune_fill
            metrics["dune_fill_m3"] = dune_fill
        a = ProfileAssessment(
            needs_fill=berm_deficit > 0.0,
            deficit_m3=deficit_m3,
            placement_m3=placement_m3,
            metrics=metrics,
        )
        a.force = self.emergency_force(a, cfg)
        return a


def _dune_target(tg: GeometryThresholds, ref) -> tuple[float, float, float] | None:
    """Target ``(front_relief, width, front_fraction)`` for the synthesized dune,
    or ``None`` when there's no dune to restore.

    Anchored on the idealized reference ``ref`` (the pinned as-built / "previous"
    geometry), NOT the storm-damaged fit: the back structure does not erode, so the
    restore rebuilds the dune the profile HAD — even when the damaged profile has
    lost it.  Height (crest above the berm) and footprint width come from the
    template metrics, ``ref`` as fallback; the crest splits the footprint by the
    reference front:back width ratio (else symmetric).
    """
    if not np.isfinite(ref.dune_crest_x):
        return None
    height = float(tg.dune_height) if tg.dune_height is not None else float(ref.dune_front_relief)
    width = float(tg.dune_width) if tg.dune_width is not None else float(ref.dune_width)
    if not (np.isfinite(height) and height > 0.0 and np.isfinite(width) and width > 0.0):
        return None
    fw, bw = ref.dune_front_width, ref.dune_back_width
    front_frac = fw / (fw + bw) if np.isfinite(fw) and np.isfinite(bw) and fw + bw > 0.0 else 0.5
    return height, width, float(front_frac)


def _synthesize_berm_template(
    profile: Profile, cfg: ReachConfig, ref
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Synthesize a berm(+dune)-restore template on the profile grid.

    The whole subaerial shape is anchored on the idealized reference ``ref`` (the
    pinned as-built / "previous" berm position), NOT the storm-damaged profile:
    storm erosion attacks the beach front, not the back structure, so the restore
    rebuilds the berm/dune WHERE THEY WERE rather than following the retreated
    shoreline the damaged fit reads.  The berm is rebuilt flat at ``BE`` to the
    target width, the foreshore ramps down at the reference slope, and (when the
    reference has a dune) the dune is rebuilt to ``BE + target_relief`` over the
    target footprint, its landward toe fixed at the reference position.

    Fill is subaerial and fill-only (see the placement step): the result never dips
    below the existing bed and never places sand below the datum, so the subaqueous
    profile CSHORE runs on is untouched.

    Returns ``(template, dune_span)`` where ``dune_span`` is the ``(seaward_toe,
    landward_toe)`` x-range of the rebuilt dune (for metering its fill volume), or
    ``None`` when no dune was synthesized.
    """
    x = profile.x
    zb = profile.zb
    datum = float(cfg.msl)
    tg = cfg.nourishment.template_geometry
    # Target geometry is template-metrics-driven; the idealized ``ref`` is the
    # fallback for any unset metric AND the horizontal anchor for the whole shape.
    be = float(tg.berm_height) if tg.berm_height is not None else float(ref.berm_elevation)
    target_bw = float(tg.berm_width) if tg.berm_width is not None else float(ref.berm_width)
    if tg.foreshore_slope is not None:
        slope = float(tg.foreshore_slope)
    elif np.isfinite(ref.foreshore_slope) and ref.foreshore_slope > 0:
        slope = float(ref.foreshore_slope)
    else:
        slope = 0.1

    dune = _dune_target(tg, ref)
    if dune is not None:
        relief, width, front_frac = dune
        de = be + relief  # target crest elevation
        ue = float(ref.upland_elevation) if np.isfinite(ref.upland_elevation) else be
        x_lt = float(ref.dune_crest_x) + float(ref.dune_back_width)  # idealized landward toe
        x_lt = min(max(x_lt, float(x[0])), float(x[-1]))
        x_crest = x_lt - width * (1.0 - front_frac)  # back_width landward of the crest
        x_bl = x_lt - width  # seaward dune toe = berm landward edge
    else:
        # No dune in the idealized profile: berm ends at the reference dune toe / upland rise.
        if np.isfinite(ref.dune_crest_x):
            x_bl = float(ref.dune_crest_x) - float(ref.dune_front_width)  # seaward dune toe
        else:
            # No dune at all: anchor the berm's LANDWARD edge at the reference berm's
            # back.  ``berm_width`` is the crest FLAT, measured landward of the foreshore
            # foot, so the back sits a foreshore-run + crest-width landward of the
            # shoreline; dropping the foreshore run drops the berm a foreshore-width
            # seaward and overtops the existing foreshore -> phantom deficit.
            x_bl = float(ref.shoreline_x) + (be - datum) / slope + float(ref.berm_width)
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
    form = None
    if dune is not None:
        x_crest = max(x_crest, x_bl)
        x_lt = max(x_lt, x_crest)
        x_far = x_lt
        # Crest shape: config override, else auto (trapezoid if the idealized dune
        # had a measured plateau, else triangle).
        has_plateau = np.isfinite(ref.dune_top_width) and ref.dune_top_width > 0.0
        form = tg.dune_form or ("trapezoid" if has_plateau else "triangle")
        if form == "trapezoid":
            tw = float(ref.dune_top_width) if has_plateau else _DUNE_TOP_FRAC * width
            xc0 = max(x_crest - tw / 2.0, x_bl)
            xc1 = min(x_crest + tw / 2.0, x_lt)
            kx += [xc0, xc1, x_lt]
            kz += [de, de, ue]
        else:  # triangle apex (gaussian re-shapes its footprint below)
            kx += [x_crest, x_lt]
            kz += [de, ue]
    front = np.interp(x, kx, kz, left=float(zb[0]), right=kz[-1])
    if form == "gaussian":
        # Rounded skew-gaussian crest on the linear toe baseline (BE seaward toe ->
        # UE landward toe), independent front/back scales — mirrors the fitter's
        # gaussian form.  Overwrites the linear dune footprint only.
        span = max(x_lt - x_bl, 1e-6)
        base = be + (ue - be) * (x - x_bl) / span
        amp = de - (be + (ue - be) * (x_crest - x_bl) / span)
        sig = np.where(
            x <= x_crest, max((x_crest - x_bl) / 2.0, 1.0), max((x_lt - x_crest) / 2.0, 1.0)
        )
        bump = amp * np.exp(-0.5 * ((x - x_crest) / np.maximum(sig, 1e-6)) ** 2)
        reg = (x >= x_bl) & (x <= x_lt)
        front[reg] = (base + bump)[reg]

    # Placement is SUBAERIAL only: raise the bed to the template where the template
    # stands above BOTH the current bed (fill-only, never carve) AND the datum (dry
    # beach + dune).  Below the datum the bed is left natural — the subaqueous fill
    # is a borrow-accounting concern (the ``(BE+DClose)`` DoC inflation in ``assess``),
    # not a change to the bed CSHORE runs on.
    tmpl = zb.copy()  # preserve the measured back and the whole subaqueous profile
    fill = (x <= x_far) & (front > zb) & (front > datum)
    tmpl[fill] = front[fill]
    span = (x_bl, x_lt) if dune is not None else None
    return tmpl, span


class GeometricAssessor(FittedAssessor):
    """``FittedAssessor``'s decision (measured berm-width shortfall via the fitter),
    plus the dune-fill metering.  The restore template itself is now the shared
    parametric synthesis (``ProfileAssessor.restore_template`` → the
    ``template_geometry`` shape on the profile grid), so ``geometric`` no longer
    owns a distinct template — it differs from ``fitted`` only in metering the
    rebuilt dune's subaerial fill into ``placement_m3`` (``_extra_placement``) and
    in taking its target berm width from ``template_geometry`` rather than as-built.
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
        _tmpl, span = _synthesize_berm_template(profile, cfg, ref)
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
