"""The public ``fit_profile`` entry point — orchestrates detection, dune-form
selection, and idealization into a ``ProfileMetrics`` + ``IdealizedProfile``."""

from __future__ import annotations

import numpy as np

from .detect import (
    _analysis_cap,
    _classify,
    _detect_berm_and_toe,
    _dx_of,
    _landward_crop,
    _smooth,
)
from .forms import _fit_no_dune, _scarp, _select_dune_form
from .geometry import last_wet_dry_crossing, volume_above_datum
from .types import (
    _CREST_EPS,
    _LEVEL_TOL,
    _MIN_NODES,
    _MIN_PLATEAU_M,
    _NAN,
    _REF_WIN_M,
    IdealizedProfile,
    MorphType,
    ProfileMetrics,
)


def fit_profile(
    x: np.ndarray,
    zb: np.ndarray,
    berm_elevation: float,
    datum: float = 0.0,
    ref: ProfileMetrics | None = None,
    max_feature_length: float | str | None = "auto",
) -> tuple[ProfileMetrics, IdealizedProfile | None]:
    """Extract morphology metrics and an idealized profile from raw ``(x, zb)``.

    ``berm_elevation`` is the *design* berm crest BE used only for
    classification; the berm elevation is otherwise *measured* from the profile.
    ``ref`` (optional) constrains the dune-crest search to ±30 m around
    ``ref.dune_crest_x`` to avoid spurious post-storm peaks.

    ``max_feature_length`` bounds the fit to the primary beach+dune system so
    far-landward terrain (a second dune, back-barrier) on a long cross-shore
    profile is not folded into a single idealized dune:

    * ``"auto"`` (default) — unattended: crop at the saddle behind the seaward-most
      dune when a *second* prominent dune follows it, otherwise fit the whole
      profile.  A no-op on single-dune / no-dune profiles, so it does not disturb
      a clean fit.
    * ``float`` (m) — an explicit window that many metres landward of the shoreline.
    * ``None`` — off; always fit the whole profile.

    Returns ``(metrics, idealized)``; ``idealized`` is ``None`` for
    INDETERMINATE fits.
    """
    x = np.asarray(x, dtype=float)
    zb = np.asarray(zb, dtype=float)
    n = len(x)
    BE = float(berm_elevation)

    if n < _MIN_NODES:
        return ProfileMetrics(), None

    dx = _dx_of(x)
    zs = _smooth(zb, dx)

    # Analysis window: cap the fit at a beach-scale span landward of the shoreline
    # so distant terrain does not distort the crop / upland / idealization.
    # Seaward (submerged) nodes are kept — they sit below the datum and are
    # ignored by feature detection anyway.
    cap = _analysis_cap(x, zs, dx, n, datum, max_feature_length)
    if cap is not None and cap >= _MIN_NODES:
        x, zb, zs, n = x[:cap], zb[:cap], zs[:cap], cap

    vol = volume_above_datum(x, zb, datum)
    # Landward crop: fit against the beach+dune+near-upland window, taking the
    # upland from the settled plateau rather than the fixed trailing window so
    # far-landward data past the dune doesn't distort UE / the idealization.
    upland_start, upland_end = _landward_crop(zs, dx, n)
    K = upland_end - upland_start + 1
    UE = float(np.median(zb[upland_start : upland_end + 1]))

    # --- Shoreline (located on smoothed z) ---
    x_shore, shore_idx = last_wet_dry_crossing(x, zs, datum, end=upland_start)
    if x_shore is None:
        if np.all(zs[:upland_start] > datum):
            x_shore, shore_idx = float(x[0]), 0
        else:
            m = ProfileMetrics(upland_elevation=UE, volume_above_datum=vol, n_upland_nodes=K)
            return m, None

    # --- Dune crest (smoothed location, raw elevation; plateau-aware) ---
    lo, hi = shore_idx, upland_start
    if ref is not None and not np.isnan(ref.dune_crest_x):
        rl = int(np.searchsorted(x, ref.dune_crest_x - _REF_WIN_M))
        rh = int(np.searchsorted(x, ref.dune_crest_x + _REF_WIN_M))
        lo, hi = max(lo, rl), min(hi, max(rh, rl + 1))
    if hi - lo < 1:
        lo, hi = shore_idx, upland_start

    reg = slice(lo, hi)
    zmax_s = float(np.max(zs[reg]))
    near = lo + np.where(zs[reg] >= zmax_s - _CREST_EPS)[0]
    crest_x = float(np.mean(x[near]))
    crest_idx = int(near[np.argmin(np.abs(x[near] - crest_x))])
    crest_x0, crest_x1 = float(x[near[0]]), float(x[near[-1]])
    top_width = crest_x1 - crest_x0
    # Crest elevation: on a flat (trapezoidal) top use the median top level, which
    # is robust to the single noise spike a raw max over the wide plateau would
    # catch; a peaked (triangular) top keeps its apex, since a median there would
    # be pulled below the true crest by the flanks.
    DE = float(np.median(zb[near])) if top_width >= _MIN_PLATEAU_M else float(np.max(zb[near]))

    morph_type = _classify(UE, BE, DE)

    (
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        berm_width,
        seaward_base,
        seaward_toe_idx,
        foreshore_slope,
    ) = _detect_berm_and_toe(x, zb, zs, dx, shore_idx, crest_idx, BE, morph_type)

    # --- Dune (only for LOW_BERM / LOW_UPLAND) ---
    if morph_type == MorphType.HIGH_UPLAND.value or crest_idx <= seaward_toe_idx:
        return _fit_no_dune(
            x,
            zb,
            zs,
            dx,
            datum,
            x_shore,
            morph_type,
            berm_present,
            berm_start_idx,
            berm_end_idx,
            berm_elev_meas,
            berm_width,
            shore_idx,
            seaward_toe_idx,
            upland_start,
            UE,
            vol,
            K,
            foreshore_slope,
        )

    # Landward toe: from crest, first descent to upland level.
    landward_toe_idx = upland_start
    for i in range(crest_idx, upland_start):
        if zs[i] >= UE + _LEVEL_TOL >= zs[i + 1]:
            landward_toe_idx = i + 1
            break

    ideal, geom, fit_quality = _select_dune_form(
        x,
        zb,
        dx,
        datum,
        BE,
        UE,
        x_shore,
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        seaward_base,
        seaward_toe_idx,
        crest_x,
        DE,
        upland_start,
        landward_toe_idx,
    )
    (
        DE,
        crest_x,
        top_width,
        dune_front_width,
        dune_back_width,
        dune_width,
        dune_front_relief,
        dune_back_relief,
        dune_front_slope,
        dune_back_slope,
    ) = geom

    # --- Scarp detection (steep run OR idealized residual, per zone) ---

    dune_scarp, dune_sc_h = _scarp(x, zb, ideal, seaward_toe_idx, crest_idx, datum)
    berm_zone_lo = shore_idx
    berm_zone_hi = seaward_toe_idx
    berm_scarp, berm_sc_h = _scarp(x, zb, ideal, berm_zone_lo, berm_zone_hi, datum)
    berm_scarp = berm_scarp or (not berm_present)
    _sc_heights = [h for h in (dune_sc_h, berm_sc_h) if not np.isnan(h)]
    scarp_height = float(max(_sc_heights)) if _sc_heights else _NAN

    m = ProfileMetrics(
        morph_type=morph_type,
        shoreline_x=x_shore,
        foreshore_slope=foreshore_slope,
        berm_elevation=berm_elev_meas,
        berm_width=berm_width,
        dune_crest_elevation=DE,
        dune_crest_x=crest_x,
        dune_width=dune_width,
        dune_front_width=dune_front_width,
        dune_back_width=dune_back_width,
        dune_top_width=top_width,
        dune_front_relief=dune_front_relief,
        dune_back_relief=dune_back_relief,
        dune_front_slope=dune_front_slope,
        dune_back_slope=dune_back_slope,
        upland_elevation=UE,
        volume_above_datum=vol,
        berm_scarp=berm_scarp,
        dune_scarp=dune_scarp,
        scarp_height=float(scarp_height),
        fit_quality=fit_quality,
        n_upland_nodes=K,
    )

    # --- Landmark ordering validation ---
    order = [shore_idx, berm_start_idx, seaward_toe_idx, crest_idx, landward_toe_idx, upland_start]
    if any(b < a for a, b in zip(order, order[1:])):
        m.morph_type = MorphType.INDETERMINATE.value
        return m, ideal

    return m, ideal
