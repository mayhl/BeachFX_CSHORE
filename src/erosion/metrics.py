"""Profile morphology metrics: BeachFX classification and an idealized fit.

Extracts 0-D metrics from a cross-shore profile snapshot and a piecewise-linear
**idealized profile** that can be re-rendered for debug overlays.

CSHORE x convention: x[0] is the most offshore node; x[-1] is the most landward
node (the upland).  zb[i] is the bed elevation at x[i].  Grids are assumed
uniform (CSHORE interpolates onto a uniform dx); a sanity check warns otherwise.

Terminology
-----------
* **elevation** — absolute z above the datum (e.g. berm/dune-crest/upland
  elevation).
* **height / relief** — vertical extent above a local base (e.g. dune front
  relief = crest elevation − berm elevation).

Three BeachFX morphology types (Tech Ref §7.3.1–7.3.2), classified on the
*design* berm elevation BE:

  HIGH_UPLAND  — DE ≤ UE
  LOW_BERM     — DE > UE, BE ≤ UE
  LOW_UPLAND   — DE > UE, BE > UE

The dune sits on a **two-sided base**: its seaward toe rests on the berm, its
landward toe on the upland (generally different elevations).  Dune widths are
measured as level-set footprints at those per-side bases, so the result does
not depend on whether the dune is peaked, flat-topped, or rounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Tunable constants (physical units; converted to node counts via dx)
# ---------------------------------------------------------------------------

_MIN_NODES: int = 10  # minimum profile length before attempting a fit
_SMOOTH_WIN_M: float = 9.0  # Savitzky–Golay window (m) for feature location
_SG_POLY: int = 2  # Savitzky–Golay polynomial order
_UPLAND_WIN_M: float = 20.0  # rear window (m) used to estimate upland elevation
_FLAT_SLOPE: float = 0.02  # |dz/dx| (m/m) qualifying as a flat berm
_BERM_MAX_DRIFT: float = 0.15  # max cumulative elevation drift (m) across a flat berm run
_BERM_MAX_ABOVE_BE: float = (
    0.5  # a berm sits no higher than design BE + this (rejects upland noise-flats)
)
_LEVEL_TOL: float = 0.15  # elevation tolerance (m) for level-set crossings
_REF_WIN_M: float = 30.0  # ± window (m) for ref-constrained crest search
_CREST_EPS: float = 0.05  # elevation band (m) defining the crest plateau
_MIN_PLATEAU_M: float = 3.0  # crest plateau wider than this idealizes as a flat top
_DUNE_MIN_PROMINENCE: float = 0.5  # crest must clear the upland by this to count as a dune
_BENCH_MIN_STEP: float = 0.5  # a mid-bench terrace must be set off from berm AND upland by this
_TOP_FRAC: float = 0.90  # fraction of relief defining the crest "top" width
_SCARP_SLOPE: float = 0.6  # |dz/dx| (~31°) flagging a near-vertical scarp face
_SCARP_MIN_H: float = 0.30  # minimum vertical extent (m) of a scarp run
_SCARP_RESID: float = 0.30  # idealized-vs-raw residual (m) flagging a scarp


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MorphType(str, Enum):
    HIGH_UPLAND = "HIGH_UPLAND"
    LOW_BERM = "LOW_BERM"
    LOW_UPLAND = "LOW_UPLAND"
    INDETERMINATE = "INDETERMINATE"


@dataclass
class ProfileMetrics:
    """0-D morphology metrics for a single (profile, snapshot) pair.

    All fields default to NaN/False so partial and indeterminate fits construct
    cleanly.  ``*_elevation`` are absolute (above datum); ``*_relief`` are
    vertical extents above a local base.
    """

    morph_type: str = MorphType.INDETERMINATE.value

    # Shoreline / beach
    shoreline_x: float = _NAN
    foreshore_slope: float = _NAN
    berm_elevation: float = _NAN  # measured flat-berm elevation
    berm_width: float = 0.0

    # Dune (NaN when no dune, i.e. HIGH_UPLAND / INDETERMINATE)
    dune_crest_elevation: float = _NAN  # DE: absolute crest elevation
    dune_crest_x: float = _NAN
    dune_width: float = 0.0  # total footprint (seaward toe → landward toe)
    dune_front_width: float = 0.0  # seaward toe → crest
    dune_back_width: float = 0.0  # crest → landward toe
    dune_top_width: float = 0.0  # near-crest plateau width
    dune_front_relief: float = _NAN  # DE − berm_elevation
    dune_back_relief: float = _NAN  # DE − upland_elevation
    dune_front_slope: float = _NAN  # ascending seaward face (dz/dx)
    dune_back_slope: float = _NAN  # descending landward face (|dz/dx|)

    # Upland
    upland_elevation: float = _NAN  # UE

    # Volume
    volume_above_datum: float = 0.0  # ∫ max(zb − datum, 0) dx  (m²/m)

    # Scarp diagnostics
    berm_scarp: bool = False
    dune_scarp: bool = False
    scarp_height: float = _NAN  # vertical extent of the cut face (m)

    # Fit diagnostics
    fit_quality: float = _NAN  # RMS(raw − idealized) over the z>0 region
    n_upland_nodes: int = 0


@dataclass
class IdealizedProfile:
    """Piecewise-linear idealization, rendered from a handful of landmark knots.

    Constructed entirely from scalars (no per-node arrays persisted).  Overlay
    the raw profile with ``evaluate(x)``; the residual ``zb − evaluate(x)`` is
    the scarp/debug signal.
    """

    knots_x: np.ndarray
    knots_z: np.ndarray

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(
            x,
            self.knots_x,
            self.knots_z,
            left=float(self.knots_z[0]),
            right=float(self.knots_z[-1]),
        )


# ---------------------------------------------------------------------------
# Shared geometry helpers (also used by profile.py / nourishment.py)
# ---------------------------------------------------------------------------


def last_wet_dry_crossing(
    x: np.ndarray,
    z: np.ndarray,
    datum: float = 0.0,
    end: int | None = None,
) -> tuple[float | None, int | None]:
    """Locate the last (most landward) wet→dry zero-crossing.

    Scans ``z[:end]`` for rising crossings of ``datum`` and linearly
    interpolates the crossing position.  Returns ``(x_shore, shore_idx)`` for
    the last such crossing (``shore_idx`` = first dry node after it), or
    ``(None, None)`` when none exists.
    """
    sub = z if end is None else z[:end]
    transitions = np.where(np.diff((sub > datum).astype(int)) > 0)[0]
    if len(transitions) == 0:
        return None, None
    ci = int(transitions[-1])
    z0, z1 = float(z[ci]), float(z[ci + 1])
    dz = z1 - z0
    frac = (datum - z0) / dz if abs(dz) > 1e-9 else 0.0
    return float(x[ci] + frac * (x[ci + 1] - x[ci])), ci + 1


def volume_above_datum(x: np.ndarray, z: np.ndarray, datum: float = 0.0) -> float:
    """Cross-shore area above ``datum``: ∫ max(z − datum, 0) dx  (m²/m)."""
    return float(np.trapezoid(np.maximum(z - datum, 0.0), x))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify(UE: float, BE: float, DE: float) -> str:
    # Require the crest to clear the upland by a real margin, so noise on a
    # rising HIGH_UPLAND back is not mistaken for a dune (extends BeachFX's bare
    # DE > UE test, which assumes clean parametric elevations).
    if not (DE > UE + _DUNE_MIN_PROMINENCE):
        return MorphType.HIGH_UPLAND.value
    return MorphType.LOW_BERM.value if BE <= UE else MorphType.LOW_UPLAND.value


def _linear_slope(x: np.ndarray, z: np.ndarray) -> float:
    """Least-squares slope dz/dx (0.0 for < 2 nodes)."""
    if len(x) < 2:
        return 0.0
    A = np.column_stack([x, np.ones(len(x))])
    slope = float(np.linalg.lstsq(A, z, rcond=None)[0][0])
    return slope


def _dx_of(x: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-3, atol=1e-6):
        log.warning("fit_profile: non-uniform x spacing; results may be unreliable")
    return dx


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def _smooth(z: np.ndarray, dx: float) -> np.ndarray:
    """Savitzky–Golay smoothing for feature *location*; raw z is used for magnitudes."""
    n = len(z)
    w = _odd(max(_SG_POLY + 2, int(round(_SMOOTH_WIN_M / dx))))
    if w > n:
        w = _odd(n)
    if w <= _SG_POLY or w < 3:
        return z.copy()
    return savgol_filter(z, w, _SG_POLY)


def _longest_flat_run(
    zs: np.ndarray, dx: float, lo: int, hi: int, z_ceiling: float = np.inf
) -> tuple[int, int]:
    """Longest run of near-flat nodes within [lo, hi].  Returns (start, end) idx
    (start == end when no flat run exists).

    A run needs both a low node-to-node slope AND bounded cumulative drift from
    its start, so a sustained gentle ramp (e.g. a rising HIGH_UPLAND back, whose
    per-node slope is below _FLAT_SLOPE) is not mistaken for one long berm.
    Runs whose median elevation exceeds ``z_ceiling`` are skipped, so a noise-flat
    high on the upland is not picked as the berm."""
    best = (lo, lo)
    best_len = 0
    i = lo
    while i < hi:
        j = i
        while (
            j < hi
            and abs(zs[j + 1] - zs[j]) / dx < _FLAT_SLOPE
            and abs(zs[j + 1] - zs[i]) <= _BERM_MAX_DRIFT
        ):
            j += 1
        if j - i > best_len and float(np.median(zs[i : j + 1])) <= z_ceiling:
            best_len, best = j - i, (i, j)
        i = max(j + 1, i + 1)
    return best


def _landward_crop(zs: np.ndarray, dx: float, n: int) -> tuple[int, int]:
    """Locate the settled upland = the *trailing* flat plateau that reaches the
    landward end.  Returns its (start, end) node indices; falls back to the rear
    window when the back is still rising at the end (no trailing plateau).

    Cropping the fit to end at the upland start keeps far-landward data — a flat
    plateau reached partway, or irrelevant nodes past the dune — from distorting
    the upland elevation and the idealized profile.  A mid-profile berm is never
    mistaken for the upland because it does not reach the end."""
    K = max(3, min(int(round(_UPLAND_WIN_M / dx)), n - 1))
    end = n - 1
    i = end
    while (
        i > 0
        and abs(zs[i - 1] - zs[i]) / dx < _FLAT_SLOPE
        and abs(zs[i - 1] - zs[end]) <= _BERM_MAX_DRIFT
    ):
        i -= 1
    if end - i >= K:
        return i, end
    return n - 1 - K, n - 1  # no trailing plateau: fall back to rear window


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_profile(
    x: np.ndarray,
    zb: np.ndarray,
    berm_elevation: float,
    datum: float = 0.0,
    ref: ProfileMetrics | None = None,
) -> tuple[ProfileMetrics, IdealizedProfile | None]:
    """Extract morphology metrics and an idealized profile from raw ``(x, zb)``.

    ``berm_elevation`` is the *design* berm crest BE used only for
    classification; the berm elevation is otherwise *measured* from the profile.
    ``ref`` (optional) constrains the dune-crest search to ±30 m around
    ``ref.dune_crest_x`` to avoid spurious post-storm peaks.

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

    # --- Berm: longest flat run between shore and crest (measured elevation) ---
    berm_start_idx, berm_end_idx = _longest_flat_run(
        zs, dx, shore_idx, crest_idx, z_ceiling=BE + _BERM_MAX_ABOVE_BE
    )
    if berm_end_idx > berm_start_idx:
        berm_elev_meas = float(np.median(zb[berm_start_idx : berm_end_idx + 1]))
        # Noise can clip the flat run short of the true seaward edge; extend it
        # down the foreshore to where the profile first reaches berm level (the
        # landward edge is extended to the dune toe below).  berm_width then
        # spans berm_start -> seaward_toe, computed once the toe is known.
        for i in range(berm_start_idx, shore_idx, -1):
            if zs[i] >= berm_elev_meas - _LEVEL_TOL:
                berm_start_idx = i
            else:
                break
        berm_present = True
    else:
        berm_elev_meas = _NAN
        berm_width = 0.0
        berm_present = False

    # Seaward toe = where the ascent toward the crest begins.  A berm's landward
    # edge is the starting guess, but noise can truncate the detected berm short
    # of the true toe; advance through any remaining berm-level flat so the dune
    # front isn't measured across (and idealized as a ramp over) leftover berm.
    seaward_base = berm_elev_meas if berm_present else BE
    if berm_present:
        seaward_toe_idx = berm_end_idx
        # Only extend toward a dune; a HIGH_UPLAND back has no dune toe, so the
        # berm must not creep into the rising upland.
        if morph_type != MorphType.HIGH_UPLAND.value:
            for i in range(berm_end_idx, crest_idx):
                if zs[i] <= berm_elev_meas + _LEVEL_TOL:
                    seaward_toe_idx = i
                else:
                    break
    else:
        seaward_toe_idx = shore_idx
        for i in range(shore_idx, crest_idx):
            if zs[i] <= seaward_base + _LEVEL_TOL <= zs[i + 1]:
                seaward_toe_idx = i
                break

    # Berm spans its seaward edge to the dune toe (the full berm-level flat).
    if berm_present:
        berm_width = float(x[seaward_toe_idx] - x[berm_start_idx])

    # --- Foreshore slope (shore → berm start) on raw z ---
    fs_hi = berm_start_idx if berm_present else seaward_toe_idx
    foreshore_slope = (
        max(0.0, _linear_slope(x[shore_idx : fs_hi + 1], zb[shore_idx : fs_hi + 1]))
        if fs_hi > shore_idx
        else _NAN
    )

    # --- Dune (only for LOW_BERM / LOW_UPLAND) ---
    if morph_type == MorphType.HIGH_UPLAND.value or crest_idx <= seaward_toe_idx:
        bench = (
            _mid_bench(x, zs, zb, dx, berm_end_idx + 1, upland_start, berm_elev_meas, UE)
            if berm_present
            else None
        )
        ideal = _ideal_no_dune(
            x,
            datum,
            x_shore,
            berm_present,
            berm_start_idx,
            berm_end_idx,
            berm_elev_meas,
            UE,
            upland_start,
            bench=bench,
        )
        # Berm-zone scarp detection — steep-run only, since a HIGH_UPLAND berm/
        # upland boundary is fuzzy and the residual path would false-flag a rising
        # back.  A berm-less beach still reads as scarped.
        berm_scarp, berm_sc_h = _scarp(
            x, zb, ideal, shore_idx, seaward_toe_idx, datum, residual=False
        )
        berm_scarp = berm_scarp or (not berm_present)
        m = ProfileMetrics(
            morph_type=morph_type,
            shoreline_x=x_shore,
            foreshore_slope=foreshore_slope,
            berm_elevation=berm_elev_meas,
            berm_width=berm_width,
            upland_elevation=UE,
            volume_above_datum=vol,
            berm_scarp=berm_scarp,
            scarp_height=berm_sc_h,
            n_upland_nodes=K,
        )
        m.fit_quality = _fit_quality(x, zb, ideal, datum)
        return m, ideal

    # Landward toe: from crest, first descent to upland level.
    landward_toe_idx = upland_start
    for i in range(crest_idx, upland_start):
        if zs[i] >= UE + _LEVEL_TOL >= zs[i + 1]:
            landward_toe_idx = i + 1
            break

    # --- Idealized dune form, then least-squares refined against the raw ---
    plateau = top_width >= _MIN_PLATEAU_M
    ideal = _ideal_dune(
        x,
        datum,
        x_shore,
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        crest_x0 if plateau else crest_x,
        DE,
        UE,
        upland_start,
        landward_toe_idx,
        seaward_toe_idx,
        crest_x_end=crest_x1 if plateau else None,
        seaward_toe_z=seaward_base,
    )
    ideal = _refine_dune(x, zb, ideal.knots_x, ideal.knots_z, datum)

    # Read dune geometry back from the refined form.
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
    ) = _dune_geom_from_knots(ideal.knots_x, ideal.knots_z, berm_elev_meas, BE, berm_present, UE)
    fit_quality = _fit_quality(x, zb, ideal, datum)

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


# ---------------------------------------------------------------------------
# Idealized-profile assembly + scarp/quality
# ---------------------------------------------------------------------------


def _knots_to_ideal(kx: list[float], kz: list[float]) -> IdealizedProfile:
    order = np.argsort(kx)
    return IdealizedProfile(np.asarray(kx)[order], np.asarray(kz)[order])


# The morphology class selects the idealized *function form*.  LOW_UPLAND and
# LOW_BERM share the dune form (they differ only in the BE-vs-UE classification);
# HIGH_UPLAND (and any degenerate crest-below-toe) uses the no-dune form.


def _mid_bench(x, zs, zb, dx, lo, hi, berm_elev, UE):
    """A flat terrace between the berm and the upland (a stepped HIGH_UPLAND back).
    Returns ``(start, end, elev)`` or ``None``.  Requires a genuinely flat shelf
    set off from both the berm and the upland by a real step, so a monotonic ramp
    — whose drift-bounded segments look flat — is not split into a phantom bench."""
    if hi - lo < 3:
        return None
    K = max(3, int(round(_MIN_PLATEAU_M / dx)))
    b0, b1 = _longest_flat_run(zs, dx, lo, hi, z_ceiling=UE - _LEVEL_TOL)
    if b1 - b0 < K:
        return None
    bench_z = float(np.median(zb[b0 : b1 + 1]))
    slope = abs(_linear_slope(x[b0 : b1 + 1], zs[b0 : b1 + 1]))
    if (
        slope < _FLAT_SLOPE / 3
        and bench_z - berm_elev >= _BENCH_MIN_STEP
        and UE - bench_z >= _BENCH_MIN_STEP
    ):
        return b0, b1, bench_z
    return None


def _ideal_no_dune(
    x,
    datum,
    x_shore,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev,
    UE,
    upland_start,
    bench=None,
) -> IdealizedProfile:
    """HIGH_UPLAND form: shore → berm → [optional mid-bench] → rise to the upland
    level at the crop point → flat.  Never ramps across the whole back to the far
    end.  ``bench`` is an optional ``(start, end, elev)`` intermediate terrace."""
    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        kx += [float(x[berm_start_idx]), float(x[berm_end_idx])]
        kz += [float(berm_elev), float(berm_elev)]
        if bench is not None:
            b0, b1, bz = bench
            kx += [float(x[b0]), float(x[b1])]
            kz += [float(bz), float(bz)]
    # Rise to the upland level at the crop point, then flat — for berm AND
    # berm-less backs (else a no-berm HIGH_UPLAND ramps straight to the far end
    # and misses the plateau).
    kx.append(float(x[upland_start]))
    kz.append(float(UE))
    kx.append(float(x[-1]))
    kz.append(float(UE))
    return _knots_to_ideal(kx, kz)


def _ideal_dune(
    x,
    datum,
    x_shore,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev,
    crest_x,
    crest_z,
    UE,
    upland_start,
    landward_toe_idx,
    seaward_toe_idx,
    crest_x_end=None,
    seaward_toe_z=None,
) -> IdealizedProfile:
    """LOW_UPLAND / LOW_BERM form: shore → berm → dune(front, crest[, plateau],
    back) → upland.  Two-sided base: seaward toe on the berm, landward toe on the
    upland (generally different elevations)."""
    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        # Extend the flat berm to the seaward toe (the true ascent start) so the
        # dune front ramps from the toe, not across leftover berm.
        berm_land = (
            seaward_toe_idx
            if seaward_toe_idx is not None and seaward_toe_idx > berm_end_idx
            else berm_end_idx
        )
        kx += [float(x[berm_start_idx]), float(x[berm_land])]
        kz += [float(berm_elev), float(berm_elev)]
    elif (
        seaward_toe_idx is not None
        and seaward_toe_z is not None
        and x_shore < float(x[seaward_toe_idx]) < float(crest_x)
    ):
        # No berm: knot at the dune's seaward toe so the front bends at the
        # foreshore→dune-front knee rather than one straight line to the crest.
        kx.append(float(x[seaward_toe_idx]))
        kz.append(float(seaward_toe_z))
    kx.append(float(crest_x))
    kz.append(float(crest_z))
    # Flat-topped (trapezoidal) dune: second crest knot at the plateau's landward
    # edge so the top idealizes flat, not as a single apex.
    if crest_x_end is not None and crest_x_end > crest_x:
        kx.append(float(crest_x_end))
        kz.append(float(crest_z))
    toe = landward_toe_idx if landward_toe_idx is not None else upland_start
    kx.append(float(x[toe]))
    kz.append(float(UE))
    kx.append(float(x[-1]))
    kz.append(float(UE))
    return _knots_to_ideal(kx, kz)


def _refine_dune(x, zb, kx, kz, datum, win: float = 8.0) -> IdealizedProfile:
    """Least-squares refine of the interior dune knots (seaward toe → landward
    toe) against the raw profile, starting from the detected idealization.

    Knot x stays within ±``win`` of detection; knot z is bounded to
    ``[datum, max(zb)]`` so the crest can only be pulled DOWN toward the data,
    never pushed above it — which corrects a noise-inflated sharp apex while
    leaving a rounded (gaussian) crest at its detected height.  Shoreline, berm,
    and upland knots stay fixed."""
    kx = np.asarray(kx, dtype=float).copy()
    kz = np.asarray(kz, dtype=float).copy()
    free = list(range(2, len(kx) - 1))  # interior: seaward toe → landward toe
    wet = zb > datum
    if not free or not np.any(wet) or float(np.max(zb)) <= datum:
        return IdealizedProfile(kx, kz)
    nf = len(free)
    z_ceiling = float(np.max(zb))

    def resid(p):
        kx2, kz2 = kx.copy(), kz.copy()
        kx2[free] = np.sort(p[:nf])
        kz2[free] = p[nf:]
        return (zb - np.interp(x, kx2, kz2, left=kz2[0], right=kz2[-1]))[wet]

    lo = np.concatenate([kx[free] - win, np.full(nf, datum)])
    hi = np.concatenate([kx[free] + win, np.full(nf, z_ceiling)])
    p0 = np.clip(np.concatenate([kx[free], kz[free]]), lo, hi)
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=2000)
        kx[free] = np.sort(sol.x[:nf])
        kz[free] = sol.x[nf:]
    except Exception:
        pass
    return IdealizedProfile(kx, kz)


def _dune_geom_from_knots(kx, kz, berm_elev, BE, berm_present, UE):
    """Read dune geometry (crest, widths, slopes, reliefs) back from the refined
    idealized knots.  The crest is the highest knot (a plateau spans several)."""
    DE = float(np.max(kz))
    crest_ks = np.where(kz >= DE - 1e-9)[0]
    c0, c1 = int(crest_ks[0]), int(crest_ks[-1])
    crest_x = float(np.mean(kx[crest_ks]))
    top_width = float(kx[c1] - kx[c0])
    st, lt = max(c0 - 1, 0), min(c1 + 1, len(kx) - 1)
    st_x, st_z = float(kx[st]), float(kz[st])
    lt_x, lt_z = float(kx[lt]), float(kz[lt])
    seaward_base = berm_elev if berm_present else BE
    front_slope = max(0.0, (DE - st_z) / max(float(kx[c0]) - st_x, 1e-9))
    back_slope = max(0.0, (DE - lt_z) / max(lt_x - float(kx[c1]), 1e-9))
    return (
        DE,
        crest_x,
        top_width,
        crest_x - st_x,
        lt_x - crest_x,
        lt_x - st_x,  # front/back/total width
        DE - seaward_base,
        DE - UE,  # front/back relief
        front_slope,
        back_slope,
    )


def _fit_quality(x, zb, ideal: IdealizedProfile, datum: float) -> float:
    mask = zb > datum
    if not np.any(mask):
        return _NAN
    resid = zb[mask] - ideal.evaluate(x[mask])
    return float(np.sqrt(np.mean(resid**2)))


def _scarp(
    x, zb, ideal: IdealizedProfile, lo: int, hi: int, datum: float, residual: bool = True
) -> tuple[bool, float]:
    """Flag a scarp in [lo, hi] via a steep run OR a large idealized residual.
    Returns (present, scarp_height).  Set ``residual=False`` to use the steep-run
    signal only — needed where the idealized base is unreliable (a fuzzy
    HIGH_UPLAND berm/upland boundary would otherwise read a rising back as a
    residual "scarp")."""
    if hi - lo < 1:
        return False, _NAN
    xs, zs = x[lo : hi + 1], zb[lo : hi + 1]
    dz = np.abs(np.diff(zs))
    dxs = np.diff(xs)
    steep = dz / np.maximum(dxs, 1e-9) >= _SCARP_SLOPE
    steep_h = float(np.sum(dz[steep])) if np.any(steep) else 0.0
    resid_max = float(np.max(np.abs(zs - ideal.evaluate(xs)))) if residual else 0.0
    present = (steep_h >= _SCARP_MIN_H) or (resid_max >= _SCARP_RESID)
    return present, max(steep_h, resid_max) if present else _NAN
