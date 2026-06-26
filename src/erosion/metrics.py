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
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Tunable constants (physical units; converted to node counts via dx)
# ---------------------------------------------------------------------------

_MIN_NODES:      int   = 10     # minimum profile length before attempting a fit
_SMOOTH_WIN_M:   float = 9.0    # Savitzky–Golay window (m) for feature location
_SG_POLY:        int   = 2      # Savitzky–Golay polynomial order
_UPLAND_WIN_M:   float = 20.0   # rear window (m) used to estimate upland elevation
_FLAT_SLOPE:     float = 0.02   # |dz/dx| (m/m) qualifying as a flat berm
_LEVEL_TOL:      float = 0.15   # elevation tolerance (m) for level-set crossings
_REF_WIN_M:      float = 30.0   # ± window (m) for ref-constrained crest search
_CREST_EPS:      float = 0.05   # elevation band (m) defining the crest plateau
_TOP_FRAC:       float = 0.90   # fraction of relief defining the crest "top" width
_SCARP_SLOPE:    float = 0.6    # |dz/dx| (~31°) flagging a near-vertical scarp face
_SCARP_MIN_H:    float = 0.30   # minimum vertical extent (m) of a scarp run
_SCARP_RESID:    float = 0.30   # idealized-vs-raw residual (m) flagging a scarp


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class MorphType(str, Enum):
    HIGH_UPLAND   = "HIGH_UPLAND"
    LOW_BERM      = "LOW_BERM"
    LOW_UPLAND    = "LOW_UPLAND"
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
    berm_elevation: float = _NAN          # measured flat-berm elevation
    berm_width: float = 0.0

    # Dune (NaN when no dune, i.e. HIGH_UPLAND / INDETERMINATE)
    dune_crest_elevation: float = _NAN    # DE: absolute crest elevation
    dune_crest_x: float = _NAN
    dune_width: float = 0.0               # total footprint (seaward toe → landward toe)
    dune_front_width: float = 0.0         # seaward toe → crest
    dune_back_width: float = 0.0          # crest → landward toe
    dune_top_width: float = 0.0           # near-crest plateau width
    dune_front_relief: float = _NAN       # DE − berm_elevation
    dune_back_relief: float = _NAN        # DE − upland_elevation
    dune_front_slope: float = _NAN        # ascending seaward face (dz/dx)
    dune_back_slope: float = _NAN         # descending landward face (|dz/dx|)

    # Upland
    upland_elevation: float = _NAN        # UE

    # Volume
    volume_above_datum: float = 0.0       # ∫ max(zb − datum, 0) dx  (m²/m)

    # Scarp diagnostics
    berm_scarp: bool = False
    dune_scarp: bool = False
    scarp_height: float = _NAN            # vertical extent of the cut face (m)

    # Fit diagnostics
    fit_quality: float = _NAN             # RMS(raw − idealized) over the z>0 region
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
        return np.interp(x, self.knots_x, self.knots_z,
                         left=float(self.knots_z[0]), right=float(self.knots_z[-1]))


# ---------------------------------------------------------------------------
# Shared geometry helpers (also used by profile.py / nourishment.py)
# ---------------------------------------------------------------------------

def last_wet_dry_crossing(
    x: np.ndarray, z: np.ndarray, datum: float = 0.0, end: int | None = None,
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
    if not (DE > UE):
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


def _longest_flat_run(zs: np.ndarray, dx: float, lo: int, hi: int) -> tuple[int, int]:
    """Longest run of near-flat nodes within [lo, hi].  Returns (start, end) idx
    (start == end when no flat run exists)."""
    best = (lo, lo)
    best_len = 0
    i = lo
    while i < hi:
        j = i
        while j < hi and abs(zs[j + 1] - zs[j]) / dx < _FLAT_SLOPE:
            j += 1
        if j - i > best_len:
            best_len, best = j - i, (i, j)
        i = max(j + 1, i + 1)
    return best


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
    x  = np.asarray(x,  dtype=float)
    zb = np.asarray(zb, dtype=float)
    n  = len(x)
    BE = float(berm_elevation)

    if n < _MIN_NODES:
        return ProfileMetrics(), None

    dx  = _dx_of(x)
    zs  = _smooth(zb, dx)
    vol = volume_above_datum(x, zb, datum)
    K   = max(3, int(round(_UPLAND_WIN_M / dx)))
    K   = min(K, n - 1)
    UE  = float(np.median(zb[-K:]))
    upland_start = n - K

    # --- Shoreline (located on smoothed z) ---
    x_shore, shore_idx = last_wet_dry_crossing(x, zs, datum, end=upland_start)
    if x_shore is None:
        if np.all(zs[:upland_start] > datum):
            x_shore, shore_idx = float(x[0]), 0
        else:
            m = ProfileMetrics(upland_elevation=UE, volume_above_datum=vol,
                               n_upland_nodes=K)
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
    DE = float(np.max(zb[near]))
    top_width = float(x[near[-1]] - x[near[0]])

    morph_type = _classify(UE, BE, DE)

    # --- Berm: longest flat run between shore and crest (measured elevation) ---
    berm_start_idx, berm_end_idx = _longest_flat_run(zs, dx, shore_idx, crest_idx)
    if berm_end_idx > berm_start_idx:
        berm_elev_meas = float(np.median(zb[berm_start_idx:berm_end_idx + 1]))
        berm_width = float(x[berm_end_idx] - x[berm_start_idx])
        berm_present = True
    else:
        berm_elev_meas = _NAN
        berm_width = 0.0
        berm_present = False

    # Seaward toe = berm landward edge if a berm exists, else design-BE crossing.
    seaward_base = berm_elev_meas if berm_present else BE
    if berm_present:
        seaward_toe_idx = berm_end_idx
    else:
        seaward_toe_idx = shore_idx
        for i in range(shore_idx, crest_idx):
            if zs[i] <= seaward_base + _LEVEL_TOL <= zs[i + 1]:
                seaward_toe_idx = i
                break

    # --- Foreshore slope (shore → berm start) on raw z ---
    fs_hi = berm_start_idx if berm_present else seaward_toe_idx
    foreshore_slope = (max(0.0, _linear_slope(x[shore_idx:fs_hi + 1], zb[shore_idx:fs_hi + 1]))
                       if fs_hi > shore_idx else _NAN)

    # --- Dune (only for LOW_BERM / LOW_UPLAND) ---
    if morph_type == MorphType.HIGH_UPLAND.value or crest_idx <= seaward_toe_idx:
        m = ProfileMetrics(
            morph_type=morph_type, shoreline_x=x_shore, foreshore_slope=foreshore_slope,
            berm_elevation=berm_elev_meas, berm_width=berm_width,
            upland_elevation=UE, volume_above_datum=vol,
            berm_scarp=not berm_present, n_upland_nodes=K,
        )
        ideal = _build_ideal(x, datum, x_shore, berm_present, berm_start_idx,
                             berm_end_idx, berm_elev_meas, None, None, UE, upland_start)
        m.fit_quality = _fit_quality(x, zb, ideal, datum)
        return m, ideal

    # Landward toe: from crest, first descent to upland level.
    landward_toe_idx = upland_start
    for i in range(crest_idx, upland_start):
        if zs[i] >= UE + _LEVEL_TOL >= zs[i + 1]:
            landward_toe_idx = i + 1
            break

    dune_front_width = float(x[crest_idx] - x[seaward_toe_idx])
    dune_back_width  = float(x[landward_toe_idx] - x[crest_idx])
    dune_width       = float(x[landward_toe_idx] - x[seaward_toe_idx])
    dune_front_relief = DE - (berm_elev_meas if berm_present else BE)
    dune_back_relief  = DE - UE
    dune_front_slope = max(0.0, _linear_slope(
        x[seaward_toe_idx:crest_idx + 1], zb[seaward_toe_idx:crest_idx + 1]))
    dune_back_slope = max(0.0, -_linear_slope(
        x[crest_idx:landward_toe_idx + 1], zb[crest_idx:landward_toe_idx + 1]))

    # --- Scarp detection (steep run OR idealized residual, per zone) ---
    ideal = _build_ideal(x, datum, x_shore, berm_present, berm_start_idx, berm_end_idx,
                         berm_elev_meas, crest_x, DE, UE, upland_start,
                         landward_toe_idx=landward_toe_idx, seaward_toe_idx=seaward_toe_idx)
    fit_quality = _fit_quality(x, zb, ideal, datum)

    dune_scarp, dune_sc_h = _scarp(x, zb, ideal, seaward_toe_idx, crest_idx, datum)
    berm_zone_lo = shore_idx
    berm_zone_hi = seaward_toe_idx
    berm_scarp, berm_sc_h = _scarp(x, zb, ideal, berm_zone_lo, berm_zone_hi, datum)
    berm_scarp = berm_scarp or (not berm_present)
    scarp_height = np.nanmax([dune_sc_h, berm_sc_h]) if (dune_scarp or berm_scarp) else _NAN

    m = ProfileMetrics(
        morph_type=morph_type,
        shoreline_x=x_shore, foreshore_slope=foreshore_slope,
        berm_elevation=berm_elev_meas, berm_width=berm_width,
        dune_crest_elevation=DE, dune_crest_x=crest_x,
        dune_width=dune_width, dune_front_width=dune_front_width,
        dune_back_width=dune_back_width, dune_top_width=top_width,
        dune_front_relief=dune_front_relief, dune_back_relief=dune_back_relief,
        dune_front_slope=dune_front_slope, dune_back_slope=dune_back_slope,
        upland_elevation=UE, volume_above_datum=vol,
        berm_scarp=berm_scarp, dune_scarp=dune_scarp, scarp_height=float(scarp_height),
        fit_quality=fit_quality, n_upland_nodes=K,
    )

    # --- Landmark ordering validation ---
    order = [shore_idx, berm_start_idx, seaward_toe_idx, crest_idx,
             landward_toe_idx, upland_start]
    if any(b < a for a, b in zip(order, order[1:])):
        m.morph_type = MorphType.INDETERMINATE.value
        return m, ideal

    return m, ideal


# ---------------------------------------------------------------------------
# Idealized-profile assembly + scarp/quality
# ---------------------------------------------------------------------------

def _build_ideal(
    x, datum, x_shore, berm_present, berm_start_idx, berm_end_idx, berm_elev,
    crest_x, crest_z, UE, upland_start, landward_toe_idx=None, seaward_toe_idx=None,
) -> IdealizedProfile:
    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        kx += [float(x[berm_start_idx]), float(x[berm_end_idx])]
        kz += [float(berm_elev), float(berm_elev)]
    if crest_x is not None:
        kx.append(float(crest_x)); kz.append(float(crest_z))
        toe = landward_toe_idx if landward_toe_idx is not None else upland_start
        kx.append(float(x[toe])); kz.append(float(UE))
    kx.append(float(x[-1])); kz.append(float(UE))
    order = np.argsort(kx)
    return IdealizedProfile(np.asarray(kx)[order], np.asarray(kz)[order])


def _fit_quality(x, zb, ideal: IdealizedProfile, datum: float) -> float:
    mask = zb > datum
    if not np.any(mask):
        return _NAN
    resid = zb[mask] - ideal.evaluate(x[mask])
    return float(np.sqrt(np.mean(resid ** 2)))


def _scarp(x, zb, ideal: IdealizedProfile, lo: int, hi: int, datum: float
           ) -> tuple[bool, float]:
    """Flag a scarp in [lo, hi] via a steep run OR a large idealized residual.
    Returns (present, scarp_height)."""
    if hi - lo < 1:
        return False, _NAN
    xs, zs = x[lo:hi + 1], zb[lo:hi + 1]
    dz = np.abs(np.diff(zs))
    dxs = np.diff(xs)
    steep = dz / np.maximum(dxs, 1e-9) >= _SCARP_SLOPE
    steep_h = float(np.sum(dz[steep])) if np.any(steep) else 0.0
    resid = np.abs(zs - ideal.evaluate(xs))
    resid_max = float(np.max(resid))
    present = (steep_h >= _SCARP_MIN_H) or (resid_max >= _SCARP_RESID)
    return present, max(steep_h, resid_max) if present else _NAN
