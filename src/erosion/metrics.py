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
from scipy.signal import find_peaks, savgol_filter

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
_GAUSS_MIN_NODES: int = 5  # minimum dune-region nodes before attempting a gaussian fit
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


@dataclass
class _FormFit:
    """A fitted dune-form candidate carried through form selection: the idealized
    profile, its RMS misfit, its free-parameter count ``k`` (crest-shape DOF, used
    by the BIC), and the 10-tuple dune geometry read back from it."""

    ideal: IdealizedProfile
    rms: float
    k: int
    geom: tuple


def _bic(rms: float, n: int, k: int) -> float:
    """Bayesian Information Criterion for a least-squares form fit under a Gaussian
    error model: ``n·ln(σ̂²) + k·ln n`` with ``σ̂² = rms²`` (dropped additive
    constants are common to every candidate over the same ``n`` points, so they
    cancel in the argmin).  Lower wins.  ``k`` is the crest-shape DOF that
    distinguishes the forms (triangle 2 < trapezoid 3 < gaussian 4); the base/toe
    knots they share add a constant that cancels."""
    if n <= 0:
        return float("inf")
    var = max(rms * rms, 1e-12)  # floor a (near-)perfect fit off −∞
    return float(n * np.log(var) + k * np.log(n))


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


def _analysis_cap(
    x: np.ndarray,
    zs: np.ndarray,
    dx: float,
    n: int,
    datum: float,
    mode: float | str | None,
) -> int | None:
    """Landward node index to crop the fit at (exclusive), or ``None`` for no crop.

    ``mode`` is ``"auto"`` (crop behind the seaward-most dune when a second dune
    follows — see ``_auto_cap``), an explicit window length in metres landward of
    the shoreline, or ``None`` (never crop)."""
    if mode is None:
        return None
    _, s0 = last_wet_dry_crossing(x, zs, datum)
    if s0 is None:
        return None
    if mode == "auto":
        return _auto_cap(zs, dx, n, s0, datum)
    return min(n, s0 + int(round(float(mode) / dx)) + 1)


def _auto_cap(zs: np.ndarray, dx: float, n: int, s0: int, datum: float) -> int | None:
    """Auto analysis window: when a second dune rises landward of the seaward-most
    one *across a real swale*, crop at the saddle between them (plus an upland
    margin) so the fit sees a single beach+dune.  Returns ``None`` (no crop) for
    the single-dune / no-dune case, left untouched.

    Two peaks count as *separate* dunes only when the saddle between them falls at
    least halfway from the lower crest down toward the datum — a genuine swale.
    Minor dips on one broad or bumpy dune complex (and the two shoulders of a
    single flat-topped dune) do not clear that bar, so they are not split."""
    dry = zs[s0:]
    peaks, _ = find_peaks(dry, prominence=_DUNE_MIN_PROMINENCE)
    margin = int(round(_UPLAND_WIN_M / dx))  # keep some upland to measure UE
    for a, b in zip(peaks, peaks[1:]):
        a, b = int(a), int(b)
        saddle = a + int(np.argmin(dry[a : b + 1]))  # low point between the pair
        lower = min(dry[a], dry[b])
        valley = lower - dry[saddle]
        if valley >= _DUNE_MIN_PROMINENCE and valley >= 0.5 * (lower - datum):
            cap = s0 + min(saddle + margin, (saddle + b) // 2)  # never reach dune 2
            return min(n, cap + 1)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    # --- Idealized dune form: fit candidate forms, select by BIC ---
    # A triangular apex, a trapezoidal flat top (broad/multi-crest massif), and a
    # skew-gaussian bump (rounded crest, independent front/back slopes).  Each is
    # least-squares fit against the raw; the winner minimizes the BIC, which trades
    # misfit against the crest-shape DOF (triangle 2 < trapezoid 3 < gaussian 4) so
    # a genuinely peaked dune is not given a spurious flat top, and a rounded dune
    # is not forced into a sharp apex, without a hand-tuned RMS margin.
    def _knot_candidate(cx, cz, cx_end, k):
        idl = _ideal_dune(
            x,
            datum,
            x_shore,
            berm_present,
            berm_start_idx,
            berm_end_idx,
            berm_elev_meas,
            cx,
            cz,
            UE,
            upland_start,
            landward_toe_idx,
            seaward_toe_idx,
            crest_x_end=cx_end,
            seaward_toe_z=seaward_base,
        )
        # Keep a trapezoid's two top knots at a shared elevation through the fit.
        flat_top = None
        if cx_end is not None:
            tops = np.where(np.abs(idl.knots_z - cz) < 1e-9)[0]
            if len(tops) == 2:
                flat_top = (int(tops[0]), int(tops[1]))
        idl = _refine_dune(x, zb, idl.knots_x, idl.knots_z, datum, flat_top=flat_top)
        geom = _dune_geom_from_knots(idl.knots_x, idl.knots_z, berm_elev_meas, BE, berm_present, UE)
        return _FormFit(idl, _fit_quality(x, zb, idl, datum), k, geom)

    nwet = int(np.sum(zb > datum))
    cands = [_knot_candidate(crest_x, DE, None, 2)]  # triangle

    # Trapezoidal candidate: the broad near-crest band (a relief fraction below the
    # apex), leveled at its median so a bumpy multi-crest top idealizes flat.
    top_tol = max(_CREST_EPS, (1.0 - _TOP_FRAC) * (DE - seaward_base))
    band = np.arange(seaward_toe_idx, landward_toe_idx + 1)
    band = band[zb[band] >= DE - top_tol]
    if len(band) >= 2 and (x[band[-1]] - x[band[0]]) >= _MIN_PLATEAU_M:
        cands.append(
            _knot_candidate(float(x[band[0]]), float(np.median(zb[band])), float(x[band[-1]]), 3)
        )

    gauss = _gaussian_candidate(
        x, zb, datum, dx, seaward_toe_idx, landward_toe_idx, crest_x, DE,
        seaward_base, UE, berm_present, berm_start_idx, berm_end_idx,
        berm_elev_meas, x_shore,
    )
    if gauss is not None:
        cands.append(gauss)

    best = min(cands, key=lambda c: _bic(c.rms, nwet, c.k))
    ideal = best.ideal
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
    ) = best.geom
    fit_quality = best.rms

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


def _refine_dune(x, zb, kx, kz, datum, win: float = 8.0, flat_top=None) -> IdealizedProfile:
    """Least-squares refine of the interior dune knots (seaward toe → landward
    toe) against the raw profile, starting from the detected idealization.

    Knot x stays within ±``win`` of detection; knot z is bounded to
    ``[datum, max(zb)]`` so the crest can only be pulled DOWN toward the data,
    never pushed above it — which corrects a noise-inflated sharp apex while
    leaving a rounded (gaussian) crest at its detected height.  Shoreline, berm,
    and upland knots stay fixed.  ``flat_top=(i, j)`` ties knot ``j``'s elevation
    to knot ``i``'s so a trapezoidal top stays flat (a genuine plateau, with a
    well-defined crest and width) instead of tilting into a general quadrilateral
    under the fit."""
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
        if flat_top is not None:
            kz2[flat_top[1]] = kz2[flat_top[0]]
        return (zb - np.interp(x, kx2, kz2, left=kz2[0], right=kz2[-1]))[wet]

    lo = np.concatenate([kx[free] - win, np.full(nf, datum)])
    hi = np.concatenate([kx[free] + win, np.full(nf, z_ceiling)])
    p0 = np.clip(np.concatenate([kx[free], kz[free]]), lo, hi)
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=2000)
        kx[free] = np.sort(sol.x[:nf])
        kz[free] = sol.x[nf:]
        if flat_top is not None:
            kz[flat_top[1]] = kz[flat_top[0]]
    except Exception:
        pass
    return IdealizedProfile(kx, kz)


def _gaussian_candidate(
    x,
    zb,
    datum,
    dx,
    seaward_toe_idx,
    landward_toe_idx,
    crest_x,
    DE,
    seaward_base,
    UE,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev_meas,
    x_shore,
) -> _FormFit | None:
    """Skew-gaussian dune candidate: a bump ``A·exp(−½((x−x₀)/σ)²)`` with
    *independent* front/back scales ``σ_f, σ_b`` riding on the linear two-sided toe
    baseline (seaward toe on the berm, landward toe on the upland).  Bounded
    least-squares against the raw profile over the dune region.  The fitted curve
    is sampled at the profile nodes into a piecewise-linear ``IdealizedProfile`` (so
    ``evaluate``/scarp/viz/golden are unchanged), with the outer shore/berm/upland
    knots stitched on.  Returns ``None`` if the region is too short or the fit
    fails.  Unlike the knot forms, the smooth flanks do not trip the residual scarp,
    so a rounded dune stops false-flagging a front scarp."""
    lo, hi = int(seaward_toe_idx), int(landward_toe_idx)
    if hi - lo + 1 < _GAUSS_MIN_NODES:
        return None
    xs, xl = float(x[lo]), float(x[hi])
    if xl <= xs:
        return None
    reg = slice(lo, hi + 1)
    xr, zr = x[reg], zb[reg]
    span = xl - xs

    def baseline(xx):  # linear two-sided base connecting the toes
        return seaward_base + (UE - seaward_base) * (xx - xs) / span

    base_r = baseline(xr)

    def curve(p):
        A, x0, sf, sb = p
        sig = np.where(xr <= x0, sf, sb)
        return base_r + A * np.exp(-0.5 * ((xr - x0) / np.maximum(sig, 1e-6)) ** 2)

    zceil = float(np.max(zb))
    A0 = max(DE - float(baseline(crest_x)), 0.1)
    p0 = [A0, float(crest_x), max((crest_x - xs) / 2.0, dx), max((xl - crest_x) / 2.0, dx)]
    lo_b = [0.0, xs, dx, dx]
    hi_b = [max(zceil - min(seaward_base, UE), 0.1), xl, span, span]
    p0 = np.clip(p0, lo_b, hi_b)
    try:
        sol = least_squares(lambda p: curve(p) - zr, p0, bounds=(lo_b, hi_b), max_nfev=2000)
    except Exception:
        return None
    A, x0, sf, sb = (float(v) for v in sol.x)
    # Keep the crest under the data, like the knot refinement's z-ceiling.
    if float(baseline(x0)) + A > zceil:
        A = zceil - float(baseline(x0))

    # Sample the fitted curve to knots; pin the toe endpoints to the baseline so
    # the dune segment meets the berm/upland cleanly.
    z_samp = baseline(xr) + A * np.exp(
        -0.5 * ((xr - x0) / np.maximum(np.where(xr <= x0, sf, sb), 1e-6)) ** 2
    )
    z_samp = z_samp.copy()
    z_samp[0], z_samp[-1] = seaward_base, UE

    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        berm_land = seaward_toe_idx if seaward_toe_idx > berm_end_idx else berm_end_idx
        kx += [float(x[berm_start_idx]), float(x[berm_land])]
        kz += [float(berm_elev_meas), float(berm_elev_meas)]
    kx += [float(v) for v in xr]
    kz += [float(v) for v in z_samp]
    kx.append(float(x[-1]))
    kz.append(float(UE))
    ideal = _knots_to_ideal(kx, kz)

    rms = _fit_quality(x, zb, ideal, datum)
    geom = _gaussian_geom(A, x0, sf, sb, xs, xl, seaward_base, UE, baseline)
    return _FormFit(ideal, rms, 4, geom)


def _gaussian_geom(A, x0, sf, sb, xs, xl, seaward_base, UE, baseline):
    """Dune geometry from the skew-gaussian parameters (crest from ``x₀/A``; widths
    are toe→crest footprints, slopes toe-to-crest secants — the same shape-agnostic
    definitions the knot forms use).  A gaussian has no plateau, so top width = 0."""
    crest_x = float(np.clip(x0, xs, xl))
    DE = float(baseline(crest_x) + A)
    front_width = crest_x - xs
    back_width = xl - crest_x
    front_relief = DE - seaward_base
    back_relief = DE - UE
    return (
        DE,
        crest_x,
        0.0,  # top width (peaked, no plateau)
        front_width,
        back_width,
        xl - xs,
        front_relief,
        back_relief,
        max(0.0, front_relief / max(front_width, 1e-9)),
        max(0.0, back_relief / max(back_width, 1e-9)),
    )


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


# ---------------------------------------------------------------------------
# Profile review — flag fits that a human should classify by hand
# ---------------------------------------------------------------------------

_REVIEW_RMS_TOL: float = 0.30  # idealized RMS (m) above which a fit is low-confidence
_REVIEW_MARGIN: float = 0.25  # elevation band (m) around a class threshold = marginal


class ReviewFlag(str, Enum):
    """Reasons a fitted profile is flagged for manual classification."""

    INDETERMINATE = "indeterminate"  # fit failed / landmark ordering inconsistent
    MULTI_DUNE = "multi_dune"  # >=2 prominent dunes: multi-dune vs eroded-dune ambiguity
    LOW_FIT_CONFIDENCE = "low_fit_confidence"  # idealized RMS above tolerance
    MARGINAL_CLASS = "marginal_class"  # classification sits on a threshold (flips under noise)
    AMBIGUOUS_SHORELINE = "ambiguous_shoreline"  # more than one wet->dry crossing


@dataclass
class ProfileReview:
    """Outcome of :func:`review_profile`.  ``needs_review`` is true when any flag
    fired; ``flags`` lists the reasons (``ReviewFlag`` values)."""

    needs_review: bool
    flags: list[str]
    fit_quality: float


def review_profile(
    x: np.ndarray,
    zb: np.ndarray,
    m: ProfileMetrics,
    ideal: IdealizedProfile | None,
    datum: float = 0.0,
    rms_tol: float = _REVIEW_RMS_TOL,
) -> ProfileReview:
    """Flag a fitted profile whose automatic classification is unreliable.

    Real survey profiles are not always the clean single-dune form the fitter
    idealizes; this QC pass routes the ambiguous ones to a human instead of
    trusting a low-confidence fit.  Takes the raw profile and its fit
    (``m``, ``ideal`` from :func:`fit_profile`).
    """
    x = np.asarray(x, dtype=float)
    zb = np.asarray(zb, dtype=float)

    if ideal is None or m.morph_type in ("", MorphType.INDETERMINATE.value):
        return ProfileReview(True, [ReviewFlag.INDETERMINATE.value], float(m.fit_quality))

    flags: list[str] = []
    dx = _dx_of(x)
    zs = _smooth(zb, dx)
    _, s0 = last_wet_dry_crossing(x, zs, datum)
    if s0 is not None:
        # Multiple prominent dunes: which is the design dune is a judgment call.
        peaks, _ = find_peaks(zs[s0:], prominence=_DUNE_MIN_PROMINENCE)
        if len(peaks) >= 2:
            flags.append(ReviewFlag.MULTI_DUNE.value)
        # More than one wet->dry crossing seaward of the upland = ambiguous shoreline.
        crossings = np.where(np.diff((zs[: s0 + 1] > datum).astype(int)) > 0)[0]
        if len(crossings) > 1:
            flags.append(ReviewFlag.AMBIGUOUS_SHORELINE.value)

    if not np.isnan(m.fit_quality) and m.fit_quality > rms_tol:
        flags.append(ReviewFlag.LOW_FIT_CONFIDENCE.value)

    # Marginal classification: the crest barely clears the upland, so the
    # dune / no-dune decision (HIGH_UPLAND vs LOW_*) would flip under a hair of
    # noise.  (The LOW_BERM/LOW_UPLAND split turns on the per-reach *design* berm,
    # not measured here, so it is left to a caller that has that value.)
    UE, DE = m.upland_elevation, m.dune_crest_elevation
    if not np.isnan(DE) and abs((DE - UE) - _DUNE_MIN_PROMINENCE) < _REVIEW_MARGIN:
        flags.append(ReviewFlag.MARGINAL_CLASS.value)

    return ProfileReview(bool(flags), flags, float(m.fit_quality))
