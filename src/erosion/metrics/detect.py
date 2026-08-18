"""Feature-detection internals: smoothing, landward crop, analysis-window capping,
and berm/toe/foreshore location."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .types import (
    _BERM_MAX_ABOVE_BE,
    _BERM_MAX_DRIFT,
    _DUNE_MIN_PROMINENCE,
    _FLAT_SLOPE,
    _LEVEL_TOL,
    _NAN,
    _SG_POLY,
    _SMOOTH_WIN_M,
    _UPLAND_WIN_M,
    MorphType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared cross-shore geometry helpers (also used by profile.py / nourishment/)
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


def erosion_volume_above_msl(
    x: np.ndarray, z_ref: np.ndarray, z_now: np.ndarray, msl: float = 0.0
) -> float:
    """Subaerial erosion deficit above ``msl``: ∫ max(0, ref − now) dx  (m²/m).

    ``ref`` (target) and ``now`` (current) are both clipped at ``msl`` so only
    the dry-beach zone counts, then differenced per station and floored at 0 —
    accreted sections (now > ref) contribute nothing and never offset a deficit
    elsewhere.  Erosion-only, unlike a difference of two ``volume_above_datum``
    integrals (which lets surplus cancel deficit).
    """
    ref = np.maximum(z_ref, msl)
    now = np.maximum(z_now, msl)
    return float(np.trapezoid(np.maximum(ref - now, 0.0), x))


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
    """Savitzky–Golay smoothing for feature *location*; raw z is used for magnitudes.

    The floor must survive ``_odd``: a poly-2 filter through 3 points is an exact
    fit, so a floor of ``_SG_POLY + 2`` (= 4, decremented to 3 by ``_odd``) turned
    smoothing OFF for every dx >= ~2 m -- exactly the real 10 ft survey spacing.
    ``_SG_POLY + 3`` keeps a genuine 5-node window on coarse grids.
    """
    n = len(z)
    w = _odd(max(_SG_POLY + 3, int(round(_SMOOTH_WIN_M / dx))))
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


def _detect_berm_and_toe(x, zb, zs, dx, shore_idx, crest_idx, BE, morph_type):
    """Locate the berm, the dune's seaward toe, and the foreshore slope.

    Returns ``(berm_present, berm_start_idx, berm_end_idx, berm_elev_meas,
    berm_width, seaward_base, seaward_toe_idx, foreshore_slope)``.
    """
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
    return (
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        berm_width,
        seaward_base,
        seaward_toe_idx,
        foreshore_slope,
    )
