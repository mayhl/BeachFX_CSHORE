"""Profile morphology metrics: BeachFX classification and piecewise-linear fit.

Extracts 0-D metrics from a CSHORE cross-shore profile snapshot.

CSHORE x convention: x[0] is the most offshore node; x[-1] is the most
landward node (the upland).  zb[i] is the bed elevation at x[i].

Three BeachFX morphology types (Tech Ref §7.3.1–7.3.2):

  HIGH_UPLAND  — UE ≥ DE:  upland elevation at or above dune crest; no
                            distinct dune relief above the upland.
  LOW_BERM     — DE > UE, BE ≤ UE:  dune present, but berm elevation is
                            at or below upland elevation.
  LOW_UPLAND   — DE > UE, BE > UE:  dune present and berm height exceeds
                            upland elevation.

Only ``berm_elevation`` (BE) is required as a config input.  DE is the
measured argmax of zb (restricted to the non-upland region); UE is the
mean of the last K rear-profile nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_NODES:       int   = 10    # minimum profile length before attempting fit
_UPLAND_K:        int   = 10    # number of rear nodes used to estimate UE
_BERM_TOL:        float = 0.15  # elevation tolerance (m) for berm_end (dune-foot) scan
_BERM_FLAT_SLOPE: float = 0.02  # max |dz/dx| (m/m) qualifying as flat berm
_BERM_MIN_FRAC:   float = 0.75  # berm_start must be ≥ this fraction of BE above datum


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
    """0-D morphology metrics for a single (profile, snapshot) pair."""

    morph_type: str             # MorphType.value

    # Shoreline
    shoreline_x: float          # x where zb = datum (m from offshore baseline)

    # Beach
    berm_width: float           # BW: horizontal dry-beach flat width (m), ≥ 0
    foreshore_slope: float      # beach face slope dz/dx (dimensionless, > 0)

    # Dune  (NaN for HIGH_UPLAND)
    dune_height: float          # DE: dune crest elevation (m)
    dune_x: float               # x-coordinate of dune crest (m); used for constrained ref search
    dune_width: float           # DW: front + back dune width (m)
    dune_front_slope: float     # ascending slope, seaward dune face (dz/dx)
    dune_back_slope: float      # descending slope, landward dune face (|dz/dx|)

    # Upland
    upland_elevation: float     # UE: mean elevation of rear K nodes (m)

    # Volume
    volume_above_datum: float   # ∫ max(zb − datum, 0) dx  (m²/m)

    # Scarp diagnostics
    scarp_present: bool         # True when BW == 0 (no flat berm detected)
    max_beach_slope: float      # steepest |dz/dx| in beach-to-dune zone
    dune_front_resid: float     # RMS of linear fit over dune-front segment

    n_upland_nodes: int         # K value actually used for UE estimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(UE: float, BE: float, DE: float) -> str:
    if DE <= UE:
        return MorphType.HIGH_UPLAND.value
    elif BE <= UE:
        return MorphType.LOW_BERM.value
    else:
        return MorphType.LOW_UPLAND.value


def _linear_slope(x: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Least-squares linear fit of z on x.  Returns (slope, rms_residual)."""
    if len(x) < 2:
        return 0.0, 0.0
    A = np.column_stack([x, np.ones(len(x))])
    coeffs, res, _, _ = np.linalg.lstsq(A, z, rcond=None)
    slope = float(coeffs[0])
    if len(res) > 0:
        rms = float(np.sqrt(max(res[0] / len(z), 0.0)))
    else:
        rms = float(np.sqrt(np.mean((z - (A @ coeffs)) ** 2)))
    return slope, rms


def _max_abs_slope(x: np.ndarray, z: np.ndarray) -> float:
    """Maximum |dz/dx| over segment.  Returns NaN for segments < 2 nodes."""
    dx = np.diff(x)
    valid = dx > 0
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(np.diff(z)[valid] / dx[valid])))


def _null_metrics(vol: float = 0.0, UE: float = float("nan"), K: int = 0) -> ProfileMetrics:
    nan = float("nan")
    return ProfileMetrics(
        morph_type=MorphType.INDETERMINATE.value,
        shoreline_x=nan, berm_width=0.0, foreshore_slope=nan,
        dune_height=nan, dune_x=nan, dune_width=0.0,
        dune_front_slope=nan, dune_back_slope=nan,
        upland_elevation=UE, volume_above_datum=vol,
        scarp_present=False, max_beach_slope=nan,
        dune_front_resid=nan, n_upland_nodes=K,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_profile(
    x: np.ndarray,
    zb: np.ndarray,
    berm_elevation: float,
    datum: float = 0.0,
    ref: ProfileMetrics | None = None,
) -> ProfileMetrics:
    """Extract 0-D morphology metrics from a cross-shore profile.

    Parameters
    ----------
    x : ndarray
        Cross-shore node positions (m).  x[0] = offshore, x[-1] = landward.
    zb : ndarray
        Bed elevation at each node (m).
    berm_elevation : float
        Design berm crest elevation BE (m).  Only config input required.
    datum : float
        Shoreline datum (m).  Default 0.0 = mean sea level.
    ref : ProfileMetrics | None
        Reference metrics from the equilibrium/pre-storm profile.  When
        provided, the dune-crest search is constrained to a ±30 m window
        around ``ref.dune_x`` to prevent the detector jumping to spurious
        peaks after storms.  Falls back to global argmax if no peak is
        found in the window.
    """
    x  = np.asarray(x,  dtype=float)
    zb = np.asarray(zb, dtype=float)
    n  = len(x)
    BE = float(berm_elevation)
    nan = float("nan")

    if n < _MIN_NODES:
        return _null_metrics()

    K = min(_UPLAND_K, max(2, n // 10))

    # --- Volume (always computable) ---
    vol = float(np.trapezoid(np.maximum(zb - datum, 0.0), x))

    # --- Upland elevation ---
    UE = float(np.mean(zb[-K:]))

    # --- Dune detection (restrict to non-upland region) ---
    upland_start = max(n - K, 1)       # first index of flat upland region

    # Constrained search when reference dune_x is available
    ref_dune_x = ref.dune_x if (ref is not None and not np.isnan(ref.dune_x)) else None
    if ref_dune_x is not None:
        x_lo, x_hi = ref_dune_x - 30.0, ref_dune_x + 30.0
        mask = (x[:upland_start] >= x_lo) & (x[:upland_start] <= x_hi)
        if np.any(mask):
            idx_arr = np.where(mask)[0]
            dune_idx = int(idx_arr[np.argmax(zb[idx_arr])])
        else:
            dune_idx = int(np.argmax(zb[:upland_start]))
    else:
        dune_idx = int(np.argmax(zb[:upland_start]))

    DE = float(zb[dune_idx])
    dune_x_val = float(x[dune_idx])

    # --- Shoreline: last wet→dry transition seaward of dune ---
    #     In CSHORE x, scanning left→right crosses datum from below (wet→dry).
    sub = zb[:dune_idx + 1]
    dry = (sub > datum).astype(int)
    transitions = np.where(np.diff(dry) > 0)[0]   # wet→dry rising crossings

    if len(transitions) > 0:
        ci = int(transitions[-1])
        z0, z1 = float(zb[ci]), float(zb[ci + 1])
        dz = z1 - z0
        frac = (datum - z0) / dz if abs(dz) > 1e-9 else 0.0
        x_shore = float(x[ci] + frac * (x[ci + 1] - x[ci]))
        shore_idx = ci + 1
    elif np.all(sub > datum):
        x_shore = float(x[0])    # entire visible portion is dry
        shore_idx = 0
    else:
        # Profile does not emerge from below datum before dune — can't fit
        m = _null_metrics(vol=vol, UE=UE, K=K)
        m.dune_height = DE
        m.dune_x = dune_x_val
        return m

    # Sanity guard: need at least a few nodes between shore and dune
    if dune_idx <= shore_idx:
        return ProfileMetrics(
            morph_type=_classify(UE, BE, DE),
            shoreline_x=x_shore, berm_width=0.0, foreshore_slope=nan,
            dune_height=DE, dune_x=dune_x_val, dune_width=0.0,
            dune_front_slope=nan, dune_back_slope=nan,
            upland_elevation=UE, volume_above_datum=vol,
            scarp_present=True, max_beach_slope=nan,
            dune_front_resid=nan, n_upland_nodes=K,
        )

    # --- Berm detection ---
    # berm_start: first node where:
    #   (a) elevation >= _BERM_MIN_FRAC * BE (above the lower half of berm elevation
    #       — adapts to long-run erosion drift without needing a fixed tolerance), and
    #   (b) forward slope < _BERM_FLAT_SLOPE (we are on the flat berm, not the
    #       steep beach face that passes through the same elevation window).
    berm_start_idx = dune_idx   # fallback: no berm
    for i in range(shore_idx, dune_idx):
        if zb[i] >= _BERM_MIN_FRAC * BE:
            slope_fwd = abs(zb[i + 1] - zb[i]) / max(x[i + 1] - x[i], 1e-9)
            if slope_fwd < _BERM_FLAT_SLOPE:
                berm_start_idx = i
                break

    # berm_end: last index in [shore_idx, dune_idx] where zb ≤ BE (dune foot)
    berm_end_idx = shore_idx    # fallback: no berm
    for i in range(dune_idx, shore_idx - 1, -1):
        if zb[i] <= BE + _BERM_TOL:
            berm_end_idx = i
            break

    if berm_start_idx < berm_end_idx:
        BW = max(0.0, float(x[berm_end_idx] - x[berm_start_idx]))
        scarp = False
    else:
        BW = 0.0
        scarp = True
        berm_end_idx = berm_start_idx   # collapse to same point

    # --- Foreshore slope (shore_idx → berm_start_idx) ---
    if berm_start_idx > shore_idx:
        beta_f, _ = _linear_slope(x[shore_idx:berm_start_idx + 1],
                                   zb[shore_idx:berm_start_idx + 1])
        beta_f = max(0.0, beta_f)   # foreshore must ascend landward
    else:
        beta_f = nan

    # --- Steepest slope in beach-dune zone ---
    max_sl = _max_abs_slope(x[shore_idx:dune_idx + 1], zb[shore_idx:dune_idx + 1])

    # --- HIGH_UPLAND: no dune structure above upland ---
    morph_type = _classify(UE, BE, DE)
    if morph_type == MorphType.HIGH_UPLAND.value:
        return ProfileMetrics(
            morph_type=morph_type,
            shoreline_x=x_shore, berm_width=BW, foreshore_slope=beta_f,
            dune_height=DE, dune_x=dune_x_val, dune_width=0.0,
            dune_front_slope=nan, dune_back_slope=nan,
            upland_elevation=UE, volume_above_datum=vol,
            scarp_present=scarp, max_beach_slope=max_sl,
            dune_front_resid=nan, n_upland_nodes=K,
        )

    # --- LOW_BERM / LOW_UPLAND: fit dune front and back slopes ---

    # Dune front (berm_end_idx → dune_idx)
    if dune_idx > berm_end_idx:
        beta_df, df_resid = _linear_slope(x[berm_end_idx:dune_idx + 1],
                                           zb[berm_end_idx:dune_idx + 1])
        beta_df   = max(0.0, beta_df)   # dune front must ascend
        dw_front  = float(x[dune_idx] - x[berm_end_idx])
    else:
        beta_df  = nan
        df_resid = nan
        dw_front = 0.0

    # Dune back (dune_idx → upland_start)
    if upland_start > dune_idx + 1:
        beta_db_raw, _ = _linear_slope(x[dune_idx:upland_start + 1],
                                        zb[dune_idx:upland_start + 1])
        beta_db  = max(0.0, -beta_db_raw)  # dune back descends; store positive magnitude
        dw_back  = float(x[upland_start] - x[dune_idx])
    else:
        beta_db = nan
        dw_back = 0.0

    return ProfileMetrics(
        morph_type=morph_type,
        shoreline_x=x_shore,
        berm_width=BW,
        foreshore_slope=float(beta_f),
        dune_height=DE,
        dune_x=dune_x_val,
        dune_width=dw_front + dw_back,
        dune_front_slope=float(beta_df),
        dune_back_slope=float(beta_db),
        upland_elevation=UE,
        volume_above_datum=vol,
        scarp_present=scarp,
        max_beach_slope=float(max_sl),
        dune_front_resid=float(df_resid) if not np.isnan(df_resid) else nan,
        n_upland_nodes=K,
    )
