"""Shared types and tunable constants for the metrics package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

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
