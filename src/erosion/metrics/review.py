"""Profile review — flag fits that a human should classify by hand."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import find_peaks

from .detect import _dx_of, _smooth
from .geometry import last_wet_dry_crossing
from .types import _DUNE_MIN_PROMINENCE, IdealizedProfile, MorphType, ProfileMetrics

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
