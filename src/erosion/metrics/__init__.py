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

The implementation is split across the package: ``types`` (constants + data
classes), ``detect`` (feature location + shared cross-shore helpers), ``forms``
(idealized-profile assembly + form selection), and ``fit`` (the ``fit_profile``
entry point + the manual-classification review pass).
"""

from __future__ import annotations

from .detect import (
    _analysis_cap,
    _dx_of,
    _smooth,
    erosion_volume_above_msl,
    last_wet_dry_crossing,
    volume_above_datum,
)
from .fit import ProfileReview, ReviewFlag, fit_profile, review_profile
from .types import IdealizedProfile, MorphType, ProfileMetrics

__all__ = [
    "IdealizedProfile",
    "MorphType",
    "ProfileMetrics",
    "ProfileReview",
    "ReviewFlag",
    "erosion_volume_above_msl",
    "fit_profile",
    "last_wet_dry_crossing",
    "review_profile",
    "volume_above_datum",
    # Re-exported for tests that exercise internals directly.
    "_analysis_cap",
    "_dx_of",
    "_smooth",
]
