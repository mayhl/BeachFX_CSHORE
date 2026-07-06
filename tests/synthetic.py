"""Synthetic cross-shore profile generator for metrics round-trip tests.

Builds a subaerial beach profile (shoreline → foreshore → berm → dune → upland)
from explicit parameters and returns the ground-truth feature values, so
``fit_profile`` can be tested by generate-then-recover.  Dune shape is pluggable
(triangular / trapezoidal / gaussian) and perturbations (noise, scarp) can be
layered on so the tests exercise robustness, not just self-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DuneSpec:
    shape: str = "triangular"  # "triangular" | "trapezoidal" | "gaussian"
    crest_elevation: float = 5.0
    front_width: float = 15.0
    back_width: float = 15.0
    top_width: float = 0.0  # trapezoidal plateau width
    sigma: float = 6.0  # gaussian width


@dataclass
class Truth:
    shoreline_x: float
    berm_elevation: float
    berm_width: float
    upland_elevation: float
    has_dune: bool
    dune_crest_elevation: float
    dune_crest_x: float
    dune_width: float
    morph_type: str


def _classify(UE: float, BE: float, DE: float) -> str:
    if not (DE > UE):
        return "HIGH_UPLAND"
    return "LOW_BERM" if BE <= UE else "LOW_UPLAND"


def make_profile(
    *,
    shoreline_x: float = 20.0,
    foreshore_slope: float = 0.1,
    berm_elevation: float = 2.0,
    berm_width: float = 30.0,
    dune: DuneSpec | None = None,
    upland_elevation: float = 3.0,
    design_BE: float | None = None,
    dx: float = 1.0,
    x_max: float = 320.0,
    datum: float = 0.0,
    noise: float = 0.0,
    scarp: tuple[float, float] | None = None,  # (face_x, face_height): erode seaward of face_x
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, Truth]:
    """Return ``(x, z, truth)``.  ``design_BE`` defaults to ``berm_elevation``."""
    x = np.arange(0.0, x_max + dx / 2, dx)
    BE = berm_elevation if design_BE is None else design_BE

    toe_slope = 0.05
    berm_start = shoreline_x + (berm_elevation - datum) / foreshore_slope
    berm_end = berm_start + berm_width

    kx = [x[0], shoreline_x, berm_start, berm_end]
    kz = [datum - toe_slope * shoreline_x, datum, berm_elevation, berm_elevation]

    has_dune = dune is not None
    crest_x = float("nan")
    DE = upland_elevation
    gauss = None
    if has_dune and dune.shape in ("triangular", "trapezoidal"):
        c0 = berm_end + dune.front_width
        top = dune.top_width if dune.shape == "trapezoidal" else 0.0
        c1 = c0 + top
        land_toe = c1 + dune.back_width
        kx += [c0, c1, land_toe]
        kz += [dune.crest_elevation, dune.crest_elevation, upland_elevation]
        crest_x = 0.5 * (c0 + c1)
        DE = dune.crest_elevation
    elif has_dune and dune.shape == "gaussian":
        # Base steps berm → upland over the dune span; a Gaussian bump is added on top.
        span = 6.0 * dune.sigma
        land_toe = berm_end + span
        x0 = berm_end + span / 2.0
        kx += [land_toe]
        kz += [upland_elevation]
        crest_x = x0
        DE = dune.crest_elevation
        gauss = (x0, dune.sigma)
    kx.append(float(x_max))
    kz.append(upland_elevation)
    z = np.interp(x, kx, kz)

    if gauss is not None:
        x0, sigma = gauss
        amp = dune.crest_elevation - float(np.interp(x0, kx, kz))
        z = z + amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

    # Truth dune width = footprint on the CLEAN profile: seaward toe at the berm
    # edge, landward toe at the first descent to (upland + band).  The band keeps
    # a Gaussian's asymptotic tail from pushing the toe arbitrarily far landward.
    dune_width = 0.0
    if has_dune:
        ci = int(np.argmin(np.abs(x - crest_x)))
        band = 0.2
        land = next((x[i] for i in range(ci, len(x)) if z[i] <= upland_elevation + band), x[-1])
        dune_width = float(land - berm_end)

    truth = Truth(
        shoreline_x=shoreline_x,
        berm_elevation=berm_elevation,
        berm_width=berm_width,
        upland_elevation=upland_elevation,
        has_dune=has_dune,
        dune_crest_elevation=DE,
        dune_crest_x=crest_x,
        dune_width=dune_width,
        morph_type=_classify(upland_elevation, BE, DE),
    )

    if scarp is not None:
        xc, drop = scarp
        # Localized erosional scarp: waves cut the beach/dune back from seaward,
        # leaving an eroded bench SEAWARD of a steep face at xc with the profile
        # LANDWARD of it intact.  The face height is ``drop``.  (Not a global
        # offset — the dune/upland behind the scarp are unchanged.)
        eroded = float(np.interp(xc, x, z)) - drop
        z = np.where(x < xc, np.minimum(z, eroded), z)

    if noise > 0.0:
        z = z + np.random.default_rng(seed).normal(0.0, noise, size=len(x))

    return x, z, truth
