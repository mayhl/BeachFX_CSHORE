"""Shared cross-shore geometry helpers (also used by profile.py / nourishment.py)."""

from __future__ import annotations

import numpy as np


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
