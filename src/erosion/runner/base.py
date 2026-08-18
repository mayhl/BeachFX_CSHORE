from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..profile import Profile


class InundationError(RuntimeError):
    """CSHORE ran but the profile was overtopped/inundated -- a PHYSICAL outcome.

    This is the only exception ``run_storm`` absorbs into the INUNDATION path
    (storm skipped, profile unchanged, INUNDATION snapshot).  Every other
    exception is a DEFECT -- a missing binary, a truncated output, a parser
    bug -- and must propagate; letting defects ride the inundation path made
    the model silently drop its most damaging storms as "overtopped".
    """


class CSHOREResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zb: np.ndarray  # final bed elevation, meters, shape (N,)
    x: np.ndarray  # cross-shore positions, meters, shape (N,)
    eta: np.ndarray  # mean water level at final BC timestep (OSETUP "setup")
    Hs: np.ndarray  # significant wave height at final BC timestep
    runup_m: float  # 2% runup from ODOC, meters (0.0 if not reported)
    # Landward wet-computation limit node count (hydro valid over nodes < jr); 0 = unknown.
    jr: int = 0


class CSHORERunner(ABC):
    @abstractmethod
    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult: ...


class MockCSHORERunner(CSHORERunner):
    """Test stand-in: scoops a fixed index window by 0.1 x peak Hs."""

    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        peak_Hs = float(np.max(storm_forcing["Hs"]))
        n = len(profile.zb)

        zb_out = profile.zb.copy()
        lo = min(50, n)
        hi = min(150, n)
        zb_out[lo:hi] -= 0.1 * peak_Hs

        dummy = np.zeros(n)
        return CSHOREResult(
            zb=zb_out,
            x=profile.x.copy(),
            eta=dummy,
            Hs=dummy,
            runup_m=0.0,
            jr=n,  # mock: whole domain "wet"
        )
