from __future__ import annotations

import numpy as np

from ..profile import Profile
from .base import CSHOREResult, CSHORERunner


class MockCSHORERunner(CSHORERunner):
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
