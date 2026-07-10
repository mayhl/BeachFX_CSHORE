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


class MockCSStorm(CSHORERunner):
    """Single mock CSHORE: each storm scoops a uniform ``depth`` (m) off the bed.

    CSHORE is the only task worth stubbing (expensive, output not known
    a-priori), so this is the *one* mock the event-generation tests need: the
    scooped volume becomes a deterministic deficit that the **real**
    ``VolumeAssessor`` turns into a nourishment campaign — everything
    downstream (deficit, scheduling, recovery) stays real.

    ``depth`` is a single metres value applied to every profile, or a
    ``profile_id → spec`` mapping where a spec is a depth, a per-storm ``list`` of
    depths (consumed one per storm), or ``"inundation"`` to make CSHORE fail
    (→ INUNDATION path).
    """

    INUNDATION = "inundation"  # raise → run_parallel_cshore isolates it as INUNDATION

    def __init__(self, depth: float = 0.5):
        self.depth = depth
        self._calls: dict[str, int] = {}

    def _spec(self, pid: str):
        spec = self.depth if not isinstance(self.depth, dict) else self.depth.get(pid, 0.0)
        if isinstance(spec, (list, tuple)):
            i = self._calls.get(pid, 0)
            self._calls[pid] = i + 1
            return spec[min(i, len(spec) - 1)]
        return spec

    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        spec = self._spec(profile.id)
        if spec == self.INUNDATION:
            raise RuntimeError("mock CSHORE failure (inundation)")
        zb = profile.zb - float(spec)  # scoop a uniform chunk out of the profile
        dummy = np.zeros(len(zb))
        return CSHOREResult(zb=zb, x=profile.x.copy(), eta=dummy, Hs=dummy, runup_m=0.0, jr=len(zb))
