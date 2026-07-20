"""Test doubles for CSHORE — the one task worth stubbing (expensive, output not
known a-priori).

The vocabulary here states damage as *morphology* — "the storm cut 20 m off the
berm" — rather than as a number tuned until some threshold happens to trip.  Each
form synthesizes a plausible eroded **bed**; the real assessor then measures the
deficit off it, the real decider gates it, and the real scheduler places it.  So the
double picks the damage and nothing else:

    the mock chooses the damage → the assessor chooses the deficit → the decider
    chooses to mobilize

Keeping those three apart is what makes the suite honest.  A double that peeked at
``volume_trigger`` and cut "just enough to trigger" would make the gate untestable by
construction: it could invert its comparison and every scenario would still pass.

Why not a uniform scoop (the retired ``MockCSStorm``): lowering the whole bed by dz
drops the berm, the dune and the shoreface together, so the *measured berm width* is
unchanged and ``FittedAssessor`` (and ``GeometricAssessor``, which subclasses it) report
a deficit of exactly zero.  Only ``VolumeAssessor`` could see that damage, which is why
every event test had to pin ``assessor="volume"``.  A berm cut is visible to all three.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from erosion.profile import Profile
from erosion.runner.base import CSHOREResult, CSHORERunner


class Damage(ABC):
    """What one storm does to one profile."""

    @abstractmethod
    def bed(self, profile: Profile) -> np.ndarray:
        """The eroded bed this damage leaves.  May raise, to model a CSHORE failure."""


def _erode_to(profile: Profile, face: np.ndarray) -> np.ndarray:
    """Cut the bed down to ``face`` — never up.  A storm removes sand; it never adds it,
    so every form here is erosion-only and a no-op wherever the bed is already lower."""
    return np.minimum(profile.zb, face)


@dataclass(frozen=True)
class BermCut(Damage):
    """Retreat the shoreline ``metres`` landward, re-cutting the beach face at the
    reference foreshore slope.  The dune and upland are untouched — storms attack the
    front of the beach, not the back.

    Measured berm width drops by ``metres``, so the fill deficit is the dry wedge the
    retreat carved out: ``metres x berm_elevation x width_m``, give or take the
    foreshore ramp.  That is the point of this form — the deficit is arithmetic the
    reader can do in their head, not an emergent number someone once transcribed.
    """

    metres: float

    def bed(self, profile: Profile) -> np.ndarray:
        ref = profile.ref_metrics
        # Slide the beach face landward: whatever sat at x now sits at x + metres.  A
        # translation (rather than a fresh line struck from the new waterline) is what keeps
        # the cut joined to the shoreface below — strike the line instead and it lands on
        # the old bed as a vertical scarp, which no storm makes and no fitter believes.
        shifted = np.interp(profile.x - self.metres, profile.x, profile.zb, left=profile.zb[0])
        # Only the face travels.  The berm plateau and everything landward of it hold their
        # ground, so the cut stops where the retreated face regains berm height — without
        # that, ``min`` would shear the dune off flat at berm elevation.
        face = np.where(shifted < ref.berm_elevation - 1e-9, shifted, np.inf)
        return _erode_to(profile, face)

    def deficit_m3(self, profile: Profile, width_m: float = 1.0) -> float:
        """The deficit this cut advertises — the arrange contract the honesty test pins
        against what the real assessors actually measure."""
        return self.metres * profile.ref_metrics.berm_elevation * width_m


@dataclass(frozen=True)
class DuneScarp(Damage):
    """Cut a scarp ``metres`` back into the dune front, leaving a steep face standing at
    the berm.

    This is what a storm actually does to a dune: it undercuts the seaward face and calves
    it off, leaving the scarp the fitter's ``dune_scarp`` / ``scarp_height`` metrics exist
    to measure.  It is not ``DuneCut`` — see that form's warning.

    ``angle_deg`` is the face angle from horizontal.  A fresh storm scarp stands near
    vertical and relaxes toward the angle of repose (~34 deg for dry sand) over the days
    after; 70 deg is a reasonable just-after-the-storm default.  Vertical (90) is available
    but it is a numerical fiction — on a 1 m grid it puts the whole face in one cell — so
    prefer a finite angle unless the test is specifically about the degenerate case.
    """

    metres: float
    angle_deg: float = 70.0

    def bed(self, profile: Profile) -> np.ndarray:
        ref = profile.ref_metrics
        x_toe = ref.dune_crest_x - ref.dune_front_width  # seaward foot of the dune
        x_scarp = x_toe + self.metres  # where the face is left standing
        # Seaward of the scarp the dune is gone, planed off at berm height; the face then
        # climbs at ``angle_deg`` until it meets the dune the storm did not reach.
        if self.angle_deg >= 90.0:
            face = np.where(profile.x < x_scarp, ref.berm_elevation, np.inf)
        else:
            rise = np.tan(np.radians(self.angle_deg)) * (profile.x - x_scarp)
            face = ref.berm_elevation + np.maximum(rise, 0.0)
        return _erode_to(profile, face)


@dataclass(frozen=True)
class DuneCut(Damage):
    """Shave ``metres`` off the dune crest, leaving the berm intact.

    Prefer ``DuneScarp`` for storm damage — a storm undercuts the dune face, it does not
    plane the crest flat.  This form exists for the geometric emergency trigger, which
    reads crest *height* (``trigger_geometry.dune_height``) directly.

    Mind the cliff: shaving a trapezoidal crest widens its flat top, so the dune's
    prominence over the upland collapses.  Past ~1.5 m on ``template_profile`` the fitter
    stops calling it a dune at all (morph_type flips to HIGH_UPLAND, crest goes NaN) and
    ``FittedAssessor`` then reports a *constant* deficit no matter how much more sand is
    taken.  Scenarios that lean on this form must stay on the near side of that.
    """

    metres: float

    def bed(self, profile: Profile) -> np.ndarray:
        ceiling = profile.ref_metrics.dune_crest_elevation - self.metres
        return _erode_to(profile, np.full(len(profile.x), ceiling))


@dataclass(frozen=True)
class Overwash(Damage):
    """Berm cut and dune shave together — the storm that drives the CAT_* responses."""

    berm: float
    dune: float

    def bed(self, profile: Profile) -> np.ndarray:
        cut = BermCut(self.berm).bed(profile)
        ceiling = profile.ref_metrics.dune_crest_elevation - self.dune
        return np.minimum(cut, ceiling)


class Inundation(Damage):
    """CSHORE produced nothing.  Raises, because that is how the real failure presents:
    ``run_parallel_cshore`` catches the exception and isolates the profile.  A double
    that returned ``None`` instead would slip past the catch and stop testing it."""

    def bed(self, profile: Profile) -> np.ndarray:
        raise RuntimeError("mock CSHORE failure (inundation)")


NONE = BermCut(0.0)  # a storm that passed through without touching the beach

# Canonical sizes against ``template_profile`` (BE = 2.0 m) and the conventional 30 m3
# trigger, so a scenario can say what it means: MINOR recovers, SEVERE mobilizes, and
# MASSIVE (the whole 30 m berm) is what a slow production rate stretches into an
# interruptible multi-week placement.  Deficits are the cut x 2 m3/m the reader can
# check in their head (the volume assessor adds ~2 m3 of foreshore ramp on top).
MINOR = BermCut(10.0)  # ~22 m3 — under the gate
SEVERE = BermCut(20.0)  # ~42 m3 — over the gate
MASSIVE = BermCut(30.0)  # ~62 m3 — the full berm


class ScriptedRunner(CSHORERunner):
    """A CSHORE double that reads a script of ``Damage`` per (profile, storm).

        ScriptedRunner(SEVERE)                      # every profile, every storm
        ScriptedRunner([SEVERE, MINOR])             # storm 0, then storm 1
        ScriptedRunner({"p0": [SEVERE, MINOR],      # per profile
                        "p1": Inundation()})

    Storms are consumed in order per profile.  A script shorter than the schedule repeats
    its last entry, so ``[SEVERE]`` reads as "the first storm does the damage, the rest
    pass through".  A profile the script does not name takes no damage.
    """

    def __init__(self, script: Damage | list[Damage] | dict[str, Damage | list[Damage]]):
        self.script = script
        self._calls: dict[str, int] = {}

    def _damage(self, pid: str) -> Damage:
        spec = self.script.get(pid, NONE) if isinstance(self.script, dict) else self.script
        if isinstance(spec, (list, tuple)):
            i = self._calls.get(pid, 0)
            self._calls[pid] = i + 1
            return spec[min(i, len(spec) - 1)]
        return spec

    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        zb = self._damage(profile.id).bed(profile)  # may raise (Inundation)
        n = len(zb)
        dummy = np.zeros(n)
        return CSHOREResult(zb=zb, x=profile.x.copy(), eta=dummy, Hs=dummy, runup_m=0.0, jr=n)
