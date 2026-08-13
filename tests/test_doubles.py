"""The arrange contract for the CSHORE doubles.

Every event scenario reasons in named damage — "MINOR is under the gate, SEVERE is over
it" — and that reasoning is only sound if the damage a form *advertises* is the deficit
the **real** assessor measures off the bed it synthesizes.  These tests pin that.  When
they fail, the scenarios that lean on them are lying, and this file says so directly
rather than surfacing as a dozen inscrutable sequence mismatches.
"""

from __future__ import annotations

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.nourishment.assess import FittedAssessor, GeometricAssessor, VolumeAssessor
from erosion.runner.base import CSHOREResult
from tests.builders import ncfg, template_profile
from tests.doubles import (
    NONE,
    BermCut,
    Damage,
    DuneCut,
    DuneScarp,
    Inundation,
    Overwash,
    ScriptedRunner,
)

ASSESSORS = {
    "volume": VolumeAssessor,
    "fitted": FittedAssessor,
    "geometric": GeometricAssessor,
}


def _assessed(damage: Damage, assessor: str, width_m: float = 1.0) -> float:
    """The deficit the REAL assessor measures off the bed the double synthesized."""
    p = template_profile("p0")
    p.zb = damage.bed(p)
    cfg = ReachConfig(nourishment=ncfg(volume_trigger=30.0, assessor=assessor))
    return ASSESSORS[assessor]().assess(p, cfg, width_m).deficit_m3


class TestBermCutIsHonest:
    @pytest.mark.parametrize("assessor", list(ASSESSORS))
    @pytest.mark.parametrize("cut", [5.0, 10.0, 15.0, 20.0, 30.0])
    def test_the_deficit_it_advertises_is_the_deficit_the_assessor_measures(self, assessor, cut):
        """``cut x berm_elevation`` — the whole reason this form exists.  The fitted and
        geometric assessors land on it exactly; the volume assessor adds a fixed ~2 m3 of
        foreshore ramp, hence the absolute term (a fixed offset swamps rel at small cuts)."""
        advertised = BermCut(cut).deficit_m3(template_profile("p0"))
        assert _assessed(BermCut(cut), assessor) == pytest.approx(advertised, rel=0.10, abs=2.5)

    @pytest.mark.parametrize("assessor", list(ASSESSORS))
    def test_every_assessor_sees_it(self, assessor):
        """The failure that retired the uniform scoop: it lowered berm and dune together,
        left the measured berm width intact, and so read as a deficit of exactly zero on
        the two dune-aware assessors.  A suite that pins ``assessor="volume"`` to work
        around that is not testing the assessor the default config selects."""
        assert _assessed(BermCut(20.0), assessor) > 30.0

    @pytest.mark.parametrize("assessor", list(ASSESSORS))
    def test_a_bigger_cut_is_a_bigger_deficit(self, assessor):
        deficits = [_assessed(BermCut(c), assessor) for c in (5.0, 10.0, 20.0, 30.0)]
        assert deficits == sorted(deficits)

    def test_it_scales_with_the_longshore_width(self):
        assert _assessed(BermCut(20.0), "fitted", width_m=500.0) == pytest.approx(
            500.0 * _assessed(BermCut(20.0), "fitted", width_m=1.0)
        )

    def test_it_leaves_the_dune_alone(self):
        """A storm attacks the front of the beach.  If the cut sheared the dune off flat,
        it would trip the emergency triggers and every scenario would mean something else."""
        p = template_profile("p0")
        crest = p.ref_metrics.dune_crest_elevation
        assert BermCut(20.0).bed(p).max() == pytest.approx(p.zb.max())
        assert BermCut(20.0).bed(p).max() >= crest - 1e-9


class TestDamageIsErosionOnly:
    @pytest.mark.parametrize(
        "damage",
        [BermCut(20.0), DuneScarp(5.0), DuneCut(2.0), Overwash(berm=20.0, dune=1.0), NONE],
    )
    def test_no_form_ever_adds_sand(self, damage):
        p = template_profile("p0")
        assert np.all(damage.bed(p) <= p.zb + 1e-12)

    def test_none_is_a_true_no_op(self):
        p = template_profile("p0")
        np.testing.assert_array_equal(NONE.bed(p), p.zb)


class TestDuneScarp:
    def test_it_planes_the_face_to_berm_height_up_to_the_scarp(self):
        p = template_profile("p0")
        ref = p.ref_metrics
        x_toe = ref.dune_crest_x - ref.dune_front_width
        bed = DuneScarp(5.0).bed(p)
        gone = (p.x > x_toe + 0.5) & (p.x < x_toe + 4.5)
        np.testing.assert_allclose(bed[gone], ref.berm_elevation, atol=1e-9)

    def test_the_crest_survives_unlike_dune_cut(self):
        """The reason this form exists: a scarp eats the face, not the crest, so the fitter
        keeps seeing a dune and the HIGH_UPLAND cliff that bites ``DuneCut`` never trips."""
        p = template_profile("p0")
        assert DuneScarp(10.0).bed(p).max() == pytest.approx(p.zb.max())

    def test_it_leaves_the_berm_alone(self):
        p = template_profile("p0")
        berm = p.x < p.ref_metrics.dune_crest_x - p.ref_metrics.dune_front_width
        np.testing.assert_allclose(DuneScarp(10.0).bed(p)[berm], p.zb[berm])

    def test_a_steeper_face_removes_less_sand(self):
        """The face climbs from the scarp position, so a shallow post-relaxation angle (34,
        repose) undercuts more of the dune than a fresh near-vertical one; degenerate 90
        removes the least because everything landward of the scarp line stands."""
        p = template_profile("p0")
        removed = [(p.zb - DuneScarp(5.0, angle_deg=a).bed(p)).sum() for a in (34.0, 70.0, 90.0)]
        assert removed[0] > removed[1] > removed[2] > 0.0

    def test_the_dune_aware_assessors_are_blind_to_it(self):
        """Documents CURRENT behavior, not desired behavior: the fitted deficit is berm-width
        shortfall only, so a scarp that strips tens of m3 off the dune reads as exactly zero
        and can reach a decision only via the geometric emergency trigger.  Whether the
        fitted assessor should carry a dune term is an open design question — if this fails
        because one landed, update it deliberately, don't loosen it."""
        assert _assessed(DuneScarp(15.0), "volume") > 20.0
        assert _assessed(DuneScarp(15.0), "fitted") == 0.0


class TestDuneCut:
    def test_it_lowers_the_crest_by_what_it_says(self):
        p = template_profile("p0")
        assert DuneCut(1.5).bed(p).max() == pytest.approx(p.ref_metrics.dune_crest_elevation - 1.5)

    def test_it_leaves_the_berm_alone(self):
        p = template_profile("p0")
        berm = p.x < p.ref_metrics.dune_crest_x - p.ref_metrics.dune_front_width
        np.testing.assert_allclose(DuneCut(1.5).bed(p)[berm], p.zb[berm])


class TestScriptedRunner:
    def _bed(self, runner: ScriptedRunner, pid: str = "p0") -> np.ndarray:
        return runner.run(template_profile(pid), {}).zb

    def test_a_bare_damage_applies_to_every_profile_and_storm(self):
        r = ScriptedRunner(BermCut(20.0))
        for pid in ("p0", "p1"):
            np.testing.assert_allclose(self._bed(r, pid), BermCut(20.0).bed(template_profile(pid)))

    def test_a_list_is_consumed_one_entry_per_storm(self):
        r = ScriptedRunner([BermCut(20.0), NONE])
        assert self._bed(r).min() < template_profile("p0").zb.min() + 1e9  # storm 0 cuts
        np.testing.assert_array_equal(self._bed(r), template_profile("p0").zb)  # storm 1 passes

    def test_a_short_script_repeats_its_last_entry(self):
        r = ScriptedRunner([BermCut(20.0)])
        first = self._bed(r)
        np.testing.assert_allclose(self._bed(r), first)  # storm 1 gets storm 0's damage

    def test_each_profile_gets_its_own_script_and_its_own_storm_counter(self):
        r = ScriptedRunner({"p0": [BermCut(20.0), NONE], "p1": NONE})
        cut_p0 = self._bed(r, "p0")
        assert cut_p0.min() < template_profile("p0").zb.min() + 1e9
        np.testing.assert_array_equal(self._bed(r, "p1"), template_profile("p1").zb)
        np.testing.assert_array_equal(self._bed(r, "p0"), template_profile("p0").zb)  # p0's storm 1

    def test_an_unnamed_profile_takes_no_damage(self):
        r = ScriptedRunner({"p0": BermCut(20.0)})
        np.testing.assert_array_equal(self._bed(r, "p9"), template_profile("p9").zb)

    def test_inundation_raises_the_way_a_failing_solver_does(self):
        """``run_parallel_cshore`` isolates the profile by catching the exception.  A double
        that returned an empty result instead would slip past the catch untested."""
        with pytest.raises(RuntimeError, match="inundation"):
            self._bed(ScriptedRunner(Inundation()))

    def test_it_returns_a_bed_on_the_profiles_own_grid(self):
        result = ScriptedRunner(BermCut(20.0)).run(template_profile("p0"), {})
        assert isinstance(result, CSHOREResult)
        np.testing.assert_array_equal(result.x, template_profile("p0").x)
        assert result.jr == len(result.zb)


def test_the_double_never_reads_the_trigger():
    """The honesty line, asserted structurally: the double picks the damage, the assessor
    picks the deficit, the decider picks whether to mobilize.  A double that sized its cut
    from ``volume_trigger`` would make the gate untestable — it could invert its comparison
    and every scenario would still pass.  So ``run()`` takes no config, and the same script
    must produce the same bed no matter what the trigger is set to."""
    beds = [ScriptedRunner(BermCut(20.0)).run(template_profile("p0"), {}).zb for _ in range(2)]
    np.testing.assert_array_equal(*beds)
