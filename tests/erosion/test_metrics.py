"""Tests for erosion.metrics: shared helpers + fit_profile generate→recover."""

import os

import numpy as np
import pytest

from erosion.metrics import (
    MorphType,
    erosion_volume_above_msl,
    fit_profile,
    last_wet_dry_crossing,
    volume_above_datum,
)
from erosion.profile import load_raw_profile
from tests.synthetic import DuneSpec, make_profile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestLastWetDryCrossing:
    def test_crossing_found(self):
        x = np.arange(6, dtype=float)
        z = np.array([-2.0, -1.0, 1.0, 2.0, 2.0, 2.0])
        x_sh, idx = last_wet_dry_crossing(x, z)
        assert idx == 2
        assert x_sh == pytest.approx(1.5)  # datum crossing between idx 1 and 2

    def test_no_crossing_all_dry(self):
        assert last_wet_dry_crossing(np.arange(5.0), np.ones(5)) == (None, None)

    def test_no_crossing_all_wet(self):
        assert last_wet_dry_crossing(np.arange(5.0), -np.ones(5)) == (None, None)

    def test_end_window_excludes_later_crossing(self):
        x = np.arange(6, dtype=float)
        z = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])  # only crossing at idx 3→4
        assert last_wet_dry_crossing(x, z, end=3) == (None, None)
        assert last_wet_dry_crossing(x, z)[1] == 4

    def test_flat_segment_fallback(self):
        x = np.arange(4, dtype=float)
        z = np.array([-1.0, 0.0, 0.0, 1.0])  # dz≈0 at the crossing node
        x_sh, idx = last_wet_dry_crossing(x, z)
        assert idx is not None


class TestVolumeAboveDatum:
    def test_triangle_area(self):
        x = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 2.0, 0.0])  # triangle, base 2, height 2 → area 2
        assert volume_above_datum(x, z) == pytest.approx(2.0)

    def test_ignores_below_datum(self):
        x = np.arange(3.0)
        z = np.array([-5.0, -5.0, -5.0])
        assert volume_above_datum(x, z) == pytest.approx(0.0)


class TestErosionVolumeAboveMSL:
    def test_erosion_only_surplus_does_not_cancel(self):
        # Eroded at 0,1,4 (+1 each) but ACCRETED at 2,3 (-1 each). The old
        # difference-of-integrals nets to 0; erosion-only must keep the deficit.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        tpl = np.full(5, 2.0)
        now = np.array([1.0, 1.0, 3.0, 3.0, 1.0])
        assert volume_above_datum(x, tpl) - volume_above_datum(x, now) == pytest.approx(0.0)
        assert erosion_volume_above_msl(x, tpl, now) == pytest.approx(2.0)

    def test_clips_at_msl(self):
        # Only the dry beach (above msl) counts toward the deficit.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        tpl = np.full(5, 2.0)
        now = np.array([1.0, 1.0, 3.0, 3.0, 1.0])
        assert erosion_volume_above_msl(x, tpl, now, msl=1.5) == pytest.approx(1.0)

    def test_pure_erosion_matches_datum_difference(self):
        # With no accretion, erosion-only equals the datum-difference (msl == datum).
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        tpl = np.full(5, 2.0)
        now = tpl - 0.5
        old = volume_above_datum(x, tpl) - volume_above_datum(x, now)
        assert erosion_volume_above_msl(x, tpl, now) == pytest.approx(old)


# ---------------------------------------------------------------------------
# fit_profile — generate then recover
# ---------------------------------------------------------------------------


def _fit(x, z, design_BE):
    return fit_profile(x, z, design_BE)


class TestFitRoundTrip:
    @pytest.mark.parametrize("shape", ["triangular", "trapezoidal", "gaussian"])
    def test_recovers_features(self, shape):
        dune = DuneSpec(
            shape=shape,
            crest_elevation=5.0,
            front_width=15.0,
            back_width=20.0,
            top_width=12.0,
            sigma=7.0,
        )
        x, z, truth = make_profile(
            berm_elevation=2.0, berm_width=30.0, upland_elevation=1.0, dune=dune
        )
        m, ideal = _fit(x, z, design_BE=2.0)

        assert m.morph_type == truth.morph_type == MorphType.LOW_UPLAND.value
        assert m.shoreline_x == pytest.approx(truth.shoreline_x, abs=2.0)
        assert m.berm_elevation == pytest.approx(truth.berm_elevation, abs=0.25)
        assert m.berm_width == pytest.approx(truth.berm_width, abs=6.0)
        assert m.dune_crest_elevation == pytest.approx(truth.dune_crest_elevation, abs=0.3)
        assert m.dune_crest_x == pytest.approx(truth.dune_crest_x, abs=0.5 * dune.top_width + 4.0)
        assert m.dune_width == pytest.approx(truth.dune_width, abs=10.0)
        assert ideal is not None

    def test_clean_fit_quality_small(self):
        x, z, _ = make_profile(dune=DuneSpec("triangular"), upland_elevation=1.0)
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.fit_quality < 0.25  # idealized hugs a clean piecewise profile

    def test_noise_robust(self):
        dune = DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0)
        x, z, truth = make_profile(dune=dune, upland_elevation=1.0, noise=0.05, seed=3)
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.LOW_UPLAND.value
        assert m.dune_crest_elevation == pytest.approx(5.0, abs=0.4)
        assert m.dune_crest_x == pytest.approx(truth.dune_crest_x, abs=5.0)


class TestClassification:
    def test_low_upland(self):
        x, z, _ = make_profile(
            berm_elevation=2.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.LOW_UPLAND.value

    def test_low_berm(self):
        x, z, _ = make_profile(
            berm_elevation=2.0,
            upland_elevation=2.5,
            dune=DuneSpec("triangular", crest_elevation=5.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.LOW_BERM.value

    def test_high_upland_no_dune(self):
        x, z, _ = make_profile(berm_elevation=2.0, upland_elevation=4.0, dune=None)
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.HIGH_UPLAND.value
        assert np.isnan(m.dune_crest_elevation)

    def test_short_profile_indeterminate(self):
        x = np.linspace(0, 8, 6)
        z = np.linspace(-1, 1, 6)
        m, ideal = fit_profile(x, z, 2.0)
        assert m.morph_type == MorphType.INDETERMINATE.value
        assert ideal is None


class TestScarp:
    def test_steep_front_flags_dune_scarp(self):
        # front_width=3, relief≈3 → front slope ≈ 1.0 (> scarp threshold)
        x, z, _ = make_profile(
            berm_elevation=2.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=3.0, back_width=20.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.dune_scarp is True
        assert m.scarp_height > 0.0

    def test_gentle_front_no_dune_scarp(self):
        x, z, _ = make_profile(
            berm_elevation=2.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=25.0, back_width=20.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.dune_scarp is False

    def test_no_berm_flags_berm_scarp(self):
        x, z, _ = make_profile(
            berm_elevation=2.0,
            berm_width=0.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.berm_scarp is True

    def test_berm_cut_height_matches_drop(self):
        # a 1 m vertical cut into the berm is flagged with ~1 m scarp height
        x, z, _ = make_profile(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=2.0,
            dune=DuneSpec("triangular", crest_elevation=6.0, front_width=15.0, back_width=20.0),
            scarp=(58.0, 1.0),
        )
        m, _ = _fit(x, z, design_BE=3.0)
        assert m.berm_scarp is True
        assert m.scarp_height == pytest.approx(1.0, abs=0.3)

    def test_clean_profile_has_no_scarp(self):
        x, z, _ = make_profile(
            berm_elevation=2.0,
            berm_width=30.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=25.0, back_width=20.0),
        )
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.berm_scarp is False
        assert m.dune_scarp is False
        assert np.isnan(m.scarp_height)

    def test_high_upland_runs_berm_scarp_detection(self):
        # HIGH_UPLAND (no dune) still scans the berm zone for a scarp
        x, z, _ = make_profile(
            berm_elevation=3.0, berm_width=45.0, upland_elevation=5.5, dune=None, scarp=(58.0, 1.0)
        )
        m, _ = _fit(x, z, design_BE=3.0)
        assert m.morph_type == MorphType.HIGH_UPLAND.value
        assert m.berm_scarp is True
        assert m.scarp_height == pytest.approx(1.0, abs=0.3)


class TestRefSeededBerm:
    """Berm detection seeded from a reference fit.

    Unseeded, the berm is the longest flat *run* between shore and crest, a
    contest that survey noise on a coarse grid can lose: the run fragments or a
    spurious flat wins, the berm reads narrow, and the assessor bills a phantom
    deficit.  Seeded, the berm ELEVATION comes from the reference (re-based by
    the measured upland drift, since erosion+SLC lower the whole bed) and the
    extent is the longest near-contiguous cluster at that level.  The reference
    seeds only what we look for -- a berm the storm actually took still reads as
    gone, and a storm trough through the berm is a split, not spanned width.
    """

    NOISE = 0.10  # m; survey-scale
    DX = 3.048  # 10 ft -- the real survey spacing, where the smoother window is smallest
    BERM_W = 30.0

    def _kw(self, **over):
        kw = dict(
            berm_elevation=2.0,
            berm_width=self.BERM_W,
            dune=DuneSpec("triangular"),
            dx=self.DX,
        )
        kw.update(over)
        return kw

    def _clean_and_noisy(self, seed=7):
        x, z_clean, truth = make_profile(**self._kw())
        _, z_noisy, _ = make_profile(**self._kw(), noise=self.NOISE, seed=seed)
        return x, z_clean, z_noisy, truth

    def test_unseeded_berm_is_unreliable_under_noise_on_a_real_profile(self):
        # Pins the defect the seeding exists to fix.  The clean synthetic berm
        # no longer reproduces it (the repaired smoother recovers the flat run
        # there), but the real surveyed profile still loses most of its berm on
        # some draws when unseeded.  Documents current behavior.
        raw = load_raw_profile(os.path.join(_ROOT, "data/profiles/reach1_p1.csv"), 0.3)
        x, z0 = raw["x"], raw["z"]
        ref, _ = fit_profile(x, z0, 2.0)
        rr = np.random.default_rng(4242)
        widths = []
        for _ in range(25):
            m, _ = fit_profile(x, z0 + rr.normal(0.0, self.NOISE, size=z0.shape), 2.0)
            widths.append(m.berm_width)
        assert min(widths) < 0.6 * ref.berm_width

    def test_seeding_holds_the_berm_under_noise(self):
        # Measured against the clean fit, not the generator truth: at 6 m spacing
        # every width quantizes to a node, and that bias is not what is on trial.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        for seed in range(12):
            _, z_noisy, _ = make_profile(**self._kw(), noise=self.NOISE, seed=seed)
            m, _ = fit_profile(x, z_noisy, 2.0, ref=ref)
            assert m.berm_width == pytest.approx(ref.berm_width, abs=self.DX + 1.0)

    def test_seeding_does_not_disturb_a_clean_fit(self):
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        m, _ = fit_profile(x, z_clean, 2.0, ref=ref)
        assert m.berm_width == ref.berm_width
        assert m.berm_elevation == pytest.approx(ref.berm_elevation)

    def test_scoured_berm_still_reads_as_lost(self):
        # The storm drops the berm below the reference level: nothing stands at
        # that level any more, so the seed must NOT conjure a berm back.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        z_cut = z_clean.copy()
        berm = (x >= ref.berm_x) & (x <= ref.berm_x + ref.berm_width)
        z_cut[berm] -= 0.6
        m, _ = fit_profile(x, z_cut, 2.0, ref=ref)
        assert m.berm_width == 0.0
        assert np.isnan(m.berm_elevation)
        assert np.isnan(m.berm_x)

    def test_seeding_measures_a_narrowed_berm_not_the_reference_one(self):
        # A storm that halves the berm must read as halved: the seed supplies the
        # level to look for, never the width to report.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        x_n, z_narrow, truth_n = make_profile(**self._kw(berm_width=self.BERM_W / 2.0))
        m, _ = fit_profile(x_n, z_narrow, 2.0, ref=ref)
        assert m.berm_width == pytest.approx(truth_n.berm_width, abs=self.DX + 1.0)
        assert m.berm_width < 0.75 * ref.berm_width

    def test_a_bare_ramp_is_not_a_sliver_berm(self):
        # A foreshore ramp crosses the berm level in passing.  Below
        # _MIN_BERM_NODES that crossing is not a berm, so the level set must not
        # report a one-node sliver where the storm left a plain slope.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        toe_x = ref.berm_x + ref.berm_width
        z_ramp = z_clean.copy()
        span = (x >= ref.shoreline_x) & (x <= toe_x)
        z_ramp[span] = np.interp(
            x[span], [ref.shoreline_x, toe_x], [0.0, float(np.interp(toe_x, x, z_clean))]
        )
        m, _ = fit_profile(x, z_ramp, 2.0, ref=ref)
        assert m.berm_width == 0.0
        assert np.isnan(m.berm_elevation)

    @pytest.mark.parametrize("drop", [0.16, 0.30, 1.00])
    def test_survives_uniform_bed_drift(self, drop):
        # THE regression test.  ErosionTick lowers the whole bed; the seed level
        # must ride the drift (via the upland) or every profile reads
        # berm-destroyed once cumulative drift passes _LEVEL_TOL (0.15 m) --
        # ~month 7 at the shipped rate, with zero storm damage.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        m, _ = fit_profile(x, z_clean - drop, 2.0, ref=ref)
        assert m.berm_width == ref.berm_width
        assert m.berm_elevation == pytest.approx(ref.berm_elevation - drop, abs=0.05)

    def test_survives_drift_plus_noise(self):
        # The production condition: months of ticks AND survey noise together.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        rr = np.random.default_rng(4242)
        for _ in range(12):
            z = z_clean - 0.30 + rr.normal(0.0, self.NOISE, size=z_clean.shape)
            m, _ = fit_profile(x, z, 2.0, ref=ref)
            assert m.berm_width == pytest.approx(ref.berm_width, abs=2 * self.DX)

    def test_scour_below_the_drifted_level_still_reads_as_lost(self):
        # Re-basing must not blind loss detection: after 0.3 m of uniform drift,
        # a berm scoured a further 0.5 m below the DRIFTED level is gone.
        x, z_clean, _, _ = self._clean_and_noisy()
        ref, _ = fit_profile(x, z_clean, 2.0)
        z = z_clean - 0.30
        berm = (x >= ref.berm_x) & (x <= ref.berm_x + ref.berm_width)
        z[berm] -= 0.5
        m, _ = fit_profile(x, z, 2.0, ref=ref)
        assert m.berm_width == 0.0
        assert np.isnan(m.berm_elevation)

    def test_storm_trough_is_a_split_not_spanned_width(self):
        # A trough through the berm must not be bounding-boxed into intact berm:
        # the measured width is the largest surviving piece.  BERM_W=45 keeps the
        # pieces (~15.75 m ~ 5 nodes) above the _MIN_BERM_NODES resolution floor.
        x, z_clean, truth = make_profile(**self._kw(berm_width=45.0))
        ref, _ = fit_profile(x, z_clean, 2.0)
        t0 = ref.berm_x + ref.berm_width * 0.35
        t1 = ref.berm_x + ref.berm_width * 0.65
        z = z_clean.copy()
        z[(x >= t0) & (x <= t1)] -= 0.8
        m, _ = fit_profile(x, z, 2.0, ref=ref)
        piece = ref.berm_width * 0.35
        assert 0.0 < m.berm_width <= piece + 2 * self.DX
        assert m.berm_width < 0.6 * ref.berm_width

    def test_berm_x_marks_the_seaward_edge(self):
        # berm_x with berm_width fixes the footprint, which is what the seed reads
        x, z_clean, _, truth = self._clean_and_noisy()
        m, _ = fit_profile(x, z_clean, 2.0)
        assert m.berm_x > truth.shoreline_x
        assert np.interp(m.berm_x, x, z_clean) == pytest.approx(m.berm_elevation, abs=0.2)


class TestIdealizedProfile:
    def test_evaluate_matches_raw_clean(self):
        x, z, _ = make_profile(dune=DuneSpec("triangular"), upland_elevation=1.0)
        _, ideal = _fit(x, z, design_BE=2.0)
        zi = ideal.evaluate(x)
        dry = z > 0.0
        assert np.sqrt(np.mean((z[dry] - zi[dry]) ** 2)) < 0.25
