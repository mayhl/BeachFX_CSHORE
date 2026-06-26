"""Tests for erosion.metrics: shared helpers + fit_profile generate→recover."""
import numpy as np
import pytest

from erosion.metrics import (
    MorphType, fit_profile, last_wet_dry_crossing, volume_above_datum,
)
from tests.synthetic import DuneSpec, make_profile


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class TestLastWetDryCrossing:
    def test_crossing_found(self):
        x = np.arange(6, dtype=float)
        z = np.array([-2.0, -1.0, 1.0, 2.0, 2.0, 2.0])
        x_sh, idx = last_wet_dry_crossing(x, z)
        assert idx == 2
        assert x_sh == pytest.approx(1.5)        # datum crossing between idx 1 and 2

    def test_no_crossing_all_dry(self):
        assert last_wet_dry_crossing(np.arange(5.0), np.ones(5)) == (None, None)

    def test_no_crossing_all_wet(self):
        assert last_wet_dry_crossing(np.arange(5.0), -np.ones(5)) == (None, None)

    def test_end_window_excludes_later_crossing(self):
        x = np.arange(6, dtype=float)
        z = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])   # only crossing at idx 3→4
        assert last_wet_dry_crossing(x, z, end=3) == (None, None)
        assert last_wet_dry_crossing(x, z)[1] == 4

    def test_flat_segment_fallback(self):
        x = np.arange(4, dtype=float)
        z = np.array([-1.0, 0.0, 0.0, 1.0])     # dz≈0 at the crossing node
        x_sh, idx = last_wet_dry_crossing(x, z)
        assert idx is not None


class TestVolumeAboveDatum:
    def test_triangle_area(self):
        x = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 2.0, 0.0])           # triangle, base 2, height 2 → area 2
        assert volume_above_datum(x, z) == pytest.approx(2.0)

    def test_ignores_below_datum(self):
        x = np.arange(3.0)
        z = np.array([-5.0, -5.0, -5.0])
        assert volume_above_datum(x, z) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fit_profile — generate then recover
# ---------------------------------------------------------------------------

def _fit(x, z, design_BE):
    return fit_profile(x, z, design_BE)


class TestFitRoundTrip:
    @pytest.mark.parametrize("shape", ["triangular", "trapezoidal", "gaussian"])
    def test_recovers_features(self, shape):
        dune = DuneSpec(shape=shape, crest_elevation=5.0, front_width=15.0,
                        back_width=20.0, top_width=12.0, sigma=7.0)
        x, z, truth = make_profile(berm_elevation=2.0, berm_width=30.0,
                                   upland_elevation=1.0, dune=dune)
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
        assert m.fit_quality < 0.25      # idealized hugs a clean piecewise profile

    def test_noise_robust(self):
        dune = DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0)
        x, z, truth = make_profile(dune=dune, upland_elevation=1.0, noise=0.05, seed=3)
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.LOW_UPLAND.value
        assert m.dune_crest_elevation == pytest.approx(5.0, abs=0.4)
        assert m.dune_crest_x == pytest.approx(truth.dune_crest_x, abs=5.0)


class TestClassification:
    def test_low_upland(self):
        x, z, _ = make_profile(berm_elevation=2.0, upland_elevation=1.0,
                               dune=DuneSpec("triangular", crest_elevation=5.0))
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.morph_type == MorphType.LOW_UPLAND.value

    def test_low_berm(self):
        x, z, _ = make_profile(berm_elevation=2.0, upland_elevation=2.5,
                               dune=DuneSpec("triangular", crest_elevation=5.0))
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
        x, z, _ = make_profile(berm_elevation=2.0, upland_elevation=1.0,
                               dune=DuneSpec("triangular", crest_elevation=5.0,
                                             front_width=3.0, back_width=20.0))
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.dune_scarp is True
        assert m.scarp_height > 0.0

    def test_gentle_front_no_dune_scarp(self):
        x, z, _ = make_profile(berm_elevation=2.0, upland_elevation=1.0,
                               dune=DuneSpec("triangular", crest_elevation=5.0,
                                             front_width=25.0, back_width=20.0))
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.dune_scarp is False

    def test_no_berm_flags_berm_scarp(self):
        x, z, _ = make_profile(berm_elevation=2.0, berm_width=0.0, upland_elevation=1.0,
                               dune=DuneSpec("triangular", crest_elevation=5.0))
        m, _ = _fit(x, z, design_BE=2.0)
        assert m.berm_scarp is True


class TestIdealizedProfile:
    def test_evaluate_matches_raw_clean(self):
        x, z, _ = make_profile(dune=DuneSpec("triangular"), upland_elevation=1.0)
        _, ideal = _fit(x, z, design_BE=2.0)
        zi = ideal.evaluate(x)
        dry = z > 0.0
        assert np.sqrt(np.mean((z[dry] - zi[dry]) ** 2)) < 0.25
