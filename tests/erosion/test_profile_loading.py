"""load_raw_profile: unit conversion, frame reversal, and the working-grid resample."""

import os

import numpy as np
import pytest

from erosion.profile import load_raw_profile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_P0 = os.path.join(_ROOT, "data/profiles/reach1_p0.csv")


class TestGridResample:
    def test_native_default_keeps_survey_nodes(self):
        raw = load_raw_profile(_P0, 0.3)
        assert len(raw["x"]) == 521
        assert np.diff(raw["x"])[0] == pytest.approx(3.048)

    def test_dx_produces_an_equipartitioned_grid(self):
        raw = load_raw_profile(_P0, 0.3, dx=1.0)
        d = np.diff(raw["x"])
        assert np.allclose(d, 1.0)
        assert raw["x"][0] == 0.0

    def test_resample_preserves_span_and_bed(self):
        nat = load_raw_profile(_P0, 0.3)
        fine = load_raw_profile(_P0, 0.3, dx=1.0)
        assert fine["x"][-1] == pytest.approx(nat["x"][-1], abs=1.0)
        # the fine bed IS the linear interpolant of the survey -- no smoothing,
        # no volume change beyond the piecewise-linear reading of the same data
        expect = np.interp(fine["x"], nat["x"], nat["z"])
        np.testing.assert_allclose(fine["z"], expect, atol=1e-12)
