"""fall_velocity against Soulsby (1997) literature values and physical limits."""

import pytest

from erosion.runner.vfall import fall_velocity


class TestSoulsbyFallVelocity:
    def test_fine_sand_seawater(self):
        # Soulsby (1997) optimization, 0.1 mm quartz, 20 C / 35 ppt: ~8 mm/s
        assert fall_velocity(0.1, 20.0, 35.0) == pytest.approx(0.008, rel=0.10)

    def test_medium_sand_seawater(self):
        # 0.2 mm quartz in seawater: the standard ~25 mm/s textbook value
        assert fall_velocity(0.2, 20.0, 35.0) == pytest.approx(0.025, rel=0.10)

    def test_monotone_in_grain_size(self):
        ws = [fall_velocity(d, 20.0, 35.0) for d in (0.06, 0.1, 0.2, 0.3, 0.5, 1.0)]
        assert all(a < b for a, b in zip(ws, ws[1:]))

    def test_colder_water_settles_slower(self):
        # higher viscosity at low T; same grain
        assert fall_velocity(0.2, 5.0, 35.0) < fall_velocity(0.2, 25.0, 35.0)

    def test_stokes_regime_scales_with_d_squared(self):
        # For small D* the Soulsby form reduces to Stokes: ws ~ d^2
        r = fall_velocity(0.1, 20.0, 35.0) / fall_velocity(0.05, 20.0, 35.0)
        assert r == pytest.approx(4.0, rel=0.10)
