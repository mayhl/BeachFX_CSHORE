"""units.py conversion factors and the coercion rules the configs stand on."""

import pytest

from erosion.units import _CY_TO_M3, M3_TO_CY, parse_ufloat


class TestFactors:
    def test_cy_is_exact(self):
        # 1 yd^3 = 27 ft^3 x 0.3048^3 -- the 0.7646 literal this replaced carried
        # +59 ppm and its hand-written reciprocal disagreed with it
        assert _CY_TO_M3 == 27.0 * 0.3048**3

    def test_reporting_factor_is_the_exact_reciprocal(self):
        assert M3_TO_CY * _CY_TO_M3 == pytest.approx(1.0, rel=1e-15)

    def test_cy_round_trip_is_lossless(self):
        assert 1000.0 * _CY_TO_M3 * M3_TO_CY == pytest.approx(1000.0, rel=1e-12)


class TestCoercion:
    def test_bare_float_uses_the_ft_default(self):
        assert parse_ufloat(6.0, "m", "ft") == pytest.approx(6.0 * 0.3048)

    def test_dict_form_overrides_the_default(self):
        assert parse_ufloat({"value": 6.0, "units": "m"}, "m", "ft") == pytest.approx(6.0)

    def test_metric_context_remaps_the_ft_default(self):
        assert parse_ufloat(6.0, "m", "ft", context={"input_units": "m"}) == pytest.approx(6.0)

    def test_context_immune_field_ignores_context(self):
        # default_input None = internal unit always (e.g. CSHORE params)
        assert parse_ufloat(6.0, "m", None, context={"input_units": "m"}) == pytest.approx(6.0)

    def test_volume_context_equivalence(self):
        # cy-default fields read bare floats as m3 under the metric context
        assert parse_ufloat(100.0, "m3", "cy", context={"input_units": "m"}) == pytest.approx(100.0)
        assert parse_ufloat(100.0, "m3", "cy") == pytest.approx(100.0 * _CY_TO_M3)
