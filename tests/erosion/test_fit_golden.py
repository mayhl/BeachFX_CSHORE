"""Golden-parameter regression for the profile fitter + opt-in PNG gallery.

Each synthetic case in ``tests/fit_gallery.CASES`` has a committed golden JSON
(``tests/goldens/fits/<name>.json``) capturing its full fit.  This module fits
the case fresh and asserts every metric + idealized knot matches the golden
within tolerance, so a drift in ``fit_profile`` fails loudly.

Regenerate the goldens after an intentional fitter change:

    REGEN_FIT_GOLDENS=1 uv run pytest tests/erosion/test_fit_golden.py

Render the visual gallery (fresh fit + regenerated-from-params overlay):

    uv run pytest tests/erosion/test_fit_golden.py --plot
"""

import math
import os

import pytest

from tests import fit_gallery as fg

_ABS = 1e-6
_REL = 1e-6


def _close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= _ABS + _REL * abs(b)
    return a == b


@pytest.mark.parametrize("case", fg.CASES, ids=lambda c: c.name)
def test_fit_golden(case, plot):
    x, zb, m, ideal, _ = fg.run_case(case)
    fresh = fg.case_to_golden(case, m, ideal)

    if os.environ.get("REGEN_FIT_GOLDENS"):
        fg.save_golden(fresh)
        pytest.skip(f"regenerated golden for {case.name}")

    golden = fg.load_golden(case.name)

    # Every metric field matches within tolerance.
    for field, want in golden["metrics"].items():
        got = fresh["metrics"][field]
        assert _close(got, want), f"{case.name}.{field}: {got!r} != golden {want!r}"

    # Idealized knots match (same landmarks, same positions).
    assert (fresh["knots_x"] is None) == (golden["knots_x"] is None)
    if golden["knots_x"] is not None:
        assert len(fresh["knots_x"]) == len(golden["knots_x"])
        for gx, wx in zip(fresh["knots_x"], golden["knots_x"]):
            assert _close(gx, wx), f"{case.name} knot_x {gx} != {wx}"
        for gz, wz in zip(fresh["knots_z"], golden["knots_z"]):
            assert _close(gz, wz), f"{case.name} knot_z {gz} != {wz}"

    if plot:
        fg.save_fit_png(case)
        fg.regenerate_png(golden)


def test_regenerate_matches_fresh_fit():
    """The idealized curve rebuilt from saved knots equals the fresh fit's curve."""
    import numpy as np

    from erosion.metrics import IdealizedProfile

    case = fg.CASES[0]
    x, zb, m, ideal, _ = fg.run_case(case)
    golden = fg.case_to_golden(case, m, ideal)

    rebuilt = IdealizedProfile(np.asarray(golden["knots_x"]), np.asarray(golden["knots_z"]))
    np.testing.assert_allclose(rebuilt.evaluate(x), ideal.evaluate(x), atol=1e-9)


@pytest.mark.parametrize("case", fg.CASES, ids=lambda c: c.name)
def test_fit_invariants(case):
    """Physical invariants no golden regeneration can rubber-stamp past.

    The value comparison above accepts whatever a regeneration wrote (and treats
    NaN == NaN as a pass), so a fit gone physically wrong could be frozen in as
    the new golden.  These hold for EVERY case, golden or fresh.
    """
    import math

    x, zb, m, ideal, _ = fg.run_case(case)

    assert m.berm_width >= 0.0
    assert m.dune_width >= 0.0
    for w in (m.dune_front_width, m.dune_back_width, m.dune_top_width):
        assert w >= 0.0

    if not math.isnan(m.berm_x):
        assert x[0] <= m.berm_x <= x[-1]
        assert m.berm_width > 0.0  # berm_x and berm_width assert existence together
    if not math.isnan(m.berm_elevation) and not math.isnan(m.dune_crest_elevation):
        assert m.dune_crest_elevation >= m.berm_elevation
    if not math.isnan(m.shoreline_x) and not math.isnan(m.dune_crest_x):
        assert m.shoreline_x <= m.dune_crest_x
    if not math.isnan(m.scarp_height):
        assert m.scarp_height >= 0.0
    if not math.isnan(m.fit_quality):
        assert m.fit_quality >= 0.0
    if ideal is not None:
        # knots ordered and inside the domain
        kx = list(ideal.knots_x)
        assert all(a <= b for a, b in zip(kx, kx[1:]))
        assert x[0] <= kx[0] and kx[-1] <= x[-1]
