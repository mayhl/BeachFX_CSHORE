"""Golden-parameter regression for post-storm recovery + opt-in PNG gallery.

Each synthetic case in ``tests/recovery_gallery.CASES`` has a committed golden
JSON (``tests/goldens/recovery/<name>.json``) capturing the recovery fractions
and recovered-bed volumes.  This module recomputes the recovery fresh and asserts
the fractions + volumes match the golden within tolerance, so a drift in
``_recovery_fraction`` or ``recovered_bed`` fails loudly.  It also checks the
recovery invariants (fraction range/monotonicity, below-berm mask, bed bounds).

Regenerate the goldens after an intentional mechanics change:

    REGEN_RECOVERY_GOLDENS=1 uv run pytest tests/erosion/test_recovery_golden.py

Render the visual gallery (fresh fan + regenerated-from-params fan):

    uv run pytest tests/erosion/test_recovery_golden.py --plot
"""

import math
import os

import numpy as np
import pytest

from tests import recovery_gallery as rg

_ABS = 1e-9


def _close(a, b) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= _ABS + _ABS * abs(b)


@pytest.mark.parametrize("case", rg.CASES, ids=lambda c: c.name)
def test_recovery_golden(case, plot):
    x, zb_pre, zb_post, fractions, recovered = rg.run_case(case)
    fresh = rg.case_to_golden(case, fractions, recovered, x)

    if os.environ.get("REGEN_RECOVERY_GOLDENS"):
        rg.save_golden(fresh)
        pytest.skip(f"regenerated golden for {case.name}")

    golden = rg.load_golden(case.name)

    assert len(fresh["fractions"]) == len(golden["fractions"])
    for gf, wf in zip(fresh["fractions"], golden["fractions"]):
        assert _close(gf, wf), f"{case.name} fraction {gf} != golden {wf}"
    for gv, wv in zip(fresh["volumes"], golden["volumes"]):
        assert _close(gv, wv), f"{case.name} volume {gv} != golden {wv}"

    # --- Recovery invariants (independent of the golden values) ---
    # Fractions are in [0, 1] and non-decreasing in elapsed time.
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert all(b >= a - _ABS for a, b in zip(fractions, fractions[1:]))

    lo = np.minimum(zb_post, zb_pre)
    hi = np.maximum(zb_post, zb_pre)
    for f, rec in zip(fractions, recovered):
        # Every recovered node sits between the post- and pre-storm beds.
        assert np.all(rec <= hi + _ABS) and np.all(rec >= lo - _ABS)
        # Above the berm mask, nodes are frozen at the post-storm bed.
        if case.z_berm is not None:
            above = zb_post >= case.z_berm
            np.testing.assert_allclose(rec[above], zb_post[above], atol=_ABS)

    # These scenarios all rebuild (pre volume ≥ post volume), so recovered volume
    # climbs monotonically toward the target.
    vols = fresh["volumes"]
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))

    if plot:
        rg.save_recovery_png(case)
        rg.regenerate_png(golden)


def test_regenerate_matches_fresh_run():
    """The recovered fan rebuilt from saved fractions equals the fresh run's fan."""
    case = rg.CASES[0]
    x, zb_pre, zb_post, fractions, recovered = rg.run_case(case)
    golden = rg.case_to_golden(case, fractions, recovered, x)

    for f, rec in zip(golden["fractions"], recovered):
        rebuilt = rg.recovered_bed(zb_post, zb_pre, f, case.z_berm)
        np.testing.assert_allclose(rebuilt, rec, atol=1e-12)
