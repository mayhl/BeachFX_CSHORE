"""Unit tests for the ``fit_profile`` analysis window (``max_feature_length``).

The window bounds the fit to the primary beach+dune so a long profile's distant
terrain is not folded into one idealized dune.  ``"auto"`` (default) crops only
when a second dune rises across a real swale; it is a no-op otherwise.
"""

import numpy as np

from erosion.metrics import _analysis_cap, _dx_of, _smooth, fit_profile
from tests.synthetic import DuneSpec, make_profile

_TWO_DUNE = dict(
    berm_elevation=2.0,
    berm_width=30.0,
    upland_elevation=2.0,
    x_max=400.0,
    dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
    dune2=DuneSpec("triangular", crest_elevation=6.0, front_width=15.0, back_width=20.0),
    dune2_gap=40.0,
)
_BE = 2.0


def _crest(m):
    return m.dune_crest_x


def test_none_never_crops():
    """max_feature_length=None fits the whole profile → grabs the taller landward dune."""
    x, z, _ = make_profile(**_TWO_DUNE)
    m, _ = fit_profile(x, z, _BE, max_feature_length=None)
    # Landward (taller) dune wins without a window; it sits well past the primary.
    assert m.dune_crest_x > 140


def test_auto_isolates_primary_dune():
    """auto crops at the swale → the seaward (primary) dune is the fitted crest."""
    x, z, _ = make_profile(**_TWO_DUNE)
    off, _ = fit_profile(x, z, _BE, max_feature_length=None)
    auto, _ = fit_profile(x, z, _BE, max_feature_length="auto")
    assert auto.dune_crest_x < 120  # primary dune, seaward of the swale
    assert auto.dune_crest_x < off.dune_crest_x
    assert auto.fit_quality < off.fit_quality  # the single-dune form now matches


def test_auto_noop_single_dune():
    """One dune → auto changes nothing versus the full-profile fit."""
    kw = dict(
        berm_elevation=2.0,
        upland_elevation=3.0,
        dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
    )
    x, z, _ = make_profile(**kw)
    off, ioff = fit_profile(x, z, 2.0, max_feature_length=None)
    auto, iauto = fit_profile(x, z, 2.0, max_feature_length="auto")
    assert off.morph_type == auto.morph_type
    assert off.fit_quality == auto.fit_quality
    np.testing.assert_array_equal(ioff.knots_x, iauto.knots_x)


def test_auto_noop_flat_top_dune():
    """A single trapezoidal (flat-topped) dune is not split at its two shoulders."""
    kw = dict(
        berm_elevation=2.0,
        upland_elevation=1.0,
        dune=DuneSpec(
            "trapezoidal", crest_elevation=5.0, front_width=15.0, back_width=20.0, top_width=12.0
        ),
    )
    x, z, _ = make_profile(**kw)
    dx = _dx_of(x)
    assert _analysis_cap(x, _smooth(z, dx), dx, len(x), 0.0, "auto") is None


def test_auto_noop_shallow_dip_massif():
    """Two crests over a shallow dip (not a real swale) read as one bumpy dune."""
    x = np.arange(0.0, 301.0, 1.0)
    # crest 5 → dip to only 4 (1 m, < half the 5 m crest) → crest 5 → upland 2
    kx = [0, 20, 60, 90, 110, 140, 170, 300]
    kz = [-1, 0, 2.0, 5.0, 4.0, 5.0, 2.0, 2.0]
    z = np.interp(x, kx, kz)
    dx = _dx_of(x)
    assert _analysis_cap(x, _smooth(z, dx), dx, len(x), 0.0, "auto") is None


def test_explicit_window_caps_from_shoreline():
    """A float window crops that many metres landward of the shoreline."""
    x, z, _ = make_profile(**_TWO_DUNE)
    dx = _dx_of(x)
    zs = _smooth(z, dx)
    cap = _analysis_cap(x, zs, dx, len(x), 0.0, 60.0)
    # shoreline ~x=20; cap ends ~60 m landward of it (± a node), well before dune 2.
    assert cap is not None and 75 <= x[cap - 1] <= 85
