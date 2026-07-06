"""Unit tests for ``review_profile`` — the QC pass that flags fitted profiles
needing manual classification."""

import numpy as np

from erosion.metrics import ProfileMetrics, ReviewFlag, fit_profile, review_profile
from tests.synthetic import DuneSpec, make_profile


def _review(spec, be):
    x, z, _ = make_profile(**spec)
    m, ideal = fit_profile(x, z, be)
    return review_profile(x, z, m, ideal), m


def test_clean_single_dune_passes():
    spec = dict(
        berm_elevation=2.0,
        upland_elevation=3.0,
        dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
    )
    r, _ = _review(spec, 2.0)
    assert not r.needs_review
    assert r.flags == []


def test_clean_high_upland_passes():
    r, _ = _review(dict(berm_elevation=2.0, upland_elevation=4.0, dune=None), 2.0)
    assert not r.needs_review


def test_two_dune_flags_multi_dune():
    spec = dict(
        berm_elevation=2.0,
        berm_width=30.0,
        upland_elevation=2.0,
        x_max=400.0,
        dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
        dune2=DuneSpec("triangular", crest_elevation=6.0, front_width=15.0, back_width=20.0),
        dune2_gap=40.0,
    )
    r, _ = _review(spec, 2.0)
    assert r.needs_review
    assert ReviewFlag.MULTI_DUNE.value in r.flags


def test_marginal_dune_flags_marginal_class():
    # Crest clears the upland by ~0.55 m — just past the 0.5 m dune threshold, so
    # the dune / no-dune call is on a knife-edge.
    spec = dict(
        berm_elevation=2.0,
        upland_elevation=3.0,
        dune=DuneSpec("triangular", crest_elevation=3.55, front_width=15.0, back_width=20.0),
    )
    r, m = _review(spec, 2.0)
    assert not np.isnan(m.dune_crest_elevation)
    assert ReviewFlag.MARGINAL_CLASS.value in r.flags


def test_indeterminate_when_no_fit():
    r = review_profile(np.arange(10.0), np.zeros(10), ProfileMetrics(), None)
    assert r.needs_review
    assert r.flags == [ReviewFlag.INDETERMINATE.value]


def test_low_confidence_flag_on_high_rms():
    # A tolerance of 0 forces any nonzero-RMS fit to read as low-confidence.
    x, z, _ = make_profile(
        berm_elevation=2.0,
        upland_elevation=3.0,
        dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
        noise=0.05,
        seed=1,
    )
    m, ideal = fit_profile(x, z, 2.0)
    r = review_profile(x, z, m, ideal, rms_tol=0.0)
    assert ReviewFlag.LOW_FIT_CONFIDENCE.value in r.flags
