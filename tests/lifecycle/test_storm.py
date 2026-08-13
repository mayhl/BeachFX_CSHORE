"""Unit tests for storm schedule, _recovery_fraction, run_parallel_cshore, and
classify_storm_response."""

import math

import numpy as np
import pandas as pd
import pytest

from erosion.config import ReachConfig
from erosion.metrics import MorphType, ProfileMetrics
from erosion.profile import Profile
from erosion.runner import MockCSHORERunner
from erosion.runner.base import CSHOREResult, CSHORERunner
from erosion.storm import (
    _recovery_fraction,
    build_storm_schedule,
    classify_storm_response,
    run_parallel_cshore,
)
from erosion.types import StormResponseType
from tests.builders import SIM_START, profile
from tests.builders import forcing as _forcing
from tests.builders import storms as _storms


def _p(n: int = 100):
    return profile(n=n, x_max=200.0, zb="ramp")


class TestBuildStormSchedule:
    def test_correct_count(self):
        assert len(build_storm_schedule(_storms(3), SIM_START, ReachConfig())) == 3

    def test_sorted_by_t(self):
        schedule = build_storm_schedule(_storms(5), SIM_START, ReachConfig())
        ts = [r.t for r in schedule]
        assert ts == sorted(ts)

    def test_t_days_correct(self):
        schedule = build_storm_schedule(_storms(3), SIM_START, ReachConfig())
        expected = [20.0, 40.0, 60.0]
        for rec, exp in zip(schedule, expected):
            assert rec.t == pytest.approx(exp)

    def test_lifecycle_filter(self):
        df0 = _storms(2, lifecycle=0)
        df1 = _storms(3, lifecycle=1)
        combined = pd.concat([df0, df1], ignore_index=True)
        schedule = build_storm_schedule(combined, SIM_START, ReachConfig(), lifecycle=0)
        assert len(schedule) == 2

    def test_forcing_keys(self):
        schedule = build_storm_schedule(_storms(1), SIM_START, ReachConfig())
        required = {"timebc_wave", "Hs", "Hrms", "Tp", "Wsetup", "swlbc", "angle"}
        assert required.issubset(schedule[0].forcing.keys())

    def test_hrms_is_hs_over_sqrt2(self):
        schedule = build_storm_schedule(_storms(1), SIM_START, ReachConfig())
        f = schedule[0].forcing
        np.testing.assert_allclose(f["Hrms"], f["Hs"] / math.sqrt(2.0), rtol=1e-10)


class TestRecoveryFraction:
    @pytest.mark.parametrize(
        "dt, T, model, expected",
        [
            (0.0, 10.0, "linear", 0.0),  # start of recovery
            (5.0, 10.0, "linear", 0.5),  # halfway
            (10.0, 10.0, "linear", 1.0),  # complete at T
            (999.0, 10.0, "linear", 1.0),  # clamped past T
            (0.0, 10.0, "exponential", 0.0),  # start of recovery
            (21.0, 21.0, "exponential", 0.9),  # T_recover == T90 → 90%
            (-1.0, 10.0, "linear", 0.0),  # negative dt → no recovery
        ],
        ids=["lin0", "lin_half", "lin_full", "lin_clamp", "exp0", "exp_T90", "neg_dt"],
    )
    def test_recovery_fraction(self, dt, T, model, expected):
        assert _recovery_fraction(dt, T, model) == pytest.approx(expected, abs=1e-10)


class TestRunParallelCshore:
    def test_zb_pre_new_same_length_as_profile(self):
        """zb_pre_new must always be on the original fixed profile grid."""
        p = _p()
        n_original = len(p.x)
        outcomes = run_parallel_cshore(
            [p], t_storm=10.0, forcing=_forcing(), runner=MockCSHORERunner(), cfg=ReachConfig()
        )
        assert len(outcomes[0].zb_pre) == n_original

    def test_profile_x_unchanged(self):
        """profile.x must not be mutated by run_parallel_cshore."""
        p = _p()
        x_before = p.x.copy()
        run_parallel_cshore(
            [p], t_storm=10.0, forcing=_forcing(), runner=MockCSHORERunner(), cfg=ReachConfig()
        )
        np.testing.assert_array_equal(p.x, x_before)

    def test_two_profiles_parallel(self):
        p0, p1 = _p(), _p()
        p1.id = "p1"
        outcomes = run_parallel_cshore(
            [p0, p1], t_storm=10.0, forcing=_forcing(), runner=MockCSHORERunner(), cfg=ReachConfig()
        )
        assert len(outcomes) == 2

    def test_zb_mutated_by_mock_runner(self):
        p = _p()
        zb_before = p.zb.copy()
        run_parallel_cshore(
            [p], t_storm=10.0, forcing=_forcing(), runner=MockCSHORERunner(), cfg=ReachConfig()
        )
        assert not np.allclose(p.zb, zb_before)


class _FailRunner(CSHORERunner):
    """Runner that always raises — simulates CSHORE failure (e.g. overtopping)."""

    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        raise RuntimeError("CSHORE failed: profile overtopped")


class TestRunParallelCshoreFailure:
    def test_failure_returns_none_result(self):
        p = _p()
        outcomes = run_parallel_cshore(
            [p], t_storm=5.0, forcing=_forcing(), runner=_FailRunner(), cfg=ReachConfig()
        )
        assert outcomes[0].result is None
        assert outcomes[0].inundated

    def test_failure_zb_unchanged(self):
        p = _p()
        zb_before = p.zb.copy()
        run_parallel_cshore(
            [p], t_storm=5.0, forcing=_forcing(), runner=_FailRunner(), cfg=ReachConfig()
        )
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_failure_zb_pre_new_is_original(self):
        p = _p()
        zb_before = p.zb.copy()
        outcomes = run_parallel_cshore(
            [p], t_storm=5.0, forcing=_forcing(), runner=_FailRunner(), cfg=ReachConfig()
        )
        np.testing.assert_array_equal(outcomes[0].zb_pre, zb_before)


def _make_metrics(
    morph_type: str,
    dune_crest_elevation: float,
    upland_elevation: float,
    berm_width: float,
) -> ProfileMetrics:
    # ProfileMetrics fields default to NaN/False; only the classification inputs matter.
    return ProfileMetrics(
        morph_type=morph_type,
        dune_crest_elevation=dune_crest_elevation,
        upland_elevation=upland_elevation,
        berm_width=berm_width,
    )


class TestClassifyStormResponse:
    BE = 1.8

    # Each metric tuple is (morph_type, dune_crest_elevation, upland_elevation, berm_width).
    @pytest.mark.parametrize(
        "pre, post, expected",
        [
            (
                (MorphType.LOW_UPLAND, 2.5, 1.0, 10.0),
                (MorphType.LOW_UPLAND, 2.0, 1.0, 5.0),
                StormResponseType.NORMAL,
            ),
            (
                (MorphType.LOW_BERM, 2.5, 1.0, 5.0),
                (MorphType.LOW_BERM, 1.5, 1.0, 0.0),
                StormResponseType.CAT_DUNE_LOST,
            ),
            (
                (MorphType.LOW_UPLAND, 2.5, 1.0, 10.0),
                (MorphType.LOW_UPLAND, 1.5, 1.0, 5.0),
                StormResponseType.CAT_PARTIAL,
            ),
            (
                (MorphType.LOW_UPLAND, 2.5, 1.0, 10.0),
                (MorphType.LOW_UPLAND, 1.5, 1.0, 0.0),
                StormResponseType.CAT_TOTAL,
            ),
            (
                (MorphType.HIGH_UPLAND, 3.0, 4.0, 0.0),
                (MorphType.HIGH_UPLAND, 3.0, 4.0, 0.0),
                StormResponseType.NORMAL,
            ),
        ],
        ids=["normal", "cat_dune_lost", "cat_partial", "cat_total", "high_upland"],
    )
    def test_classify(self, pre, post, expected):
        m_pre = _make_metrics(
            pre[0].value, dune_crest_elevation=pre[1], upland_elevation=pre[2], berm_width=pre[3]
        )
        m_post = _make_metrics(
            post[0].value,
            dune_crest_elevation=post[1],
            upland_elevation=post[2],
            berm_width=post[3],
        )
        assert classify_storm_response(m_pre, m_post, self.BE) == expected
