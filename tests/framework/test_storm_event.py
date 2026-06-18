"""Unit tests for storm schedule, _recovery_fraction, run_parallel_cshore, and
classify_storm_response."""
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from framework.config import ReachConfig
from framework.metrics import MorphType, ProfileMetrics
from framework.profile import Profile
from framework.runner.base import CSHOREResult, CSHORERunner
from framework.runner.mock import MockCSHORERunner
from framework.storm import (
    _recovery_fraction, build_storm_schedule,
    classify_storm_response, run_parallel_cshore,
)
from framework.types import SnapshotLabel, StormResponseType

SIM_START = datetime(2030, 1, 1)


def _p(n: int = 100) -> Profile:
    x = np.linspace(0, 200, n)
    return Profile(id="p0", x=x, zb=np.linspace(-2.0, 3.0, n), d50=0.3)


def _storms(n: int = 3, lifecycle: int = 0) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp(SIM_START)
    for i in range(n):
        day = (i + 1) * 20
        for dt_h in range(0, 13, 6):
            rows.append(dict(lifecycle=lifecycle, storm_id=f"S{i:02d}", hydro_tstp=dt_h,
                             date=t0 + pd.Timedelta(days=day, hours=dt_h),
                             wave_height=1.2, wave_peak_period=10.0,
                             water_elevation=0.3, wave_direction=0.0))
    return pd.DataFrame(rows)


def _forcing() -> dict:
    return {"timebc_wave": np.array([0.0, 3600.0]),
            "Hs": np.array([1.0, 2.0]), "Hrms": np.array([0.7, 1.4]),
            "Tp": np.array([8.0, 10.0]), "Wsetup": np.zeros(2),
            "swlbc": np.array([0.0, 0.3]), "angle": np.zeros(2)}


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
    def test_linear_at_zero(self):
        assert _recovery_fraction(0.0, 10.0, "linear") == pytest.approx(0.0)

    def test_linear_at_half(self):
        assert _recovery_fraction(5.0, 10.0, "linear") == pytest.approx(0.5)

    def test_linear_at_full(self):
        assert _recovery_fraction(10.0, 10.0, "linear") == pytest.approx(1.0)

    def test_linear_clamped_past_T(self):
        assert _recovery_fraction(999.0, 10.0, "linear") == pytest.approx(1.0)

    def test_exponential_at_T90_is_90pct(self):
        T90 = 21.0
        assert _recovery_fraction(T90, T90, "exponential") == pytest.approx(0.9, abs=1e-10)

    def test_exponential_at_zero(self):
        assert _recovery_fraction(0.0, 10.0, "exponential") == pytest.approx(0.0)

    def test_negative_dt_is_zero(self):
        assert _recovery_fraction(-1.0, 10.0, "linear") == pytest.approx(0.0)


class TestRunParallelCshore:
    def test_prestorm_snapshot_taken(self):
        p = _p()
        run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                            runner=MockCSHORERunner(), cfg=ReachConfig())
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.PreStorm in labels

    def test_poststorm_snapshot_taken(self):
        p = _p()
        run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                            runner=MockCSHORERunner(), cfg=ReachConfig())
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.PostStorm in labels

    def test_prestorm_before_poststorm(self):
        p = _p()
        run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                            runner=MockCSHORERunner(), cfg=ReachConfig())
        pre_idx  = next(i for i, s in enumerate(p.snapshots) if s.label == SnapshotLabel.PreStorm)
        post_idx = next(i for i, s in enumerate(p.snapshots) if s.label == SnapshotLabel.PostStorm)
        assert pre_idx < post_idx

    def test_zb_pre_new_same_length_as_profile(self):
        """zb_pre_new must always be on the original fixed profile grid."""
        p = _p()
        n_original = len(p.x)
        _, zb_pre_new = run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                                            runner=MockCSHORERunner(), cfg=ReachConfig())
        assert len(zb_pre_new[0]) == n_original

    def test_profile_x_unchanged(self):
        """profile.x must not be mutated by run_parallel_cshore."""
        p = _p()
        x_before = p.x.copy()
        run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                            runner=MockCSHORERunner(), cfg=ReachConfig())
        np.testing.assert_array_equal(p.x, x_before)

    def test_two_profiles_parallel(self):
        p0, p1 = _p(), _p()
        p1.id = "p1"
        results, zb_pre_new = run_parallel_cshore(
            [p0, p1], t_storm=10.0, forcing=_forcing(),
            runner=MockCSHORERunner(), cfg=ReachConfig())
        assert len(results) == 2
        assert len(zb_pre_new) == 2

    def test_zb_mutated_by_mock_runner(self):
        p = _p()
        zb_before = p.zb.copy()
        run_parallel_cshore([p], t_storm=10.0, forcing=_forcing(),
                            runner=MockCSHORERunner(), cfg=ReachConfig())
        assert not np.allclose(p.zb, zb_before)


class _FailRunner(CSHORERunner):
    """Runner that always raises — simulates CSHORE failure (e.g. overtopping)."""
    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        raise RuntimeError("CSHORE failed: profile overtopped")


class TestRunParallelCshoreFailure:
    def test_failure_returns_none_result(self):
        p = _p()
        results, _ = run_parallel_cshore([p], t_storm=5.0, forcing=_forcing(),
                                         runner=_FailRunner(), cfg=ReachConfig())
        assert results[0] is None

    def test_failure_takes_inundation_snapshot(self):
        p = _p()
        run_parallel_cshore([p], t_storm=5.0, forcing=_forcing(),
                            runner=_FailRunner(), cfg=ReachConfig())
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.INUNDATION in labels

    def test_failure_zb_unchanged(self):
        p = _p()
        zb_before = p.zb.copy()
        run_parallel_cshore([p], t_storm=5.0, forcing=_forcing(),
                            runner=_FailRunner(), cfg=ReachConfig())
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_failure_inundation_storm_response_type(self):
        p = _p()
        run_parallel_cshore([p], t_storm=5.0, forcing=_forcing(),
                            runner=_FailRunner(), cfg=ReachConfig())
        inundation_snap = next(
            s for s in p.snapshots if s.label == SnapshotLabel.INUNDATION
        )
        assert inundation_snap.storm_response_type == StormResponseType.INUNDATION

    def test_failure_zb_pre_new_is_original(self):
        p = _p()
        zb_before = p.zb.copy()
        _, zb_pre_new = run_parallel_cshore([p], t_storm=5.0, forcing=_forcing(),
                                            runner=_FailRunner(), cfg=ReachConfig())
        np.testing.assert_array_equal(zb_pre_new[0], zb_before)


def _make_metrics(
    morph_type: str,
    dune_height: float,
    upland_elevation: float,
    berm_width: float,
) -> ProfileMetrics:
    nan = float("nan")
    return ProfileMetrics(
        morph_type=morph_type,
        shoreline_x=nan, berm_width=berm_width, foreshore_slope=nan,
        dune_height=dune_height, dune_x=nan, dune_width=nan,
        dune_front_slope=nan, dune_back_slope=nan,
        upland_elevation=upland_elevation, volume_above_datum=0.0,
        scarp_present=False, max_beach_slope=nan, dune_front_resid=nan,
        n_upland_nodes=5,
    )


class TestClassifyStormResponse:
    BE = 1.8

    def test_normal_dune_intact(self):
        m_pre  = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=2.5, upland_elevation=1.0, berm_width=10.0)
        m_post = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=2.0, upland_elevation=1.0, berm_width=5.0)
        assert classify_storm_response(m_pre, m_post, self.BE) == StormResponseType.NORMAL

    def test_cat_dune_lost_low_berm(self):
        m_pre  = _make_metrics(MorphType.LOW_BERM.value, dune_height=2.5, upland_elevation=1.0, berm_width=5.0)
        m_post = _make_metrics(MorphType.LOW_BERM.value, dune_height=1.5, upland_elevation=1.0, berm_width=0.0)
        assert classify_storm_response(m_pre, m_post, self.BE) == StormResponseType.CAT_DUNE_LOST

    def test_cat_partial_low_upland_berm_survives(self):
        m_pre  = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=2.5, upland_elevation=1.0, berm_width=10.0)
        m_post = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=1.5, upland_elevation=1.0, berm_width=5.0)
        assert classify_storm_response(m_pre, m_post, self.BE) == StormResponseType.CAT_PARTIAL

    def test_cat_total_low_upland_berm_gone(self):
        m_pre  = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=2.5, upland_elevation=1.0, berm_width=10.0)
        m_post = _make_metrics(MorphType.LOW_UPLAND.value, dune_height=1.5, upland_elevation=1.0, berm_width=0.0)
        assert classify_storm_response(m_pre, m_post, self.BE) == StormResponseType.CAT_TOTAL

    def test_high_upland_no_dune_is_normal(self):
        m_pre  = _make_metrics(MorphType.HIGH_UPLAND.value, dune_height=3.0, upland_elevation=4.0, berm_width=0.0)
        m_post = _make_metrics(MorphType.HIGH_UPLAND.value, dune_height=3.0, upland_elevation=4.0, berm_width=0.0)
        assert classify_storm_response(m_pre, m_post, self.BE) == StormResponseType.NORMAL
