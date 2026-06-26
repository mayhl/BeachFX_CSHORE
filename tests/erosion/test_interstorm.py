"""Unit tests for run_interstorm (Phase 1 runner)."""
import numpy as np

from erosion.config import ReachConfig
from erosion.interstorm import SLCConfig, UniformErosionConfig, run_interstorm
from erosion.types import SnapshotLabel
from tests.builders import profile as _p


def _cfg(erosion=None, slc=None) -> ReachConfig:
    return ReachConfig(erosion=erosion, slc=slc)


class TestRunInterstorm:
    def test_no_config_no_change(self):
        p = _p()
        zb_before = p.zb.copy()
        run_interstorm([p], t_start=0.0, t_storm=30.0, cfg=_cfg())
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_t_start_eq_t_storm_immediate_return(self):
        p = _p()
        zb_before = p.zb.copy()
        run_interstorm([p], t_start=10.0, t_storm=10.0,
                       cfg=_cfg(erosion=UniformErosionConfig(rate=0.01)))
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_uniform_erosion_lowers_zb(self):
        p = _p()
        run_interstorm([p], t_start=0.0, t_storm=30.0,
                       cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, interval=30.0)))
        np.testing.assert_allclose(p.zb, -0.3, atol=1e-12)

    def test_slc_lowers_zb(self):
        p = _p()
        run_interstorm([p], t_start=0.0, t_storm=30.0,
                       cfg=_cfg(slc=SLCConfig(rate=0.005, interval=30.0)))
        np.testing.assert_allclose(p.zb, -0.15, atol=1e-12)

    def test_erosion_and_slc_combined(self):
        p = _p()
        run_interstorm([p], t_start=0.0, t_storm=30.0,
                       cfg=_cfg(
                           erosion=UniformErosionConfig(rate=0.01, interval=30.0),
                           slc=SLCConfig(rate=0.005, interval=30.0),
                       ))
        np.testing.assert_allclose(p.zb, -(0.3 + 0.15), atol=1e-12)

    def test_tick_count_matches_interval(self):
        """interval=10 over 30 days → 3 Periodic snapshots."""
        p = _p()
        run_interstorm([p], t_start=0.0, t_storm=30.0,
                       cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, interval=10.0)))
        periodic = [s for s in p.snapshots if s.label == SnapshotLabel.Periodic]
        assert len(periodic) == 3

    def test_two_profiles_both_updated(self):
        p0, p1 = _p("p0"), _p("p1")
        run_interstorm([p0, p1], t_start=0.0, t_storm=30.0,
                       cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, interval=30.0)))
        np.testing.assert_allclose(p0.zb, -0.3, atol=1e-12)
        np.testing.assert_allclose(p1.zb, -0.3, atol=1e-12)

    def test_per_profile_rate(self):
        p0, p1 = _p("p0"), _p("p1")
        cfg = _cfg(erosion=UniformErosionConfig(
            per_profile_rates={"p0": 0.01, "p1": 0.02}, interval=30.0,
        ))
        run_interstorm([p0, p1], t_start=0.0, t_storm=30.0, cfg=cfg)
        np.testing.assert_allclose(p0.zb, -0.3, atol=1e-12)
        np.testing.assert_allclose(p1.zb, -0.6, atol=1e-12)

    def test_partial_last_tick(self):
        """interval=10, t_storm=25 → 2 full ticks + 1 partial tick of 5 days."""
        p = _p()
        run_interstorm([p], t_start=0.0, t_storm=25.0,
                       cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, interval=10.0)))
        np.testing.assert_allclose(p.zb, -0.25, atol=1e-12)
