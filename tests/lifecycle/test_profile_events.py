"""Unit tests for the profile-scope physics: every ProfileEvent concrete type
(ErosionTick, StormResponse, Recovery incl. the z_berm mask, Full/PartialNourishment),
the ``run_interstorm`` tick runner over them, and per-profile z_berm resolution."""

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.interstorm import SLCConfig, UniformErosionConfig, run_interstorm
from erosion.metrics import ProfileMetrics
from erosion.nourishment.execute import _resolve_z_berm
from erosion.profile import (
    ErosionTick,
    FullNourishment,
    PartialNourishment,
    ProfileGeometryConfig,
    Recovery,
    StormResponse,
)
from erosion.runner.base import CSHOREResult
from erosion.storm import StormConfig
from erosion.types import SnapshotLabel
from tests.builders import profile


def _p(n: int = 100):
    return profile(n=n, x_max=200.0)


def _result(n: int = 80) -> CSHOREResult:
    x_new = np.linspace(5, 185, n)
    return CSHOREResult(zb=np.ones(n) * -0.3, x=x_new, eta=np.zeros(n), Hs=np.zeros(n), runup_m=0.1)


class TestErosionTick:
    def test_lowers_zb_by_erosion(self):
        p = _p()
        ErosionTick(t=1.0, dz_erosion=0.05).apply(p)
        np.testing.assert_allclose(p.zb, -0.05, atol=1e-14)

    def test_lowers_zb_by_slc(self):
        p = _p()
        ErosionTick(t=1.0, dz_slc=0.02).apply(p)
        np.testing.assert_allclose(p.zb, -0.02, atol=1e-14)

    def test_combined_erosion_and_slc(self):
        p = _p()
        ErosionTick(t=1.0, dz_erosion=0.05, dz_slc=0.02).apply(p)
        np.testing.assert_allclose(p.zb, -0.07, atol=1e-14)

    def test_takes_periodic_snapshot(self):
        p = _p()
        ErosionTick(t=5.0, dz_erosion=0.01).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.Periodic

    def test_snapshot_t_correct(self):
        p = _p()
        ErosionTick(t=7.0, dz_erosion=0.01).apply(p)
        assert p.snapshots[-1].t == pytest.approx(7.0)

    def test_snapshot_captures_updated_zb(self):
        p = _p()
        ErosionTick(t=1.0, dz_erosion=0.05).apply(p)
        np.testing.assert_allclose(p.snapshots[-1].zb, -0.05, atol=1e-14)

    def test_cumulative_ticks(self):
        p = _p()
        for i in range(5):
            ErosionTick(t=float(i + 1), dz_erosion=0.01).apply(p)
        np.testing.assert_allclose(p.zb, -0.05, atol=1e-14)

    def test_snapshot_is_copy(self):
        p = _p()
        ErosionTick(t=1.0, dz_erosion=0.01).apply(p)
        stored = p.snapshots[-1].zb.copy()
        p.zb[:] = 99.0
        np.testing.assert_array_equal(p.snapshots[-1].zb, stored)


class TestStormResponse:
    def test_x_unchanged(self):
        """profile.x must never be mutated — snapshots share the original fixed grid."""
        p = _p()
        x_before = p.x.copy()
        StormResponse(t=10.0, result=_result()).apply(p)
        np.testing.assert_array_equal(p.x, x_before)

    def test_zb_interpolated_onto_original_grid(self):
        """zb is re-sampled from result.x onto the original profile.x."""
        p = _p()
        r = _result()
        StormResponse(t=10.0, result=r).apply(p)
        expected = np.interp(p.x, r.x, r.zb, left=r.zb[0], right=r.zb[-1])
        np.testing.assert_allclose(p.zb, expected, atol=1e-14)

    def test_zb_length_unchanged(self):
        p = _p()
        n_before = len(p.zb)
        StormResponse(t=10.0, result=_result()).apply(p)
        assert len(p.zb) == n_before

    def test_result_mutation_does_not_affect_profile(self):
        p = _p()
        r = _result()
        StormResponse(t=10.0, result=r).apply(p)
        zb_after = p.zb.copy()
        r.zb[:] = 99.0
        np.testing.assert_array_equal(p.zb, zb_after)

    def test_takes_poststorm_snapshot(self):
        p = _p()
        StormResponse(t=10.0, result=_result()).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.PostStorm


class TestRecovery:
    def _pre_post(self, n: int = 100):
        return np.zeros(n), np.ones(n) * 2.0

    def test_fraction_zero_keeps_post_storm(self):
        p = _p()
        zb_post, zb_pre = self._pre_post()
        p.zb = zb_post.copy()
        Recovery(t=1.0, fraction=0.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, zb_post, atol=1e-14)

    def test_fraction_one_restores_pre_storm(self):
        p = _p()
        zb_post, zb_pre = self._pre_post()
        p.zb = zb_post.copy()
        Recovery(t=21.0, fraction=1.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, zb_pre, atol=1e-14)

    def test_fraction_half_midpoint(self):
        p = _p()
        zb_post, zb_pre = self._pre_post()
        p.zb = zb_post.copy()
        Recovery(t=10.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-14)

    def test_z_berm_mask_below_berm(self):
        n = 10
        zb_post = np.zeros(n)
        zb_pre = np.ones(n) * 2.0
        p = _p(n)
        p.zb = zb_post.copy()
        Recovery(t=5.0, fraction=1.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre, z_berm=1.0).apply(
            p
        )
        # zb_post=0 < z_berm=1 → all nodes below berm → recover to zb_pre=2
        np.testing.assert_allclose(p.zb, zb_pre, atol=1e-14)

    def test_z_berm_mask_above_berm_unchanged(self):
        n = 10
        zb_post = np.ones(n) * 1.5  # above z_berm=1.0
        p = _p(n)
        p.zb = zb_post.copy()
        Recovery(
            t=5.0, fraction=1.0, zb_post_storm=zb_post, zb_pre_storm=np.ones(n) * 3.0, z_berm=1.0
        ).apply(p)
        # zb_post=1.5 >= z_berm=1 → mask=False → zb unchanged
        np.testing.assert_allclose(p.zb, zb_post, atol=1e-14)

    def test_takes_rec_snapshot(self):
        p = _p()
        zb_post, zb_pre = self._pre_post()
        Recovery(t=5.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.REC

    def test_interrupted_takes_recs_snapshot(self):
        """A storm-forced (interrupted) recovery snapshots RECS, not REC."""
        p = _p()
        zb_post, zb_pre = self._pre_post()
        Recovery(
            t=5.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre, interrupted=True
        ).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.RECS

    def test_snapshot_t_stored(self):
        p = _p()
        zb_post, zb_pre = self._pre_post()
        Recovery(t=7.5, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        assert p.snapshots[-1].t == pytest.approx(7.5)


class TestFullNourishment:
    def test_sets_template(self):
        p = _p()
        template = np.ones(len(p.x)) * 3.0
        FullNourishment(t=10.0, template_zb=template).apply(p)
        np.testing.assert_array_equal(p.zb, template)

    def test_stored_as_copy(self):
        p = _p()
        template = np.ones(len(p.x)) * 3.0
        FullNourishment(t=10.0, template_zb=template).apply(p)
        template[:] = 0.0
        assert not np.allclose(p.zb, 0.0)

    def test_takes_een_snapshot_by_default(self):
        """Post-storm is the default campaign, so the end marker is EEN.

        Which label each CAMPAIGN KIND gets is the interval loop's call, asserted where it
        is made (test_event_sequences / test_nourishment_cycle) — passing a label in here
        and reading it back would only re-assert the dataclass field.
        """
        p = _p()
        FullNourishment(t=10.0, template_zb=np.ones(len(p.x))).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.EEN


class TestPartialNourishment:
    def test_fraction_one_sets_template(self):
        p = _p()
        template = np.ones(len(p.x)) * 3.0
        PartialNourishment(t=10.0, template_zb=template, fraction=1.0).apply(p)
        np.testing.assert_allclose(p.zb, template, atol=1e-14)

    def test_fraction_zero_no_change(self):
        p = _p()
        original = p.zb.copy()
        PartialNourishment(t=10.0, template_zb=np.ones(len(p.x)) * 3.0, fraction=0.0).apply(p)
        np.testing.assert_allclose(p.zb, original, atol=1e-14)

    def test_fraction_half_midpoint(self):
        n = 100
        p = _p(n)
        p.zb = np.zeros(n)
        template = np.ones(n) * 2.0
        PartialNourishment(t=10.0, template_zb=template, fraction=0.5).apply(p)
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-14)

    def test_takes_eens_snapshot_by_default(self):
        """A cut campaign closes its segment rather than leaving it open — post-storm is
        the default kind, so EENS."""
        p = _p()
        PartialNourishment(t=10.0, template_zb=np.ones(len(p.x)), fraction=0.5).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.EENS


class TestResolveZBerm:
    """Per-profile z_berm derivation: explicit config, ref fit, geometry, warn."""

    def test_explicit_config_wins(self):
        p = _p()
        p.ref_metrics = ProfileMetrics(berm_elevation=1.8)
        cfg = ReachConfig(storm=StormConfig(z_berm=2.5))  # bare number: ft in, stored m
        assert _resolve_z_berm(p, cfg) == pytest.approx(float(cfg.storm.z_berm))

    def test_derives_from_ref_metrics(self):
        p = _p()
        p.ref_metrics = ProfileMetrics(berm_elevation=1.8)
        p.geometry = ProfileGeometryConfig(berm_elevation=2.0)
        assert _resolve_z_berm(p, ReachConfig()) == pytest.approx(1.8)

    def test_falls_back_to_geometry(self):
        p = _p()
        p.ref_metrics = ProfileMetrics()  # berm_elevation NaN — fit found no berm
        p.geometry = ProfileGeometryConfig(berm_elevation=2.0)  # bare number: ft in, stored m
        assert _resolve_z_berm(p, ReachConfig()) == pytest.approx(float(p.geometry.berm_elevation))

    def test_warns_and_blends_all_when_unknown(self, caplog):
        p = _p()
        with caplog.at_level("WARNING"):
            assert _resolve_z_berm(p, ReachConfig()) is None
        assert "no berm elevation" in caplog.text


def _cfg(erosion=None, slc=None) -> ReachConfig:
    return ReachConfig(erosion=erosion, slc=slc)


class TestRunInterstorm:
    def test_no_config_no_change(self):
        p = profile()
        zb_before = p.zb.copy()
        run_interstorm([p], t_start=0.0, t_end=30.0, cfg=_cfg())
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_t_start_eq_t_end_immediate_return(self):
        p = profile()
        zb_before = p.zb.copy()
        run_interstorm(
            [p], t_start=10.0, t_end=10.0, cfg=_cfg(erosion=UniformErosionConfig(rate=0.01))
        )
        np.testing.assert_array_equal(p.zb, zb_before)

    def test_uniform_erosion_lowers_zb(self):
        p = profile()
        run_interstorm(
            [p],
            t_start=0.0,
            t_end=30.0,
            cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, tick_days=30.0)),
        )
        np.testing.assert_allclose(p.zb, -0.3, atol=1e-12)

    def test_slc_lowers_zb(self):
        p = profile()
        run_interstorm(
            [p], t_start=0.0, t_end=30.0, cfg=_cfg(slc=SLCConfig(rate=0.005, tick_days=30.0))
        )
        np.testing.assert_allclose(p.zb, -0.15, atol=1e-12)

    def test_erosion_and_slc_combined(self):
        p = profile()
        run_interstorm(
            [p],
            t_start=0.0,
            t_end=30.0,
            cfg=_cfg(
                erosion=UniformErosionConfig(rate=0.01, tick_days=30.0),
                slc=SLCConfig(rate=0.005, tick_days=30.0),
            ),
        )
        np.testing.assert_allclose(p.zb, -(0.3 + 0.15), atol=1e-12)

    def test_tick_count_matches_interval(self):
        """tick_days=10 over 30 days → 3 Periodic snapshots."""
        p = profile()
        run_interstorm(
            [p],
            t_start=0.0,
            t_end=30.0,
            cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, tick_days=10.0)),
        )
        periodic = [s for s in p.snapshots if s.label == SnapshotLabel.Periodic]
        assert len(periodic) == 3

    def test_two_profiles_both_updated(self):
        p0, p1 = profile("p0"), profile("p1")
        run_interstorm(
            [p0, p1],
            t_start=0.0,
            t_end=30.0,
            cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, tick_days=30.0)),
        )
        np.testing.assert_allclose(p0.zb, -0.3, atol=1e-12)
        np.testing.assert_allclose(p1.zb, -0.3, atol=1e-12)

    def test_per_profile_rate(self):
        p0, p1 = profile("p0"), profile("p1")
        cfg = _cfg(
            erosion=UniformErosionConfig(
                per_profile_rates={"p0": 0.01, "p1": 0.02},
                tick_days=30.0,
            )
        )
        run_interstorm([p0, p1], t_start=0.0, t_end=30.0, cfg=cfg)
        np.testing.assert_allclose(p0.zb, -0.3, atol=1e-12)
        np.testing.assert_allclose(p1.zb, -0.6, atol=1e-12)

    def test_partial_last_tick(self):
        """tick_days=10, t_end=25 → 2 full ticks + 1 partial tick of 5 days."""
        p = profile()
        run_interstorm(
            [p],
            t_start=0.0,
            t_end=25.0,
            cfg=_cfg(erosion=UniformErosionConfig(rate=0.01, tick_days=10.0)),
        )
        np.testing.assert_allclose(p.zb, -0.25, atol=1e-12)
