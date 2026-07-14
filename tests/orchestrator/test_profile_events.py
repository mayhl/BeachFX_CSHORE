"""Unit tests for all ProfileEvent concrete types.

Covers: ErosionTick, StormResponse, Recovery, FullNourishment, PartialNourishment.
"""

import numpy as np
import pytest

from erosion.profile import (
    ErosionTick,
    FullNourishment,
    PartialNourishment,
    Recovery,
    StormResponse,
)
from erosion.runner.base import CSHOREResult
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

        Which label each CAMPAIGN KIND gets is the orchestrator's call, asserted where it
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
