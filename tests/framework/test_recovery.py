"""Recovery-related tests — the Recovery ProfileEvent."""
import numpy as np
import pytest

from framework.profile import Profile, Recovery
from framework.types import SnapshotLabel


def _p(n: int = 50) -> Profile:
    return Profile(id="p0", x=np.linspace(0, 100, n), zb=np.zeros(n), d50=0.3)


class TestRecoveryEvent:
    def test_fraction_zero_no_change(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50)
        Recovery(t=1.0, fraction=0.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, 0.0, atol=1e-14)

    def test_fraction_one_full_restore(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50) * 2.0
        p.zb = zb_post.copy()
        Recovery(t=21.0, fraction=1.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, 2.0, atol=1e-14)

    def test_fraction_half(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50) * 2.0
        p.zb = zb_post.copy()
        Recovery(t=10.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-14)

    def test_takes_rec_snapshot(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50)
        Recovery(t=5.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        assert p.snapshots[-1].label == SnapshotLabel.REC

    def test_snapshot_t_stored(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50)
        Recovery(t=7.5, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        assert p.snapshots[-1].t == pytest.approx(7.5)

    def test_snapshot_is_copy(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50)
        Recovery(t=5.0, fraction=0.5, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        stored = p.snapshots[-1].zb.copy()
        p.zb[:] = 99.0
        np.testing.assert_array_equal(p.snapshots[-1].zb, stored)

    def test_z_berm_mask_below_berm(self):
        n = 10
        zb_post = np.zeros(n)
        zb_pre  = np.ones(n) * 2.0
        p = _p(n)
        p.zb = zb_post.copy()
        Recovery(t=5.0, fraction=1.0, zb_post_storm=zb_post,
                 zb_pre_storm=zb_pre, z_berm=1.0).apply(p)
        np.testing.assert_allclose(p.zb, zb_pre, atol=1e-14)

    def test_z_berm_mask_above_berm_unchanged(self):
        n = 10
        zb_post = np.ones(n) * 1.5
        p = _p(n)
        p.zb = zb_post.copy()
        Recovery(t=5.0, fraction=1.0, zb_post_storm=zb_post,
                 zb_pre_storm=np.ones(n) * 3.0, z_berm=1.0).apply(p)
        np.testing.assert_allclose(p.zb, zb_post, atol=1e-14)

    def test_fraction_one_exact_restore(self):
        p = _p()
        zb_post = np.zeros(50)
        zb_pre  = np.ones(50) * 2.5
        p.zb = zb_post.copy()
        Recovery(t=21.0, fraction=1.0, zb_post_storm=zb_post, zb_pre_storm=zb_pre).apply(p)
        np.testing.assert_allclose(p.zb, zb_pre, atol=1e-14)
