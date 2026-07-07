"""Unit tests for campaign runner: _next_available and run_campaign."""

import tempfile

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.nourishment import ActiveCampaign, _next_available, run_campaign
from erosion.profile import Profile
from erosion.results import NullResultsSink, ParquetResultsSink
from erosion.types import SnapshotLabel
from tests.builders import ncfg as _ncfg
from tests.builders import profile as _p


def _cfg(nourishment=None) -> ReachConfig:
    return ReachConfig(nourishment=nourishment)


def _zb_pre(profiles: list[Profile]) -> list[np.ndarray]:
    return [p.zb.copy() + 0.5 for p in profiles]


class TestNextAvailable:
    def test_no_blackout_returns_t(self):
        assert _next_available(10.0, 5.0, []) == pytest.approx(10.0)

    def test_inside_blackout_defers_to_end(self):
        # [10, 15) overlaps [12, 20) → defer to 20
        assert _next_available(10.0, 5.0, [(12.0, 20.0)]) == pytest.approx(20.0)

    def test_entirely_after_blackout_no_defer(self):
        assert _next_available(25.0, 5.0, [(12.0, 20.0)]) == pytest.approx(25.0)

    def test_consecutive_blackouts_skips_both(self):
        # [0,6) hits [5,15) → defer to 15; [15,21) hits [15,25) → defer to 25
        assert _next_available(0.0, 6.0, [(5.0, 15.0), (15.0, 25.0)]) == pytest.approx(25.0)

    def test_start_exactly_at_blackout_end_ok(self):
        # t=20, duration=5: [20,25) — blackout ends at 20 → no overlap
        assert _next_available(20.0, 5.0, [(12.0, 20.0)]) == pytest.approx(20.0)


class TestRunCampaignNoNourishment:
    def test_returns_t_next(self):
        p = _p()
        t, campaign = run_campaign(
            [p], _zb_pre([p]), t_storm=10.0, t_next=30.0, cfg=_cfg(), sink=NullResultsSink()
        )
        assert t == pytest.approx(30.0)
        assert campaign is None

    def test_applies_linear_recovery(self):
        """T_recover=20, dt=20 → fraction=1 → zb recovers to zb_pre."""
        p = _p()
        zb_pre = [np.ones(50) * 1.0]
        cfg = _cfg()
        cfg.storm.T_recover = 20.0
        run_campaign([p], zb_pre, t_storm=10.0, t_next=30.0, cfg=cfg, sink=NullResultsSink())
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-12)


class TestRunCampaignBelowTrigger:
    def test_below_trigger_skips_nourishment(self):
        p = _p()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9))
        t, campaign = run_campaign(
            [p], _zb_pre([p]), t_storm=10.0, t_next=30.0, cfg=cfg, sink=NullResultsSink()
        )
        assert campaign is None

    def test_below_trigger_still_applies_recovery(self):
        p = _p()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9))
        cfg.storm.T_recover = 20.0
        zb_pre = [np.ones(50)]
        run_campaign([p], zb_pre, t_storm=0.0, t_next=20.0, cfg=cfg, sink=NullResultsSink())
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-12)


class TestRunCampaignWithNourishment:
    def test_record_nourishment_called(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            p = _p()
            cfg = _cfg(nourishment=_ncfg(volume_trigger=0.001, production_rate=500.0))
            cfg.storm.T_recover = 21.0
            run_campaign(
                [p],
                [np.zeros(50)],
                t_storm=0.0,
                t_next=200.0,
                cfg=cfg,
                sink=sink,
                longshore_widths=[50.0],
            )
            assert len(sink._nourishment_rows) >= 1
            assert sink._nourishment_rows[0]["event_type"] == "FullNourishment"


class TestRunCampaignStormInterrupt:
    def test_active_campaign_returned_when_interrupted(self):
        p0, p1 = _p("p0"), _p("p1")
        ncfg = _ncfg(volume_trigger=0.001, production_rate=0.0001)
        cfg = _cfg(nourishment=ncfg)
        cfg.storm.T_recover = 21.0
        t, campaign = run_campaign(
            [p0, p1],
            [np.zeros(50), np.zeros(50)],
            t_storm=0.0,
            t_next=0.001,
            cfg=cfg,
            sink=NullResultsSink(),
            longshore_widths=[50.0, 50.0],
        )
        assert campaign is not None
        assert campaign.crew_on_site is True  # folded from the deleted test_interrupts.py


class TestRunCampaignCrewOnSite:
    def test_crew_on_site_bypasses_volume_trigger(self):
        """crew_on_site=True bypasses the volume_trigger gate."""
        p = _p()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9, production_rate=500.0))
        cfg.storm.T_recover = 21.0
        prior = ActiveCampaign(crew_on_site=True, priority_order=["p0"])
        run_campaign(
            [p],
            [np.zeros(50)],
            t_storm=0.0,
            t_next=200.0,
            cfg=cfg,
            sink=NullResultsSink(),
            longshore_widths=[50.0],
            prior=prior,
        )
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.ESN in labels
