"""Tests for storm-interrupt scenarios — handled by run_campaign in the new architecture."""
import numpy as np

from erosion.config import ReachConfig
from erosion.nourishment import ActiveCampaign, NourishmentConfig, run_campaign
from erosion.results import NullResultsSink
from erosion.types import SnapshotLabel
from tests.builders import ncfg as _ncfg, profile as _p


class TestStormInterruptsCampaign:
    """When t_next arrives before campaign completes, run_campaign applies PartialNourishment."""

    def test_interrupted_campaign_returns_active_campaign(self):
        p = _p()
        cfg = ReachConfig(nourishment=_ncfg(production_rate=0.0001))
        cfg.storm.T_recover = 21.0
        _, campaign = run_campaign(
            [p], [np.zeros(50)], t_storm=0.0, t_next=0.001,
            cfg=cfg, sink=NullResultsSink(), longshore_widths=[50.0],
        )
        assert campaign is not None

    def test_interrupted_campaign_crew_on_site(self):
        p = _p()
        cfg = ReachConfig(nourishment=_ncfg(production_rate=0.0001))
        cfg.storm.T_recover = 21.0
        _, campaign = run_campaign(
            [p], [np.zeros(50)], t_storm=0.0, t_next=0.001,
            cfg=cfg, sink=NullResultsSink(), longshore_widths=[50.0],
        )
        assert campaign is not None
        assert campaign.crew_on_site is True

    def test_partial_nourishment_ssn_snapshot(self):
        p = _p()
        cfg = ReachConfig(nourishment=_ncfg(production_rate=0.0001))
        cfg.storm.T_recover = 21.0
        run_campaign(
            [p], [np.zeros(50)], t_storm=0.0, t_next=0.001,
            cfg=cfg, sink=NullResultsSink(), longshore_widths=[50.0],
        )
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.SSN in labels

    def test_crew_on_site_bypasses_volume_trigger(self):
        """ActiveCampaign with crew_on_site=True resumes without volume check."""
        p = _p()
        cfg = ReachConfig(nourishment=NourishmentConfig(
            template_x=list(np.linspace(0, 100, 50)),
            template_z=list(np.linspace(-0.5, 3.0, 50)),
            volume_trigger={"value": 1e9, "units": "m3"},
            production_rate={"value": 500.0, "units": "m3/day"},
        ))
        cfg.storm.T_recover = 21.0
        prior = ActiveCampaign(crew_on_site=True, priority_order=["p0"])
        run_campaign(
            [p], [np.zeros(50)], t_storm=0.0, t_next=200.0,
            cfg=cfg, sink=NullResultsSink(), longshore_widths=[50.0], prior=prior,
        )
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.ESN in labels

    def test_multi_profile_remaining_order_preserved(self):
        """Prior campaign order is respected when resuming."""
        p0, p1 = _p("p0"), _p("p1")
        cfg = ReachConfig(nourishment=_ncfg(production_rate=500.0))
        cfg.storm.T_recover = 21.0
        prior = ActiveCampaign(crew_on_site=True, priority_order=["p1", "p0"])
        run_campaign(
            [p0, p1], [np.zeros(50), np.zeros(50)], t_storm=0.0, t_next=200.0,
            cfg=cfg, sink=NullResultsSink(), longshore_widths=[50.0, 50.0], prior=prior,
        )
        assert SnapshotLabel.ESN in [s.label for s in p0.snapshots]
        assert SnapshotLabel.ESN in [s.label for s in p1.snapshots]
