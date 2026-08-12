"""Unit tests for the campaign runner and the Tier-1 assessors."""

import tempfile

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.decision import ActiveCampaign
from erosion.metrics import fit_profile
from erosion.nourishment import (
    FittedAssessor,
    VolumeAssessor,
    _depth_of_closure,
    run_campaign,
)
from erosion.profile import Profile, ProfileGeometryConfig
from erosion.results import NullResultsSink, ParquetResultsSink
from erosion.runner.base import CSHOREResult
from erosion.storm import StormOutcome
from erosion.types import SnapshotLabel
from tests.builders import ncfg as _ncfg
from tests.builders import profile as _p
from tests.builders import template_profile
from tests.synthetic import DuneSpec, make_profile


def _cfg(nourishment=None) -> ReachConfig:
    return ReachConfig(nourishment=nourishment)


def _eroded(pid: str = "p0", scoop: float = 1.0) -> Profile:
    """A ``template_profile`` scooped uniformly by ``scoop`` m — a clear
    super-trigger subaerial deficit against its own as-built restore target
    (the parametric template synthesized from ``ref_metrics``)."""
    p = template_profile(pid)
    p.zb = p.zb - scoop
    return p


def _zb_pre(profiles: list[Profile]) -> list[np.ndarray]:
    return [p.zb.copy() + 0.5 for p in profiles]


# A placeholder non-None result marking a profile as NOT inundated; run_campaign
# reads only o.profile / o.zb_pre / o.inundated and never dereferences the result.
_OK_RESULT = CSHOREResult(
    zb=np.zeros(1), x=np.zeros(1), eta=np.zeros(1), Hs=np.zeros(1), runup_m=0.0
)


def _outcomes(profiles: list[Profile], zb_pre=None) -> list[StormOutcome]:
    """Wrap profiles + pre-storm beds as (non-inundated) StormOutcomes for run_campaign."""
    zb = zb_pre if zb_pre is not None else _zb_pre(profiles)
    return [StormOutcome(p, _OK_RESULT, zbp) for p, zbp in zip(profiles, zb)]


class TestRunCampaignNoNourishment:
    def test_returns_t_next(self):
        p = _p()
        t, campaign = run_campaign(
            _outcomes([p]), t_storm=10.0, t_next=30.0, cfg=_cfg(), sink=NullResultsSink()
        )
        assert t == pytest.approx(30.0)
        assert campaign is None

    def test_applies_linear_recovery(self):
        """T_recover=20, dt=20 → fraction=1 → zb recovers to zb_pre."""
        p = _p()
        zb_pre = [np.ones(50) * 1.0]
        cfg = _cfg()
        cfg.storm.T_recover = 20.0
        run_campaign(
            _outcomes([p], zb_pre), t_storm=10.0, t_next=30.0, cfg=cfg, sink=NullResultsSink()
        )
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-12)


class TestRunCampaignBelowTrigger:
    def test_below_trigger_skips_nourishment(self):
        p = _p()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9))
        t, campaign = run_campaign(
            _outcomes([p]), t_storm=10.0, t_next=30.0, cfg=cfg, sink=NullResultsSink()
        )
        assert campaign is None

    def test_below_trigger_still_applies_recovery(self):
        p = _p()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9))
        cfg.storm.T_recover = 20.0
        zb_pre = [np.ones(50)]
        run_campaign(
            _outcomes([p], zb_pre), t_storm=0.0, t_next=20.0, cfg=cfg, sink=NullResultsSink()
        )
        np.testing.assert_allclose(p.zb, 1.0, atol=1e-12)


class TestRunCampaignWithNourishment:
    def test_record_nourishment_called(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            p = _eroded()
            cfg = _cfg(
                nourishment=_ncfg(volume_trigger=0.001, production_rate=500.0, assessor="volume")
            )
            cfg.storm.T_recover = 21.0
            run_campaign(
                _outcomes([p], [p.zb.copy()]),
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
        p0, p1 = _eroded("p0"), _eroded("p1")
        ncfg = _ncfg(volume_trigger=0.001, production_rate=0.0001, assessor="volume")
        cfg = _cfg(nourishment=ncfg)
        cfg.storm.T_recover = 21.0
        t, campaign = run_campaign(
            _outcomes([p0, p1], [p0.zb.copy(), p1.zb.copy()]),
            t_storm=0.0,
            t_next=0.001,
            cfg=cfg,
            sink=NullResultsSink(),
            longshore_widths=[50.0, 50.0],
        )
        assert campaign is not None
        assert campaign.crew_on_site is True  # folded from the deleted test_interrupts.py


class TestFittedAssessor:
    """Geometric (fitted) deficit: measured berm-width shortfall vs the INIT berm."""

    def _profile(self, current_berm_width: float) -> Profile:
        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        _, zc, _ = make_profile(berm_elevation=2.0, berm_width=current_berm_width, dune=DuneSpec())
        m_init, _ = fit_profile(x, z0, 2.0, 0.0)  # as-built fit → ref_metrics (the target)
        return Profile("p0", x, zc.copy(), 0.3, ref_metrics=m_init)

    def test_no_erosion_no_fill(self):
        a = FittedAssessor().assess(self._profile(30.0), ReachConfig(), 50.0)
        assert a.needs_fill is False
        assert a.volume_m3 == pytest.approx(0.0)

    def test_eroded_berm_needs_fill(self):
        a = FittedAssessor().assess(self._profile(10.0), ReachConfig(), 50.0)
        assert a.needs_fill is True
        assert a.metrics["berm_width_deficit"] == pytest.approx(20.0, abs=1.0)
        # subaerial dry-wedge deficit = shortfall × BE × width
        assert a.volume_m3 == pytest.approx(a.metrics["berm_width_deficit"] * 2.0 * 50.0)

    def test_no_ref_metrics_no_fill(self):
        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        p = Profile("p0", x, z0, 0.3)  # no ref_metrics → not a FittedAssessor candidate
        assert FittedAssessor().assess(p, ReachConfig(), 50.0).needs_fill is False


class TestDepthOfClosure:
    """Reach default vs per-profile override resolution."""

    def test_reach_default_when_no_override(self):
        p = _p()
        assert _depth_of_closure(p, ReachConfig(depth_of_closure=6.0)) == pytest.approx(6.0)

    def test_profile_override_wins(self):
        g = ProfileGeometryConfig.model_validate(
            {"berm_elevation": 2.0, "depth_of_closure": 10.0}, context={"input_units": "m"}
        )
        p = _p()
        p.geometry = g
        assert _depth_of_closure(p, ReachConfig(depth_of_closure=6.0)) == pytest.approx(10.0)

    def test_none_override_falls_back_to_reach(self):
        g = ProfileGeometryConfig.model_validate(
            {"berm_elevation": 2.0}, context={"input_units": "m"}
        )
        p = _p()
        p.geometry = g
        assert _depth_of_closure(p, ReachConfig(depth_of_closure=6.0)) == pytest.approx(6.0)


class TestVolumeAssessorPlacement:
    """Placement volume extends the subaerial deficit down to depth of closure."""

    def _eroded(self, ncfg, ref_metrics=None) -> tuple[Profile, np.ndarray]:
        # profile sitting 0.5 m below the ncfg template → a positive deficit above MSL=0
        p = _p()
        p.zb = np.full(50, -0.5)
        p.ref_metrics = ref_metrics
        return p

    def test_placement_extends_to_doc_with_ref_be(self):
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        ref, _ = fit_profile(x, z0, 2.0, 0.0)
        be = ref.berm_elevation
        cfg = _cfg(nourishment=_ncfg(volume_trigger=0.001))
        cfg.depth_of_closure = 6.0
        p = self._eroded(cfg.nourishment, ref_metrics=ref)
        a = VolumeAssessor().assess(p, cfg, width_m=1.0)
        assert a.volume_m3 > 0.0
        assert a.placement_m3 == pytest.approx(a.volume_m3 * (be + 6.0) / be)

    def test_no_ref_metrics_no_inflation(self):
        cfg = _cfg(nourishment=_ncfg(volume_trigger=0.001))
        cfg.depth_of_closure = 6.0
        p = self._eroded(cfg.nourishment, ref_metrics=None)
        a = VolumeAssessor().assess(p, cfg, width_m=1.0)
        assert a.placement_m3 == pytest.approx(a.volume_m3)  # can't inflate without BE

    def test_zero_doc_no_inflation(self):
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        ref, _ = fit_profile(x, z0, 2.0, 0.0)
        cfg = _cfg(nourishment=_ncfg(volume_trigger=0.001))  # depth_of_closure defaults to 0.0
        p = self._eroded(cfg.nourishment, ref_metrics=ref)
        a = VolumeAssessor().assess(p, cfg, width_m=1.0)
        assert a.placement_m3 == pytest.approx(a.volume_m3)


class TestFittedAssessorPlacement:
    def _profile(self, current_berm_width: float) -> Profile:
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        _, zc, _ = make_profile(berm_elevation=2.0, berm_width=current_berm_width, dune=DuneSpec())
        m_init, _ = fit_profile(x, z0, 2.0, 0.0)
        return Profile("p0", x, zc.copy(), 0.3, ref_metrics=m_init)

    def test_placement_is_full_active_wedge(self):
        cfg = ReachConfig(depth_of_closure=6.0)
        a = FittedAssessor().assess(self._profile(10.0), cfg, 50.0)
        dd = a.metrics["berm_width_deficit"]
        assert a.metrics["depth_of_closure"] == pytest.approx(6.0)
        assert a.placement_m3 == pytest.approx(dd * (2.0 + 6.0) * 50.0)


class TestGeometricAssessor:
    """Fitted decision + a restore template SYNTHESIZED from target geometry."""

    def _realistic(self, target_bw=None):
        """As-built (30 m berm, dune crest fixed at x=65) vs a realistically eroded
        current (shoreline retreated, berm ~7 m, dune fixed)."""
        from erosion.metrics import fit_profile
        from erosion.nourishment import GeometricAssessor, NourishmentConfig

        x = np.arange(0, 121, 1.0)
        render = lambda ks: np.interp(x, *zip(*ks))  # noqa: E731
        asbuilt = render([(0, -1), (20, 0), (25, 2), (55, 2), (65, 5), (78, 3), (120, 3)])
        current = render([(0, -1), (42, 0), (47, 2), (55, 2), (65, 5), (78, 3), (120, 3)])
        ref, _ = fit_profile(x, asbuilt, 2.0, 0.0)
        p = Profile("p0", x, current.copy(), 0.3, ref_metrics=ref)
        tg = {} if target_bw is None else {"berm_width": {"value": target_bw, "units": "m"}}
        ncfg = NourishmentConfig.model_validate(
            {
                "volume_trigger": {"value": 0.001, "units": "m3"},
                "production_rate": {"value": 500.0, "units": "m3/day"},
                "template_geometry": tg,
            },
            context={"input_units": "m"},
        )
        cfg = _cfg(nourishment=ncfg)
        cfg.depth_of_closure = 6.0
        return GeometricAssessor(), p, cfg, x, asbuilt, current

    def test_basis_and_config_target(self):
        ga, p, cfg, *_ = self._realistic(target_bw=30.0)
        a = ga.assess(p, cfg, 50.0)
        assert a.metrics["basis"] == "geometric"
        assert a.metrics["target_berm_width"] == pytest.approx(30.0)
        assert a.needs_fill and a.metrics["berm_width_deficit"] > 15.0

    def test_target_falls_back_to_ref_when_unset(self):
        ga, p, cfg, *_ = self._realistic(target_bw=None)  # no config → as-built ref berm
        a = ga.assess(p, cfg, 50.0)
        assert a.metrics["target_berm_width"] == pytest.approx(p.ref_metrics.berm_width)

    def test_synthesized_template_is_fill_only_and_preserves_upland(self):
        ga, p, cfg, x, _asbuilt, current = self._realistic(target_bw=30.0)
        tmpl = ga.restore_template(p, cfg)
        assert np.all(tmpl >= current - 1e-9)  # never carves
        assert np.allclose(tmpl[x >= 90], current[x >= 90])  # upland behind the dune preserved

    def test_synthesized_template_recovers_asbuilt_berm(self):
        ga, p, cfg, x, asbuilt, _current = self._realistic(target_bw=30.0)
        tmpl = ga.restore_template(p, cfg)
        berm = x <= 60
        assert np.sqrt(np.mean((tmpl[berm] - asbuilt[berm]) ** 2)) < 0.25

    def _eroded_dune(self, dune_height=3.0, dune_width=23.0):
        """As-built dune (crest 5 at x=65, toes at x=55/78) vs a current whose dune
        front has slumped to a crest of 4.0 (front relief 2.0) — berm intact."""
        from erosion.metrics import fit_profile
        from erosion.nourishment import GeometricAssessor, NourishmentConfig

        x = np.arange(0, 121, 1.0)
        render = lambda ks: np.interp(x, *zip(*ks))  # noqa: E731
        asbuilt = render([(0, -1), (20, 0), (25, 2), (55, 2), (65, 5), (78, 3), (120, 3)])
        current = render([(0, -1), (20, 0), (25, 2), (55, 2), (65, 4.0), (78, 3), (120, 3)])
        ref, _ = fit_profile(x, asbuilt, 2.0, 0.0)
        p = Profile("p0", x, current.copy(), 0.3, ref_metrics=ref)
        ncfg = NourishmentConfig.model_validate(
            {
                "volume_trigger": {"value": 0.001, "units": "m3"},
                "production_rate": {"value": 500.0, "units": "m3/day"},
                "template_geometry": {
                    "berm_width": {"value": 30.0, "units": "m"},
                    "dune_height": {"value": dune_height, "units": "m"},
                    "dune_width": {"value": dune_width, "units": "m"},
                },
            },
            context={"input_units": "m"},
        )
        cfg = _cfg(nourishment=ncfg)
        cfg.depth_of_closure = 6.0
        return GeometricAssessor(), p, cfg, x, current

    def test_synthesized_template_rebuilds_eroded_dune(self):
        ga, p, cfg, x, current = self._eroded_dune()  # target crest = BE + 3 = 5
        tmpl = ga.restore_template(p, cfg)
        assert np.all(tmpl >= current - 1e-9)  # fill-only: never carves
        dune = (x >= 55) & (x <= 78)
        assert tmpl[dune].max() > current[dune].max() + 0.8  # crest raised back up
        assert tmpl[dune].max() == pytest.approx(5.0, abs=0.25)  # toward BE + target relief
        assert np.allclose(tmpl[x >= 90], current[x >= 90])  # upland behind the dune preserved

    def test_dune_fill_metered_into_placement(self):
        ga, p, cfg, x, current = self._eroded_dune()  # berm intact, dune slumped 4.0 -> target 5.0
        from erosion.metrics import fit_profile

        a = ga.assess(p, cfg, 50.0)
        assert a.metrics["dune_fill_m3"] > 0.0  # slumped dune -> nonzero fill
        # placement = berm wedge (BE+DClose) + the metered subaerial dune wedge
        be = p.ref_metrics.berm_elevation
        berm_wedge = a.metrics["berm_width_deficit"] * (be + a.metrics["depth_of_closure"]) * 50.0
        assert a.placement_m3 == pytest.approx(berm_wedge + a.metrics["dune_fill_m3"])
        # the metered volume matches integrating the synthesized template's dune region
        m, _ = fit_profile(x, current, be, cfg.msl, ref=p.ref_metrics)
        assert a.metrics["dune_fill_m3"] == pytest.approx(
            ga._extra_placement(p, cfg, p.ref_metrics, m, 50.0)
        )

    def test_base_fitted_assessor_meters_no_dune_fill(self):
        # The base FittedAssessor doesn't synthesize a dune, so its placement carries
        # no dune-fill term (the hook defaults to 0) — only GeometricAssessor adds it.
        from erosion.nourishment import FittedAssessor

        _ga, p, cfg, *_ = self._eroded_dune()
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert "dune_fill_m3" not in a.metrics

    def test_no_ref_metrics_places_no_fill(self):
        """Without an as-built fit basis (``ref_metrics``) there is no parametric
        target to synthesize, so the restore template is the bed itself — no fill."""
        from erosion.nourishment import GeometricAssessor

        x = np.arange(0, 121, 1.0)
        current = np.interp(x, [0, 42, 47, 55, 120], [-1, 0, 2, 2, 3])
        p = Profile("p0", x, current, 0.3)  # no ref_metrics
        cfg = _cfg(nourishment=_ncfg(volume_trigger=0.001))
        tmpl = GeometricAssessor().restore_template(p, cfg)
        np.testing.assert_allclose(tmpl, p.zb)  # no basis -> no fill


class TestDuneFormSynthesis:
    """The three synthesized dune-crest forms (``template_geometry.dune_form``):
    a sharp ``triangle`` apex, a flat-topped ``trapezoid``, a rounded ``gaussian``.
    Each rebuilds the eroded dune to the target crest, fill-only and subaerial
    (datum up), leaving the subaqueous profile CSHORE runs on untouched."""

    def _template(self, dune_form=None):
        from erosion.metrics import fit_profile
        from erosion.nourishment import GeometricAssessor, NourishmentConfig

        x = np.arange(0, 121, 1.0)
        render = lambda ks: np.interp(x, *zip(*ks))  # noqa: E731
        asbuilt = render([(0, -1), (20, 0), (25, 2), (55, 2), (65, 5), (78, 3), (120, 3)])
        current = render([(0, -1), (20, 0), (25, 2), (55, 2), (65, 4.0), (78, 3), (120, 3)])
        ref, _ = fit_profile(x, asbuilt, 2.0, 0.0)
        p = Profile("p0", x, current.copy(), 0.3, ref_metrics=ref)
        tg = {
            "berm_width": {"value": 30.0, "units": "m"},
            "dune_height": {"value": 3.0, "units": "m"},
            "dune_width": {"value": 23.0, "units": "m"},
        }
        if dune_form is not None:
            tg["dune_form"] = dune_form
        ncfg = NourishmentConfig.model_validate(
            {
                "volume_trigger": {"value": 0.001, "units": "m3"},
                "production_rate": {"value": 500.0, "units": "m3/day"},
                "template_geometry": tg,
            },
            context={"input_units": "m"},
        )
        cfg = _cfg(nourishment=ncfg)
        cfg.depth_of_closure = 6.0
        return x, current, GeometricAssessor().restore_template(p, cfg)

    def _plateau_pts(self, x, tmpl) -> int:
        dune = (x >= 55) & (x <= 80)
        return int(np.count_nonzero(np.abs(tmpl[dune] - tmpl[dune].max()) < 0.05))

    @pytest.mark.parametrize("form", ["triangle", "trapezoid", "gaussian"])
    def test_each_form_fill_only_subaerial_and_reaches_crest(self, form):
        x, current, tmpl = self._template(form)
        assert np.all(tmpl >= current - 1e-9)  # fill-only: never carves
        assert np.all(tmpl[tmpl > current + 1e-9] >= 0.0)  # subaerial: no fill below datum
        assert np.allclose(tmpl[current < 0.0], current[current < 0.0])  # subaqueous untouched
        dune = (x >= 55) & (x <= 80)
        assert tmpl[dune].max() == pytest.approx(5.0, abs=0.25)  # BE + target relief

    def test_trapezoid_has_flat_top_triangle_does_not(self):
        x, _c, tri = self._template("triangle")
        _x, _c2, trap = self._template("trapezoid")
        assert self._plateau_pts(x, tri) == 1  # sharp apex, single crest node
        assert self._plateau_pts(x, trap) >= 4  # measurable plateau

    def test_gaussian_crest_is_rounded_not_linear(self):
        """A triangle's front face is linear (≈0 second difference); the gaussian's
        rounds over — its rising increments shrink toward the crest (concave)."""
        x, _c, tri = self._template("triangle")
        _x, _c2, gau = self._template("gaussian")
        face = (x >= 60) & (x <= 65)  # dune front, up to the crest
        assert np.max(np.abs(np.diff(tri[face], 2))) < 0.02  # triangle ~linear
        assert np.min(np.diff(gau[face], 2)) < -0.02  # gaussian concave near crest

    def test_auto_form_is_triangle_without_ref_plateau(self):
        """``dune_form=None`` auto-selects triangle when the idealized dune has no
        measured plateau (``ref.dune_top_width == 0``)."""
        _xa, _ca, auto = self._template(None)
        _xt, _ct, tri = self._template("triangle")
        np.testing.assert_allclose(auto, tri)


class TestAssessorSelection:
    """Per-profile override -> reach default -> auto-classify (dune -> geometric)."""

    def _dune_profile(self, pid="p0"):
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        ref, _ = fit_profile(x, z0, 2.0, 0.0)
        return Profile(pid, x, z0.copy(), 0.3, ref_metrics=ref)

    def test_auto_classify_dune_to_geometric(self):
        from erosion.nourishment import GeometricAssessor, _select_assessor

        p = self._dune_profile()
        assert isinstance(_select_assessor(p, _ncfg()), GeometricAssessor)

    def test_auto_classify_no_dune_to_volume(self):
        from erosion.nourishment import VolumeAssessor as VA
        from erosion.nourishment import _select_assessor

        p = _p()  # no ref_metrics -> no dune
        assert isinstance(_select_assessor(p, _ncfg()), VA)

    def test_reach_default_overrides_auto(self):
        from erosion.nourishment import FittedAssessor as FA
        from erosion.nourishment import _select_assessor

        p = _p()
        ncfg = _ncfg()
        ncfg.assessor = "fitted"
        assert isinstance(_select_assessor(p, ncfg), FA)

    def test_per_profile_override_wins(self):
        from erosion.nourishment import VolumeAssessor as VA
        from erosion.nourishment import _select_assessor

        p = self._dune_profile("p7")  # would auto-classify geometric
        ncfg = _ncfg()
        ncfg.assessor = "geometric"
        ncfg.per_profile_assessor = {"p7": "volume"}
        assert isinstance(_select_assessor(p, ncfg), VA)  # explicit per-profile wins

    def test_geometric_selection_warns(self):
        """Both auto-classified and explicit geometric selection emit the legacy
        deprecation warning."""
        from erosion.nourishment import _select_assessor

        with pytest.warns(DeprecationWarning, match="legacy assessor"):
            _select_assessor(self._dune_profile(), _ncfg())  # auto-classify -> geometric
        ncfg = _ncfg()
        ncfg.assessor = "geometric"
        with pytest.warns(DeprecationWarning, match="legacy assessor"):
            _select_assessor(_p(), ncfg)  # explicit reach default

    def test_non_geometric_selection_does_not_warn(self):
        import warnings

        from erosion.nourishment import _select_assessor

        ncfg = _ncfg()
        ncfg.assessor = "fitted"
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> test failure
            _select_assessor(_p(), ncfg)


class TestNourishmentConfigValidation:
    """Trigger-presence and restore-vs-trigger geometry validators."""

    @staticmethod
    def _build(**over):
        """A minimal NourishmentConfig payload (no triggers by default) plus overrides."""
        from erosion.nourishment import NourishmentConfig

        payload = {
            "production_rate": {"value": 500.0, "units": "m3/day"},
            **over,
        }
        return NourishmentConfig.model_validate(payload, context={"input_units": "m"})

    def test_no_active_trigger_rejected(self):
        with pytest.raises(ValueError, match="no active trigger"):
            self._build()  # no volume_trigger, no emergency_volume, no trigger_geometry

    def test_volume_trigger_alone_is_valid(self):
        cfg = self._build(volume_trigger={"value": 1.0, "units": "m3"})
        assert cfg.emergency_volume is None

    def test_emergency_volume_alone_is_valid(self):
        cfg = self._build(emergency_volume={"value": 100.0, "units": "m3"})
        assert cfg.volume_trigger is None and cfg.emergency_volume == pytest.approx(100.0)

    def test_trigger_geometry_alone_is_valid(self):
        cfg = self._build(trigger_geometry={"dune_height": {"value": 2.0, "units": "m"}})
        assert cfg.volume_trigger is None and cfg.trigger_geometry.dune_height == pytest.approx(2.0)

    def test_restore_below_trigger_rejected(self):
        with pytest.raises(ValueError, match="can't clear the emergency trigger"):
            self._build(
                trigger_geometry={"dune_height": {"value": 3.0, "units": "m"}},
                template_geometry={"dune_height": {"value": 2.0, "units": "m"}},
            )

    def test_restore_meets_trigger_is_valid(self):
        cfg = self._build(
            trigger_geometry={"dune_height": {"value": 2.0, "units": "m"}},
            template_geometry={"dune_height": {"value": 2.5, "units": "m"}},
        )
        assert cfg.template_geometry.dune_height == pytest.approx(2.5)

    def test_trigger_set_restore_unset_is_valid(self):
        # unset restore target falls back to as-built (unknowable at config time) -> not checked
        cfg = self._build(trigger_geometry={"dune_width": {"value": 10.0, "units": "m"}})
        assert cfg.template_geometry.dune_width is None


class TestEmergencyTrigger:
    """Geometric emergency trigger sets force; force override mobilizes the reach."""

    def _profile_with_dune(self):
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=DuneSpec())
        ref, _ = fit_profile(x, z0, 2.0, 0.0)
        p = Profile("p0", x, z0.copy(), 0.3, ref_metrics=ref)
        m, _ = fit_profile(x, z0, 2.0, 0.0, ref=ref)
        return p, m

    def _cfg_trigger(self, **tg):
        ncfg = _ncfg(volume_trigger=1e9)  # never trips the volume gate
        ncfg.trigger_geometry.__dict__.update(tg)
        cfg = _cfg(nourishment=ncfg)
        return cfg

    def test_force_when_dune_height_below_threshold(self):
        p, m = self._profile_with_dune()
        cfg = self._cfg_trigger(dune_height=m.dune_front_relief + 1.0)  # threshold above measured
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert a.force is True

    def test_no_force_when_dune_above_threshold(self):
        p, m = self._profile_with_dune()
        cfg = self._cfg_trigger(dune_height=max(0.1, m.dune_front_relief - 1.0))
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert a.force is False

    def test_berm_width_is_not_a_trigger(self):
        p, _m = self._profile_with_dune()
        cfg = self._cfg_trigger(berm_width=1e9)  # huge berm target, but berm is not a trigger
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert a.force is False

    def test_volume_assessor_never_forces(self):
        p, _m = self._profile_with_dune()
        cfg = self._cfg_trigger(dune_height=1e9)
        a = VolumeAssessor().assess(p, cfg, 50.0)
        assert a.force is False  # no dune trigger, and emergency_volume unset

    def test_force_when_ref_dune_is_storm_erased(self):
        """A dune shaved past the fitter's prominence floor measures as NaN crest
        (HIGH_UPLAND), which must read as below-threshold, not unmeasurable."""
        p, _m = self._profile_with_dune()
        p.zb = np.minimum(p.zb, p.ref_metrics.berm_elevation)  # plane the dune off
        cfg = self._cfg_trigger(dune_height=0.5)
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert a.metrics["dune_lost"] is True
        assert a.force is True

    def test_no_force_when_profile_never_had_a_dune(self):
        from erosion.metrics import fit_profile

        x, z0, _ = make_profile(berm_elevation=2.0, berm_width=30.0, dune=None)
        ref, _ = fit_profile(x, z0, 2.0, 0.0)
        p = Profile("p0", x, z0.copy(), 0.3, ref_metrics=ref)
        cfg = self._cfg_trigger(dune_height=0.5, dune_width=5.0)
        a = FittedAssessor().assess(p, cfg, 50.0)
        assert a.metrics["dune_lost"] is False
        assert a.force is False

    def test_base_volume_emergency_force(self):
        """The assessor-agnostic base trigger: a deficit at/above emergency_volume
        forces, below does not, and an unset threshold never fires."""
        from erosion.nourishment import ProfileAssessment

        a = ProfileAssessment(needs_fill=True, volume_m3=100.0)
        ncfg = _ncfg(volume_trigger=1e9)
        cfg = _cfg(nourishment=ncfg)
        assert VolumeAssessor().emergency_force(a, cfg) is False  # threshold unset
        ncfg.emergency_volume = 100.0
        assert VolumeAssessor().emergency_force(a, cfg) is True  # deficit meets threshold
        ncfg.emergency_volume = 150.0
        assert VolumeAssessor().emergency_force(a, cfg) is False  # deficit below threshold

    def test_fitted_falls_back_to_volume_emergency(self):
        """A dune-aware assessor with no geometric trigger still forces via the base
        volume threshold (shared fallback)."""
        from erosion.nourishment import ProfileAssessment

        a = ProfileAssessment(needs_fill=False, volume_m3=100.0, metrics={})  # no dune metrics
        ncfg = _ncfg(volume_trigger=1e9)  # no trigger_geometry set
        cfg = _cfg(nourishment=ncfg)
        assert FittedAssessor().emergency_force(a, cfg) is False
        ncfg.emergency_volume = 50.0
        assert FittedAssessor().emergency_force(a, cfg) is True  # volume fallback

    def test_forced_dune_only_profile_gets_placed(self):
        """A profile whose dune trips the emergency but whose berm is intact
        (needs_fill=False) still gets a plan and is placed (dune-only restore)."""
        from erosion.nourishment import _select_assessor

        p, m = self._profile_with_dune()
        ncfg = _ncfg(volume_trigger=1e9, production_rate=500.0)  # volume gate never trips
        ncfg.template_geometry.berm_width = 1.0  # target << measured -> no berm deficit
        ncfg.trigger_geometry.__dict__.update(dune_height=m.dune_front_relief + 1.0)  # force
        cfg = _cfg(nourishment=ncfg)
        cfg.storm.T_recover = 21.0
        a = _select_assessor(p, ncfg).assess(p, cfg, 50.0)
        assert a.needs_fill is False and a.force is True  # dune-only emergency
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWP", lifecycle=0)
            run_campaign(
                _outcomes([p], [p.zb.copy()]),
                t_storm=0.0,
                t_next=1e6,
                cfg=cfg,
                sink=sink,
                longshore_widths=[50.0],
            )
            assert any(r["event_type"] == "FullNourishment" for r in sink._nourishment_rows)


class TestBorrowDrivesDuration:
    """The borrow volume (placement × ratio) drives duration & the recorded cost basis."""

    def test_ratio_scales_borrow_and_duration(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            p = _eroded()
            ncfg = _ncfg(volume_trigger=0.001, production_rate=500.0, assessor="volume")
            ncfg.borrow_to_placement_ratio = 2.0
            cfg = _cfg(nourishment=ncfg)
            cfg.storm.T_recover = 21.0
            run_campaign(
                _outcomes([p], [p.zb.copy()]),
                t_storm=0.0,
                t_next=1e6,
                cfg=cfg,
                sink=sink,
                longshore_widths=[50.0],
            )
            row = sink._nourishment_rows[0]
            assert row["event_type"] == "FullNourishment"
            assert row["borrow_m3"] == pytest.approx(row["volume_m3"] * 2.0)
            # duration ran off the borrow volume, not the placement volume
            assert row["t_end"] - row["t_start"] == pytest.approx(row["borrow_m3"] / 500.0)


class TestRunCampaignCrewOnSite:
    def test_crew_on_site_bypasses_volume_trigger(self):
        """crew_on_site=True bypasses the volume_trigger gate."""
        p = _eroded()
        cfg = _cfg(nourishment=_ncfg(volume_trigger=1e9, production_rate=500.0, assessor="volume"))
        cfg.storm.T_recover = 21.0
        prior = ActiveCampaign(crew_on_site=True, priority_order=["p0"])
        run_campaign(
            _outcomes([p], [p.zb.copy()]),
            t_storm=0.0,
            t_next=200.0,
            cfg=cfg,
            sink=NullResultsSink(),
            longshore_widths=[50.0],
            prior=prior,
        )
        labels = [s.label for s in p.snapshots]
        assert SnapshotLabel.EEN in labels
