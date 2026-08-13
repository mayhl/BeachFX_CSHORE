"""Integration tests against the real CSHORE binary — a single storm and a
3-storm chain.  Run with: pytest -m integration"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from erosion.config import ReachConfig
from erosion.profile import Profile
from erosion.reach import Reach
from erosion.results import NullResultsSink
from erosion.runner.local import LocalCSHORERunner
from erosion.types import SnapshotLabel

# Repo root, for the real-profile CSV (package imports come from pytest.ini pythonpath)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="module")
def cshore_params():
    return ReachConfig().cshore


@pytest.fixture(scope="module")
def real_profile() -> Profile:
    from erosion.profile import load_raw_profile

    raw = load_raw_profile(os.path.join(ROOT, "data/profiles/reach1_p0.csv"), 0.3)
    return Profile(id="Reach1_p0", x=raw["x"], zb=raw["z"].copy(), d50=raw["d50"])


def _storm_forcing() -> dict:
    t = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])
    return {
        "timebc_wave": t,
        "Hs": np.array([0.3, 1.2, 2.0, 1.5, 0.8]),
        "Hrms": np.array([0.21, 0.85, 1.41, 1.06, 0.57]),
        "Tp": np.array([6.0, 8.0, 10.0, 9.0, 7.0]),
        "Wsetup": np.zeros(5),
        "swlbc": np.array([0.0, 0.15, 0.4, 0.3, 0.1]),
        "angle": np.zeros(5),
    }


@pytest.mark.integration
def test_single_storm_changes_profile(cshore_params, real_profile):
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        zb_before = real_profile.zb.copy()

        result = runner.run(real_profile, _storm_forcing())

        zb_original_on_output_grid = np.interp(result.x, real_profile.x, zb_before)
        max_change = np.max(np.abs(result.zb - zb_original_on_output_grid))
        assert max_change > 1e-4, (
            f"CSHORE produced no meaningful bed-level change (max|Δzb|={max_change:.2e})"
        )

        assert len(result.zb) == len(result.x)
        assert len(result.eta) > 0
        assert len(result.Hs) > 0


@pytest.mark.integration
def test_single_storm_result_has_no_nans(cshore_params, real_profile):
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        result = runner.run(real_profile, _storm_forcing())

        assert not np.any(np.isnan(result.zb)), "NaN in zb output"
        assert not np.any(np.isnan(result.x)), "NaN in x output"


def _storms_df(n_storms: int = 3) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp(datetime(2030, 1, 1))
    Hs_series = np.array([0.3, 1.2, 2.0, 1.5, 0.8])
    t_series = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])
    for i in range(n_storms):
        day = (i + 1) * 10
        for j, (dt_s, hs) in enumerate(zip(t_series, Hs_series)):
            rows.append(
                dict(
                    lifecycle=0,
                    storm_id=f"S{i:04d}",
                    hydro_tstp=j,
                    date=t0 + pd.Timedelta(days=day) + pd.Timedelta(seconds=float(dt_s)),
                    wave_height=hs,
                    wave_peak_period=10.0,
                    water_elevation=0.3,
                    wave_direction=0.0,
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.integration
def test_three_storm_chain_completes(cshore_params, real_profile):
    """Full 3-storm chain via run_lifecycle; profile is updated after each storm."""
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        p = Profile(
            id=real_profile.id,
            x=real_profile.x.copy(),
            zb=real_profile.zb.copy(),
            d50=real_profile.d50,
        )

        sim_start = datetime(2030, 1, 1)
        Reach(
            profiles=[p],
            cfg=ReachConfig(),
            results=NullResultsSink(),
            runner=runner,
            sim_start=sim_start,
            reach_id="Reach1",
        ).run(_storms_df(3), 35.0)

        labels = [s.label for s in p.snapshots]
        assert labels[0] == SnapshotLabel.INIT
        assert sum(1 for l in labels if l == SnapshotLabel.PreStorm) == 3
        assert sum(1 for l in labels if l == SnapshotLabel.PostStorm) == 3
        assert SnapshotLabel.EndIteration in labels

        init_zb = p.snapshots[0].zb
        last_post = next(s.zb for s in reversed(p.snapshots) if s.label == SnapshotLabel.PostStorm)
        zb_init_on_grid = np.interp(p.x, np.linspace(0, p.x[-1], len(init_zb)), init_zb)
        max_change = np.max(np.abs(last_post - zb_init_on_grid))
        assert max_change > 1e-4, (
            f"3-storm chain produced no meaningful bed change (max|Δzb|={max_change:.2e})"
        )


@pytest.mark.integration
def test_three_storm_chain_no_nans(cshore_params, real_profile):
    """No NaN values in bed level or x-grid after a 3-storm chain."""
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        p = Profile(
            id=real_profile.id,
            x=real_profile.x.copy(),
            zb=real_profile.zb.copy(),
            d50=real_profile.d50,
        )

        sim_start = datetime(2030, 1, 1)
        Reach(
            profiles=[p],
            cfg=ReachConfig(),
            results=NullResultsSink(),
            runner=runner,
            sim_start=sim_start,
            reach_id="Reach1",
        ).run(_storms_df(3), 35.0)

        assert not np.any(np.isnan(p.zb)), "NaN in final zb"
        assert not np.any(np.isnan(p.x)), "NaN in x grid"
