"""Integration test — requires real CSHORE binary. Run with: pytest -m integration"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from framework.config import ReachConfig
from framework.profile import Profile
from framework.runner.local import LocalCSHORERunner


@pytest.fixture(scope="module")
def config() -> dict:
    with open(os.path.join(ROOT, "config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_profile(config) -> Profile:
    from utils.geometry import load_raw_profile
    reach_cfg = config["profile"]["Reach1"]
    raw = load_raw_profile(os.path.join(ROOT, reach_cfg["file"]), reach_cfg["d50"])
    return Profile(id="Reach1_p0", x=raw["x"], zb=raw["z"], d50=raw["d50"])


def _storm_forcing() -> dict:
    t = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])
    return {
        "timebc_wave": t,
        "Hs":    np.array([0.3, 1.2, 2.0, 1.5, 0.8]),
        "Hrms":  np.array([0.21, 0.85, 1.41, 1.06, 0.57]),
        "Tp":    np.array([6.0, 8.0, 10.0, 9.0, 7.0]),
        "Wsetup": np.zeros(5),
        "swlbc": np.array([0.0, 0.15, 0.4, 0.3, 0.1]),
        "angle": np.zeros(5),
    }


@pytest.mark.integration
def test_single_storm_changes_profile(config, real_profile):
    cshore_params = ReachConfig.model_validate(
        {"cshore": {**config.get("cshore", {}), **config.get("model_logic", {})}},
    ).cshore
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        zb_before = real_profile.zb.copy()

        result = runner.run(real_profile, _storm_forcing())

        zb_original_on_output_grid = np.interp(result.x, real_profile.x, zb_before)
        max_change = np.max(np.abs(result.zb - zb_original_on_output_grid))
        assert max_change > 1e-4, \
            f"CSHORE produced no meaningful bed-level change (max|Δzb|={max_change:.2e})"

        assert len(result.zb) == len(result.x)
        assert len(result.eta) > 0
        assert len(result.Hs) > 0


@pytest.mark.integration
def test_single_storm_result_has_no_nans(config, real_profile):
    cshore_params = ReachConfig.model_validate(
        {"cshore": {**config.get("cshore", {}), **config.get("model_logic", {})}},
    ).cshore
    with tempfile.TemporaryDirectory() as work_dir:
        runner = LocalCSHORERunner(params=cshore_params, work_dir=work_dir)
        result = runner.run(real_profile, _storm_forcing())

        assert not np.any(np.isnan(result.zb)), "NaN in zb output"
        assert not np.any(np.isnan(result.x)), "NaN in x output"
