from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from framework.config import ReachConfig
from framework.profile import Profile
from framework.reach import ReachContext
from framework.results import NullResultsSink
from framework.runner.mock import MockCSHORERunner

SIM_START = datetime(2030, 1, 1)


@pytest.fixture
def beach_profile() -> Profile:
    n = 100
    x = np.linspace(0, 200, n)
    zb = np.where(x < 40, -0.5 + x / 100, np.where(x < 80, 1.5, 3.0)).astype(float)
    return Profile(id="p0", x=x, zb=zb.copy(), d50=0.3)


@pytest.fixture
def two_profiles(beach_profile: Profile) -> list[Profile]:
    p1 = Profile(id="p1", x=beach_profile.x.copy(),
                 zb=beach_profile.zb.copy() * 0.9, d50=0.3)
    return [beach_profile, p1]


@pytest.fixture
def one_storm() -> dict:
    t = np.array([0.0, 3600.0, 7200.0, 10800.0])
    return {
        "timebc_wave": t,
        "Hs":    np.array([0.5, 1.5, 2.0, 1.0]),
        "Hrms":  np.array([0.35, 1.06, 1.41, 0.71]),
        "Tp":    np.array([6.0, 8.0, 10.0, 8.0]),
        "Wsetup": np.zeros(4),
        "swlbc": np.array([0.0, 0.3, 0.5, 0.2]),
        "angle": np.zeros(4),
    }


@pytest.fixture
def storms_df() -> pd.DataFrame:
    """3 synthetic storms at t=10, 40, 80 days from SIM_START."""
    rows = []
    t0 = pd.Timestamp(SIM_START)
    for sid, day, Hs in [("s1", 10, 1.2), ("s2", 40, 1.5), ("s3", 80, 1.0)]:
        for dt_h in range(0, 25, 6):
            rows.append(dict(
                lifecycle=0, storm_id=sid, hydro_tstp=dt_h,
                date=t0 + pd.Timedelta(days=day, hours=dt_h),
                wave_height=Hs, wave_peak_period=10.0,
                water_elevation=0.5, wave_direction=0.0,
            ))
    return pd.DataFrame(rows)


@pytest.fixture
def mock_runner() -> MockCSHORERunner:
    return MockCSHORERunner()


@pytest.fixture
def minimal_ctx() -> ReachContext:
    return ReachContext(
        reach_id="test",
        alternative_id="FWOP",
        results=NullResultsSink(),
        sim_start=SIM_START,
    )
