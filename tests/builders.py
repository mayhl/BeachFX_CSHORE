"""Shared test builders — profiles, storms, forcing, configs, and a lifecycle
runner.

Replaces the per-file ``_p`` / ``_storms`` / ``_run`` / ``_ncfg`` helpers that
were duplicated across the erosion test modules.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from erosion.config import ReachConfig
from erosion.nourishment import NourishmentConfig
from erosion.profile import Profile
from erosion.reach import ReachContext, run_lifecycle
from erosion.results import NullResultsSink
from erosion.runner.mock import MockCSHORERunner

SIM_START = datetime(2030, 1, 1)


def profile(pid: str = "p0", n: int = 50, x_max: float = 100.0, zb=None) -> Profile:
    """Build a Profile on a uniform grid.

    ``zb``: ``None`` → flat zeros; ``"ramp"`` → ``linspace(-2, 3)``; otherwise an
    explicit array-like.
    """
    x = np.linspace(0.0, x_max, n)
    if zb is None:
        z = np.zeros(n)
    elif isinstance(zb, str) and zb == "ramp":
        z = np.linspace(-2.0, 3.0, n)
    else:
        z = np.asarray(zb, dtype=float)
    return Profile(id=pid, x=x, zb=z, d50=0.3)


def storms(
    n: int = 3,
    lifecycle: int = 0,
    *,
    wave_height: float = 1.2,
    water_elevation: float = 0.3,
    peak_period: float = 10.0,
    sim_start: datetime = SIM_START,
) -> pd.DataFrame:
    """N synthetic storms at t = 20, 40, ... days, 3 hydrograph rows each."""
    rows = []
    t0 = pd.Timestamp(sim_start)
    for i in range(n):
        day = (i + 1) * 20
        for dt_h in range(0, 13, 6):
            rows.append(dict(
                lifecycle=lifecycle, storm_id=f"S{i:02d}", hydro_tstp=dt_h,
                date=t0 + pd.Timedelta(days=day, hours=dt_h),
                wave_height=wave_height, wave_peak_period=peak_period,
                water_elevation=water_elevation, wave_direction=0.0,
            ))
    return pd.DataFrame(rows)


def forcing(hs=(1.0, 2.0)) -> dict:
    """Single-storm CSHORE BC dict for run_parallel_cshore tests."""
    hs = np.asarray(hs, dtype=float)
    n = len(hs)
    return {
        "timebc_wave": np.linspace(0.0, 3600.0 * (n - 1), n),
        "Hs": hs, "Hrms": hs / np.sqrt(2.0),
        "Tp": np.full(n, 9.0), "Wsetup": np.zeros(n),
        "swlbc": np.linspace(0.0, 0.3, n), "angle": np.zeros(n),
    }


def ncfg(volume_trigger: float = 0.001, production_rate: float = 500.0,
         n: int = 50) -> NourishmentConfig:
    """A nourishment config with a simple linear template."""
    return NourishmentConfig(
        template_x=list(np.linspace(0, 100, n)),
        template_z=list(np.linspace(-0.5, 3.0, n)),
        volume_trigger={"value": volume_trigger, "units": "m3"},
        production_rate={"value": production_rate, "units": "m3/day"},
    )


def run(profiles, n_storms: int = 3, *, cfg: ReachConfig | None = None,
        sink=None, lifecycle: int = 0, sim_start: datetime = SIM_START,
        reach_id: str = "test", alternative_id: str = "FWOP"):
    """Run one lifecycle with the mock runner; return ``(profiles, sink)``."""
    cfg = cfg or ReachConfig()
    sink = sink if sink is not None else NullResultsSink()
    ctx = ReachContext(reach_id=reach_id, alternative_id=alternative_id,
                       results=sink, sim_start=sim_start, cfg=cfg)
    run_lifecycle(profiles, storms(n_storms, lifecycle=lifecycle), sim_start,
                  (n_storms + 1) * 20 + 10.0, cfg, MockCSHORERunner(), ctx,
                  lifecycle=lifecycle)
    return profiles, sink
