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
from erosion.metrics import fit_profile
from erosion.nourishment import NourishmentConfig
from erosion.profile import Profile
from erosion.reach import Reach
from erosion.results import NullResultsSink
from tests.synthetic import DuneSpec, make_profile

SIM_START = datetime(2030, 1, 1)

_STORM_COLUMNS = [
    "lifecycle",
    "storm_id",
    "hydro_tstp",
    "date",
    "wave_height",
    "wave_peak_period",
    "water_elevation",
    "wave_direction",
]


class RecordingSink(NullResultsSink):
    """Test sink that captures reach/SIM-scope decisions in memory."""

    def __init__(self):
        self.decisions: list[tuple] = []  # (kind, t, profile_id, payload)

    def record_decision(self, kind, t, profile_id=None, **payload):
        self.decisions.append((kind, t, profile_id, payload))

    @property
    def decision_kinds(self):
        return [d[0] for d in self.decisions]


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
            rows.append(
                dict(
                    lifecycle=lifecycle,
                    storm_id=f"S{i:02d}",
                    hydro_tstp=dt_h,
                    date=t0 + pd.Timedelta(days=day, hours=dt_h),
                    wave_height=wave_height,
                    wave_peak_period=peak_period,
                    water_elevation=water_elevation,
                    wave_direction=0.0,
                )
            )
    return pd.DataFrame(rows)


def storms_at(
    times,
    *,
    wave_height: float = 1.2,
    water_elevation: float = 0.3,
    peak_period: float = 10.0,
    sim_start: datetime = SIM_START,
    lifecycle: int = 0,
) -> pd.DataFrame:
    """Storms placed at explicit day offsets — for engineering event sequences.

    Unlike ``storms(n)`` (rigid 20-day spacing), ``times`` gives exact control
    over storm timing so interrupt / blackout scenarios can be built.  Each entry
    is either a day offset (float) or a ``(day, wave_height)`` pair to vary
    intensity per storm.  Three hydrograph rows per storm, as in ``storms``.
    """
    rows = []
    t0 = pd.Timestamp(sim_start)
    for i, spec in enumerate(times):
        if isinstance(spec, (tuple, list)):
            day, hs = spec
        else:
            day, hs = spec, wave_height
        for dt_h in range(0, 13, 6):
            rows.append(
                dict(
                    lifecycle=lifecycle,
                    storm_id=f"S{i:02d}",
                    hydro_tstp=dt_h,
                    date=t0 + pd.Timedelta(days=float(day), hours=dt_h),
                    wave_height=hs,
                    wave_peak_period=peak_period,
                    water_elevation=water_elevation,
                    wave_direction=0.0,
                )
            )
    # ``times=[]`` is a stormless lifecycle — still a valid schedule, so keep the
    # columns the loop selects on rather than handing back a shapeless empty frame.
    return pd.DataFrame(rows, columns=_STORM_COLUMNS if not rows else None)


def forcing(hs=(1.0, 2.0)) -> dict:
    """Single-storm CSHORE BC dict for run_parallel_cshore tests."""
    hs = np.asarray(hs, dtype=float)
    n = len(hs)
    return {
        "timebc_wave": np.linspace(0.0, 3600.0 * (n - 1), n),
        "Hs": hs,
        "Hrms": hs / np.sqrt(2.0),
        "Tp": np.full(n, 9.0),
        "Wsetup": np.zeros(n),
        "swlbc": np.linspace(0.0, 0.3, n),
        "angle": np.zeros(n),
    }


def ncfg(
    volume_trigger: float | None = 0.001,
    production_rate: float = 500.0,
    assessor: str | None = None,
    *,
    emergency_volume: float | None = None,
    cycle_interval_years: float | None = None,
    cycle_start_date: datetime | None = None,
    blackout_windows: list | None = None,
    mobilization_days: float | None = None,
    storm_conflict: str | None = None,
    template_geometry: dict | None = None,
) -> NourishmentConfig:
    """The one doorway for nourishment policy in tests (parametric restore, no array).

    The restore geometry is synthesized from each profile's ``ref_metrics``, so
    this config carries only policy.  Every knob a scenario tunes is a keyword
    here — ``volume_trigger=None`` drops the regular gate for emergency-only
    reaches; only the validator tests build raw payloads.  ``assessor`` pins the
    Tier-1 assessor — pass ``"volume"`` to pair with ``template_profile`` without
    its dune auto-classifying to the legacy geometric assessor.  Validated with an
    ``input_units="m"`` context so the ``ufloat`` fields stay in metres.
    """
    payload: dict = {"production_rate": {"value": production_rate, "units": "m3/day"}}
    if volume_trigger is not None:
        payload["volume_trigger"] = {"value": volume_trigger, "units": "m3"}
    if assessor is not None:
        payload["assessor"] = assessor
    if emergency_volume is not None:
        payload["emergency_volume"] = {"value": emergency_volume, "units": "m3"}
    if cycle_interval_years is not None:
        payload["cycle_interval_years"] = cycle_interval_years
    if cycle_start_date is not None:
        payload["cycle_start_date"] = cycle_start_date
    if blackout_windows is not None:
        payload["blackout_windows"] = blackout_windows
    if mobilization_days is not None:
        payload["mobilization_days"] = mobilization_days
    if storm_conflict is not None:
        payload["storm_conflict"] = storm_conflict
    if template_geometry is not None:
        payload["template_geometry"] = template_geometry
    return NourishmentConfig.model_validate(payload, context={"input_units": "m"})


def template_profile(pid: str = "p0") -> Profile:
    """A clean as-built berm+dune profile, fit to ``ref_metrics`` so the parametric
    restore template reconstructs its own shape (fresh deficit ≈ 0).

    The dune is trapezoidal — an engineered dune is built to a design crest width, so a
    flat top is the case worth defaulting to; a triangular knife-edge is the exception.

    Damage it with the forms in ``tests/doubles.py`` (``BermCut``, ``DuneCut``,
    ``Overwash``): each states the damage as morphology and leaves a deficit the real
    assessor measures, so a scenario says what it means instead of tuning a scoop depth
    against a threshold.
    """
    x, z, _ = make_profile(
        berm_elevation=2.0,
        berm_width=30.0,
        dune=DuneSpec(shape="trapezoidal", crest_elevation=5.0, top_width=10.0),
    )
    ref, _ = fit_profile(x, z, 2.0, 0.0)
    return Profile(id=pid, x=x, zb=z.copy(), d50=0.3, ref_metrics=ref)


def run(
    profiles,
    n_storms: int = 3,
    *,
    storms_df: pd.DataFrame | None = None,
    sim_end: float | None = None,
    cfg: ReachConfig | None = None,
    runner=None,  # required; None only so the error below can name the fix
    sink=None,
    lifecycle: int = 0,
    sim_start: datetime = SIM_START,
    reach_id: str = "test",
    alternative_id: str = "FWOP",
):
    """Run one lifecycle; return ``(profiles, sink)``.

    ``runner`` is deliberately explicit — a scenario says which stand-in drives its
    storms (usually a ``ScriptedRunner`` over ``tests/doubles.py`` damage forms;
    ``MockCSHORERunner`` only where the bed shape is immaterial).  Pass ``storms_df``
    (e.g. from ``storms_at``) to drive a custom schedule; otherwise the default
    ``storms(n_storms)`` (20-day spacing) is used.  When ``sim_end`` is omitted it
    is derived as the last storm day + 60-day tail so a post-storm campaign has
    room to complete.
    """
    if runner is None:
        raise TypeError("builders.run() needs an explicit runner (no silent mock scoop)")
    cfg = cfg or ReachConfig()
    sink = sink if sink is not None else NullResultsSink()
    sdf = storms_df if storms_df is not None else storms(n_storms, lifecycle=lifecycle)
    if sim_end is None:
        last_day = (pd.to_datetime(sdf["date"]).max() - pd.Timestamp(sim_start)).total_seconds()
        sim_end = last_day / 86400.0 + 60.0
    reach = Reach(
        profiles=profiles,
        cfg=cfg,
        results=sink,
        runner=runner,
        sim_start=sim_start,
        reach_id=reach_id,
        alternative_id=alternative_id,
        lifecycle=lifecycle,
    )
    reach.run(sdf, sim_end)
    return reach.profiles, sink
