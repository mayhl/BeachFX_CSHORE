from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .metrics import MorphType
from .profile import Profiles, StormResponse, _shoreline_shift
from .types import SnapshotLabel, StormResponseType
from .units import ufloat

if TYPE_CHECKING:
    from .config import ReachConfig
    from .metrics import ProfileMetrics
    from .profile import Profile
    from .runner.base import CSHOREResult, CSHORERunner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class StormConfig(BaseModel):
    """Parameters governing both storm response and post-storm recovery.

    Recovery is always triggered by a storm, so the two are kept together.
    ``T_recover`` is interpreted as T90 (days to 90% recovery) when
    ``recovery_model="exponential"`` and as T100 when ``"linear"``.
    Per-profile ``recovery_duration`` in ``ProfileGeometryConfig`` overrides
    this reach-wide default when set.
    """

    T_recover: ufloat("days") = 21.0
    recovery_model: Literal["linear", "exponential"] = "linear"
    z_berm: ufloat("m", "ft") | None = None  # below-berm mask; None = blend all nodes
    hydro_dt: ufloat("hours") | None = None  # resample hydrograph; None = raw CHS intervals


# ---------------------------------------------------------------------------
# Storm schedule
# ---------------------------------------------------------------------------


class StormRecord(NamedTuple):
    t: float  # days since sim_start (storm start)
    storm_id: str
    forcing: dict  # CSHORE BC dict (timebc_wave, Hs, Hrms, Tp, Wsetup, swlbc, angle)
    duration: float = 0.0  # storm length in days; storm ends (and recovery begins) at t + duration


def build_storm_schedule(
    storms_df: pd.DataFrame,
    sim_start: datetime,
    cfg: ReachConfig,
    lifecycle: int = 0,
) -> list[StormRecord]:
    """Convert a storms DataFrame into a time-ordered list of StormRecord.

    storms_df columns: lifecycle, storm_id, hydro_tstp, date,
                       wave_height (Hs), wave_peak_period (Tp),
                       water_elevation (swlbc), wave_direction (angle).
    """
    lc_df = storms_df[storms_df["lifecycle"] == lifecycle]
    sim_ts = pd.Timestamp(sim_start)
    storm_cfg = cfg.storm

    records = []
    for storm_id, group in lc_df.groupby("storm_id", sort=False):
        group = group.sort_values("hydro_tstp")
        first_date = group["date"].iloc[0]
        t_days = float((first_date - sim_ts).total_seconds() / 86400.0)

        t_in_storm = (group["date"] - first_date).dt.total_seconds().values.astype(float)
        raw = {
            "Hs": group["wave_height"].values.astype(float),
            "Tp": group["wave_peak_period"].values.astype(float),
            "swlbc": group["water_elevation"].values.astype(float),
            "angle": group["wave_direction"].values.astype(float),
        }

        if storm_cfg.hydro_dt is not None and len(t_in_storm) > 1:
            dt_s = storm_cfg.hydro_dt * 3600.0
            t_new = np.arange(0.0, t_in_storm[-1] + dt_s * 0.5, dt_s)
            raw = {k: np.interp(t_new, t_in_storm, v) for k, v in raw.items()}
            t_in_storm = t_new

        Hs = raw["Hs"]
        forcing = {
            "timebc_wave": t_in_storm,
            "Hs": Hs,
            "Hrms": Hs / np.sqrt(2.0),
            "Tp": raw["Tp"],
            "Wsetup": np.zeros(len(t_in_storm)),
            "swlbc": raw["swlbc"],
            "angle": raw["angle"],
        }
        # hydrograph span in days; PostStorm/recovery begin at t + duration
        duration = float(t_in_storm[-1] / 86400.0)
        records.append(
            StormRecord(t=t_days, storm_id=str(storm_id), forcing=forcing, duration=duration)
        )

    return sorted(records, key=lambda r: r.t)


# ---------------------------------------------------------------------------
# Storm response classification
# ---------------------------------------------------------------------------


def classify_storm_response(
    m_pre: ProfileMetrics,
    m_post: ProfileMetrics,
    BE: float,
) -> StormResponseType:
    """Classify storm impact from pre/post snapshot metrics.

    Uses BeachFX Tech Ref §7.3 criteria:
      NORMAL        — post-storm dune still above max(UE, BE)
      CAT_DUNE_LOST — LOW_BERM pre-storm; dune crest eroded below max(UE, BE)
      CAT_PARTIAL   — LOW_UPLAND pre-storm; dune gone, berm survives (BW > 0)
      CAT_TOTAL     — LOW_UPLAND pre-storm; dune and berm both gone (BW == 0)
    """
    min_dune_elev = max(float(m_post.upland_elevation), float(BE))
    if float(m_post.dune_crest_elevation) > min_dune_elev:
        return StormResponseType.NORMAL
    if m_pre.morph_type == MorphType.LOW_BERM.value:
        return StormResponseType.CAT_DUNE_LOST
    if m_pre.morph_type == MorphType.LOW_UPLAND.value:
        return (
            StormResponseType.CAT_PARTIAL
            if m_post.berm_width > 0.0
            else StormResponseType.CAT_TOTAL
        )
    return StormResponseType.NORMAL  # HIGH_UPLAND — no dune to categorically lose


# ---------------------------------------------------------------------------
# Phase 2 runner
# ---------------------------------------------------------------------------


class _PreStorm(NamedTuple):
    """Pre-storm state captured before CSHORE mutates a profile's bed.

    Only ``zb`` is snapshotted; ``profile.x`` is never mutated by any event, so the
    fixed grid is read straight off ``profile`` at use.
    """

    profile: Profile
    zb: np.ndarray


@dataclass
class StormOutcome:
    """Per-profile result of one storm's CSHORE run — the storm->reach->campaign
    boundary record that replaces the parallel ``(results, zb_pre_new)`` lists and
    the separately-threaded ``inundated`` set.

    ``result`` is ``None`` exactly when CSHORE failed; ``inundated`` reads that as
    the storm-skipped condition (profile reused unchanged, Phase 3 skipped).
    """

    profile: Profile
    result: CSHOREResult | None  # None when CSHORE failed
    zb_pre: np.ndarray  # pre-storm bed shift-registered onto the fixed grid

    @property
    def inundated(self) -> bool:
        return self.result is None


def _apply_storm_result(
    pre: _PreStorm, r: CSHOREResult | None, t_storm: float, t_post: float
) -> StormOutcome:
    """Fold one profile's CSHORE result into a ``StormOutcome``.

    On success: shift-register the pre-storm bed onto the fixed grid, take the
    PostStorm snapshot (via ``StormResponse.apply``), and classify the response.
    On failure (``r is None``): snapshot the unmodified bed as INUNDATION.
    """
    p, zb_p = pre.profile, pre.zb
    x_p = p.x  # fixed grid; never mutated
    if r is None:
        # CSHORE failed: snapshot current (unmodified) zb with INUNDATION label
        p.snapshot(SnapshotLabel.INUNDATION, t_storm)
        p.snapshots[-1].storm_response_type = StormResponseType.INUNDATION
        return StormOutcome(p, None, zb_p.copy())

    # Shift-register pre-storm profile to post-storm shoreline position,
    # keeping everything on the original fixed x-grid.
    dx = _shoreline_shift(x_p, zb_p, r.x, r.zb)
    zb_pre = np.interp(x_p - dx, x_p, zb_p, left=zb_p[0], right=zb_p[-1])
    StormResponse(t=t_post, result=r).apply(p)  # interpolates onto x_p, PostStorm at storm end

    # Attach storm response classification when geometry metrics are available
    if p.geometry is not None:
        pre_snap = p.last_snapshot(SnapshotLabel.PreStorm)
        post_snap = p.snapshots[-1]
        if pre_snap and pre_snap.metrics and post_snap.metrics:
            post_snap.storm_response_type = classify_storm_response(
                pre_snap.metrics,
                post_snap.metrics,
                p.geometry.berm_elevation,
            )

    return StormOutcome(p, r, zb_pre)


def run_parallel_cshore(
    profiles: list[Profile],
    t_storm: float,
    forcing: dict,
    runner: CSHORERunner,
    cfg: ReachConfig,
    t_post: float | None = None,
) -> list[StormOutcome]:
    """Run CSHORE for all profiles in parallel.

    PreStorm snapshot taken at ``t_storm`` (storm start) before applying results;
    StormResponse.apply() takes the PostStorm snapshot at ``t_post`` (storm end,
    defaulting to ``t_storm``).  INUNDATION (CSHORE failure) is marked at
    ``t_storm``.  All snapshots share the original fixed ``profile.x`` grid.

    Returns one ``StormOutcome`` per profile (in input order), bundling its
    CSHORE result (``None`` on failure) with the pre-storm zb shift-registered
    onto the original fixed grid, ready for Recovery event construction.
    """
    if t_post is None:
        t_post = t_storm
    profiles = Profiles(profiles)  # collection sugar; a no-op for callers already passing one

    # Capture pre-storm state and take PreStorm snapshots
    pre_storm = [_PreStorm(p, p.zb.copy()) for p in profiles]
    profiles.snapshot_all(SnapshotLabel.PreStorm, t_storm)

    # Run CSHORE in parallel with failure isolation
    def _run_safe(profile: Profile) -> CSHOREResult | None:
        try:
            return runner.run(profile, forcing)
        except Exception as exc:
            log.warning("CSHORE failed for profile %s: %s", profile.id, exc)
            return None

    log.info("Storm t=%.1fd — running CSHORE for %d profile(s)", t_storm, len(profiles))
    # Each worker spawns a CSHORE subprocess (releases the GIL), so threads give
    # real parallelism — but cap workers at the CPU count to avoid oversubscription.
    max_workers = min(len(profiles), os.cpu_count() or len(profiles))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_safe, p) for p in profiles]
        cshore_results = [f.result() for f in futs]

    # Fold each result (or failure) into its outcome on the fixed grid.
    outcomes = [
        _apply_storm_result(pre, r, t_storm, t_post) for pre, r in zip(pre_storm, cshore_results)
    ]

    log.info(
        "Storm t=%.1fd — done (%d ok, %d failed)",
        t_storm,
        sum(not o.inundated for o in outcomes),
        sum(o.inundated for o in outcomes),
    )
    return outcomes


# ---------------------------------------------------------------------------
# Recovery fraction
# ---------------------------------------------------------------------------


def _recovery_fraction(dt: float, T_recover: float, model: str) -> float:
    """Fraction of recovery completed after ``dt`` days.

    Linear:      T_recover = days to 100% recovery.
    Exponential: T_recover = T90 (days to 90% recovery).
    """
    if dt <= 0 or T_recover <= 0:
        return 0.0
    if model == "exponential":
        k = -math.log(0.1) / T_recover  # T90 convention
        return float(1.0 - math.exp(-k * dt))
    return min(dt / T_recover, 1.0)  # linear
