from __future__ import annotations

import math
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal, NamedTuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .types import SnapshotLabel, StormResponseType
from .units import ufloat

if TYPE_CHECKING:
    from .profile import Profile
    from .config import ReachConfig
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
    z_berm: ufloat("m", "ft") | None = None     # below-berm mask; None = blend all nodes
    hydro_dt: ufloat("hours") | None = None     # resample hydrograph; None = raw CHS intervals


# ---------------------------------------------------------------------------
# Storm schedule
# ---------------------------------------------------------------------------

class StormRecord(NamedTuple):
    t:        float   # days since sim_start
    storm_id: str
    forcing:  dict    # CSHORE BC dict (timebc_wave, Hs, Hrms, Tp, Wsetup, swlbc, angle)


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
            "Hs":    group["wave_height"].values.astype(float),
            "Tp":    group["wave_peak_period"].values.astype(float),
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
            "Hs":          Hs,
            "Hrms":        Hs / np.sqrt(2.0),
            "Tp":          raw["Tp"],
            "Wsetup":      np.zeros(len(t_in_storm)),
            "swlbc":       raw["swlbc"],
            "angle":       raw["angle"],
        }
        records.append(StormRecord(t=t_days, storm_id=str(storm_id), forcing=forcing))

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
    from .metrics import MorphType
    min_dune_elev = max(float(m_post.upland_elevation), float(BE))
    if float(m_post.dune_crest_elevation) > min_dune_elev:
        return StormResponseType.NORMAL
    if m_pre.morph_type == MorphType.LOW_BERM.value:
        return StormResponseType.CAT_DUNE_LOST
    if m_pre.morph_type == MorphType.LOW_UPLAND.value:
        return (StormResponseType.CAT_PARTIAL
                if m_post.berm_width > 0.0 else StormResponseType.CAT_TOTAL)
    return StormResponseType.NORMAL   # HIGH_UPLAND — no dune to categorically lose


# ---------------------------------------------------------------------------
# Phase 2 runner
# ---------------------------------------------------------------------------

def run_parallel_cshore(
    profiles: list[Profile],
    t_storm: float,
    forcing: dict,
    runner: CSHORERunner,
    cfg: ReachConfig,
) -> tuple[list[CSHOREResult | None], list[np.ndarray]]:
    """Run CSHORE for all profiles in parallel.

    PreStorm snapshot taken before applying results.
    StormResponse.apply() takes PostStorm snapshot (or INUNDATION if CSHORE
    fails).  All snapshots share the original fixed ``profile.x`` grid.

    Returns
    -------
    results
        CSHOREResult per profile, or ``None`` when CSHORE failed.
    zb_pre_new
        Pre-storm zb shift-registered onto the original fixed grid, ready
        for Recovery event construction.
    """
    from .profile import StormResponse, _shoreline_shift

    # Capture pre-storm state and take PreStorm snapshots
    x_pre  = [p.x.copy()  for p in profiles]
    zb_pre = [p.zb.copy() for p in profiles]
    for p in profiles:
        p.snapshot(SnapshotLabel.PreStorm, t_storm)

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
        futs    = [ex.submit(_run_safe, p) for p in profiles]
        raw     = [f.result() for f in futs]

    # Apply results; compute zb_pre_new on the original fixed grid
    results:     list[CSHOREResult | None] = []
    zb_pre_new:  list[np.ndarray]          = []

    for p, r, zb_p, x_p in zip(profiles, raw, zb_pre, x_pre):
        if r is None:
            # CSHORE failed: snapshot current (unmodified) zb with INUNDATION label
            p.snapshot(SnapshotLabel.INUNDATION, t_storm)
            p.snapshots[-1].storm_response_type = StormResponseType.INUNDATION
            zb_pre_new.append(zb_p.copy())
            results.append(None)
            continue

        # Shift-register pre-storm profile to post-storm shoreline position,
        # keeping everything on the original fixed x-grid.
        dx = _shoreline_shift(x_p, zb_p, r.x, r.zb)
        zb_pre_new.append(np.interp(
            x_p - dx, x_p, zb_p, left=zb_p[0], right=zb_p[-1],
        ))
        StormResponse(t=t_storm, result=r).apply(p)   # interpolates onto x_p, takes PostStorm

        # Attach storm response classification when geometry metrics are available
        if p.geometry is not None:
            pre_snap  = next(
                (s for s in reversed(p.snapshots[:-1])
                 if s.label == SnapshotLabel.PreStorm), None
            )
            post_snap = p.snapshots[-1]
            if pre_snap and pre_snap.metrics and post_snap.metrics:
                post_snap.storm_response_type = classify_storm_response(
                    pre_snap.metrics, post_snap.metrics, p.geometry.berm_elevation,
                )

        results.append(r)

    log.info("Storm t=%.1fd — done (%d ok, %d failed)",
             t_storm, sum(r is not None for r in results), sum(r is None for r in results))
    return results, zb_pre_new


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
        k = -math.log(0.1) / T_recover   # T90 convention
        return float(1.0 - math.exp(-k * dt))
    return min(dt / T_recover, 1.0)       # linear
