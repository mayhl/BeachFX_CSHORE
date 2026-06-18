#!/usr/bin/env python3
"""Full multi-reach, multi-alternative, multi-lifecycle pipeline runner.

Reads config.json, iterates over every (reach × alternative × lifecycle)
combination found in the storms parquet, and writes structured output under
the configured output root.  Lifecycles within each (reach, alternative) pair
are dispatched as Dask futures.

Usage:
    python examples/run_pipeline.py [config.json] [--workers N]

    Dask dashboard is available at http://localhost:8787 while running.

Output layout:
    {output}/
        storm_events.csv              # one row per (lifecycle, storm_id)
        {reach_id}/{alternative_id}/lc_{lifecycle:04d}/
            profiles.parquet          # all labeled snapshots — (profile, label, node) rows
            storm_hazard.parquet      # CSHORE spatial output — (profile, storm, node) rows
            profile_metrics.parquet   # 0-D morphology metrics per (profile, snapshot)
            profile_events.parquet    # event log — (profile, event) rows
            segment_events.csv        # segment-level events (nourishment, blackout)
            run_metadata.json
            run_summary.txt
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from dask.distributed import Client, LocalCluster, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from framework.config import ProfileGeometryConfig, ReachConfig
from framework.profile import Profile
from framework.reach import ReachContext, run_lifecycle
from framework.results import ParquetResultsSink
from framework.runner.local import LocalCSHORERunner
from framework.units import parse_ufloat

from utils.geometry import load_raw_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storm events output
# ---------------------------------------------------------------------------

def _saffir_simpson(wind_ms: float) -> int:
    """Map 1-min sustained wind speed (m/s) to Saffir-Simpson category (0–5).

    Thresholds from NHC kt scale converted to m/s (1 kt = 0.5144 m/s):
        Cat 1: ≥ 64 kt (32.9 m/s)
        Cat 2: ≥ 83 kt (42.7 m/s)
        Cat 3: ≥ 96 kt (49.4 m/s)
        Cat 4: ≥ 113 kt (58.1 m/s)
        Cat 5: ≥ 137 kt (70.5 m/s)

    Category 0 covers tropical depressions and tropical storms (< 64 kt).
    Caveat: CHS wind is local at the save point, not Vmax — categories are
    approximate and should be used for triage only.
    """
    if wind_ms < 32.9:
        return 0
    elif wind_ms < 42.7:
        return 1
    elif wind_ms < 49.4:
        return 2
    elif wind_ms < 58.1:
        return 3
    elif wind_ms < 70.5:
        return 4
    else:
        return 5


def _write_storm_events(storms_df: pd.DataFrame, out_root: str) -> None:
    """Write {out_root}/storm_events.csv — one row per (lifecycle, storm_id).

    Columns:
        lifecycle              – integer lifecycle index
        storm_id               – storm identifier string
        event_date             – ISO date of first hydrograph timestep
        duration_hours         – hydrograph span (last - first timestep) in hours
        saffir_simpson         – 0-5 derived from peak wind speed; None if
                                 wind_speed_ms column is absent from storms parquet
        recurrence_interval_yr – None: requires separate JPM rates file keyed by
                                 storm_id (not present in CHS timeseries CSV)
        aep                    – None: same as above; aep = 1 / recurrence_interval_yr
    """
    has_wind = "wind_speed_ms" in storms_df.columns

    records = []
    for (lc, storm_id), grp in storms_df.groupby(["lifecycle", "storm_id"], sort=True):
        dates = pd.to_datetime(grp["date"])
        event_date = dates.min().date().isoformat()
        duration_hours = (dates.max() - dates.min()).total_seconds() / 3600.0

        if has_wind:
            peak_wind = grp["wind_speed_ms"].max()
            ss = _saffir_simpson(float(peak_wind)) if not pd.isna(peak_wind) else None
        else:
            ss = None

        records.append({
            "lifecycle":             int(lc),
            "storm_id":              str(storm_id),
            "event_date":            event_date,
            "duration_hours":        round(duration_hours, 2),
            "saffir_simpson":        ss,
            "recurrence_interval_yr": None,
            "aep":                   None,
        })

    out_df = pd.DataFrame(records)
    csv_path = os.path.join(out_root, "storm_events.csv")
    os.makedirs(out_root, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    log.info("Wrote storm_events.csv → %s  (%d rows)", csv_path, len(out_df))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_widths(
    w_raw,
    priority_order: list[int],
    context: dict,
) -> list[float]:
    """Return per-profile longshore widths (metres) in priority-reordered list order.

    w_raw may be a scalar or list (in original profile-list order).
    A scalar is divided evenly across all profiles.
    """
    n = len(priority_order)
    if isinstance(w_raw, list):
        parsed = [parse_ufloat(v, "m", "ft", context) for v in w_raw]
        return [parsed[i] for i in priority_order]
    w_m = parse_ufloat(w_raw, "m", "ft", context)
    return [w_m / n] * n


def _merge_cshore(base: dict, *overrides: dict) -> dict:
    out = dict(base)
    for layer in overrides:
        out.update(layer)
    return out


def _merge_alt(global_alt: dict, reach_alt: dict) -> dict:
    merged: dict = {}
    for key in set(global_alt) | set(reach_alt):
        g = global_alt.get(key)
        r = reach_alt.get(key)
        if isinstance(g, dict) or isinstance(r, dict):
            merged[key] = {**(g or {}), **(r or {})}
        else:
            merged[key] = r if r is not None else g
    return merged


def _resolve_alternatives(global_alts: dict, reach_alts: dict) -> dict:
    all_ids = set(global_alts) | set(reach_alts)
    return {
        alt_id: _merge_alt(global_alts.get(alt_id, {}), reach_alts.get(alt_id, {}))
        for alt_id in all_ids
    }


def _load_profiles(
    profile_paths: list[str],
    d50: float,
    reach_id: str,
    priorities: list[int] | None = None,
    geometry: ProfileGeometryConfig | None = None,
) -> list[Profile]:
    """Load and return profiles sorted by priority (lower number = first in list).

    Profile IDs use original list indices so they stay stable regardless of priority.
    Default priority is list order (0, 1, 2, …).
    """
    n = len(profile_paths)
    prios = priorities if priorities is not None else list(range(n))
    if len(prios) != n:
        raise ValueError(
            f"profile_priority has {len(prios)} entries but profiles has {n}"
        )

    order = sorted(range(n), key=lambda i: prios[i])
    profiles = []
    for i in order:
        raw = load_raw_profile(profile_paths[i], d50)
        profiles.append(Profile(
            id=f"{reach_id}_p{i}",
            x=raw["x"],
            zb=raw["z"].copy(),
            d50=raw["d50"],
            geometry=geometry,
        ))
    return profiles


def _fresh_profiles(profiles: list[Profile]) -> list[Profile]:
    return [Profile(
        id=p.id,
        x=p.x.copy(),
        zb=p.zb.copy(),
        d50=p.d50,
        geometry=p.geometry,
    ) for p in profiles]


# ---------------------------------------------------------------------------
# Per-lifecycle unit of work
# ---------------------------------------------------------------------------

@dataclass
class _LifecycleJob:
    reach_id: str
    alt_id: str
    lc: int
    base_profiles: list[Profile]
    cfg: ReachConfig
    storms_df: pd.DataFrame
    sim_start: datetime
    sim_end: float
    out_root: str
    longshore_widths: list[float] = field(default_factory=list)  # metres, parallel to profiles
    save_cshore: bool = False


def _run_lifecycle(job: _LifecycleJob) -> tuple[str, str, int]:
    """Run one (reach, alternative, lifecycle). Pure function — no shared state."""
    profiles = _fresh_profiles(job.base_profiles)

    with tempfile.TemporaryDirectory(
        prefix=f"cshore_{job.reach_id}_{job.alt_id}_lc{job.lc}_"
    ) as work_dir:
        sink = ParquetResultsSink(job.out_root, job.reach_id, job.alt_id, lifecycle=job.lc)
        infile_dir = os.path.join(sink.out_dir, "infiles") if job.save_cshore else None
        runner = LocalCSHORERunner(params=job.cfg.cshore, work_dir=work_dir, infile_dir=infile_dir)
        ctx = ReachContext(
            reach_id=job.reach_id,
            alternative_id=job.alt_id,
            results=sink,
            sim_start=job.sim_start,
            cfg=job.cfg,
            longshore_widths=job.longshore_widths,
        )
        run_lifecycle(
            profiles, job.storms_df, job.sim_start, job.sim_end,
            job.cfg, runner, ctx, lifecycle=job.lc,
        )

    return job.reach_id, job.alt_id, job.lc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    config_path: str,
    max_workers: int | None = None,
    oversubscription: float = 2.0,
) -> None:
    with open(config_path) as f:
        cfg_raw = json.load(f)

    paths       = cfg_raw["paths"]
    sim         = cfg_raw["simulation"]
    base_cshore = cfg_raw.get("cshore", {})
    global_alts = cfg_raw.get("alternatives", {})
    reaches     = cfg_raw["reaches"]

    sim_start = datetime.fromisoformat(sim["sim_start"])
    sim_end   = float(sim["sim_end_days"])
    out_root    = paths["output"]
    save_cshore = paths.get("save_cshore", False)

    units_cfg    = cfg_raw.get("units", {})
    input_units  = units_cfg.get("input", "ft")
    units_context = {"input_units": input_units}

    storms_df  = pd.read_parquet(os.path.join(ROOT, paths["storms"]))
    lifecycles = sorted(storms_df["lifecycle"].unique())

    cpu_count = os.cpu_count() or 1
    n_slots   = max(1, int(cpu_count * oversubscription))
    n_workers = max_workers or n_slots
    log.info(
        "Slots: cpu=%d × %.1f = %d  →  %d Dask worker(s)",
        cpu_count, oversubscription, n_slots, n_workers,
    )

    # One thread per worker: CSHORE runs as a subprocess so the GIL is released
    # during the heavy computation.  processes=False avoids serialising storms_df
    # once per job — workers share memory instead.
    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address="localhost:8787",
    ) as cluster, Client(cluster) as client:

        log.info("Dask dashboard → http://localhost:8787/status  (%d worker(s), %d lifecycle(s))", n_workers, len(lifecycles))

        # Build all jobs across every (reach × alternative × lifecycle) first,
        # then submit the full batch so Dask can schedule them in parallel.
        all_jobs: list[_LifecycleJob] = []

        for reach_id, reach_data in reaches.items():
            reach_cshore  = reach_data.get("cshore", {})
            reach_alts    = reach_data.get("alternatives", {})
            alts          = _resolve_alternatives(global_alts, reach_alts)
            profile_paths = [os.path.join(ROOT, p) for p in reach_data["profiles"]]
            priorities    = reach_data.get("profile_priority")

            n_profiles = len(profile_paths)
            prios      = priorities if priorities is not None else list(range(n_profiles))
            order      = sorted(range(n_profiles), key=lambda i: prios[i])
            longshore_widths = _resolve_widths(
                reach_data.get("longshore_width", 1.0), order, units_context,
            )

            _be = reach_data.get("berm_elevation")
            geometry = (
                ProfileGeometryConfig.model_validate(
                    {"berm_elevation": _be, "datum": reach_data.get("datum", 0.0)},
                    context=units_context,
                )
                if _be is not None else None
            )

            for alt_id, alt_data in alts.items():
                alt_cshore    = alt_data.pop("cshore", {})
                merged_cshore = _merge_cshore(base_cshore, reach_cshore, alt_cshore)
                cfg           = ReachConfig.model_validate(
                    {**alt_data, "cshore": merged_cshore},
                    context=units_context,
                )
                base_profiles = _load_profiles(profile_paths, cfg.cshore.d50, reach_id, priorities, geometry)

                for lc in lifecycles:
                    all_jobs.append(_LifecycleJob(
                        reach_id=reach_id, alt_id=alt_id, lc=int(lc),
                        base_profiles=base_profiles, cfg=cfg,
                        storms_df=storms_df, sim_start=sim_start,
                        sim_end=sim_end, out_root=out_root,
                        longshore_widths=longshore_widths,
                        save_cshore=save_cshore,
                    ))

        log.info("Submitting %d job(s) total", len(all_jobs))
        fut_to_job = {fut: job for fut, job in zip(client.map(_run_lifecycle, all_jobs), all_jobs)}

        total_done = total_failed = 0
        for fut in as_completed(fut_to_job):
            job = fut_to_job.pop(fut)  # pop → releases job reference as each future completes
            try:
                _, _, lc = fut.result()
                log.info("  ✓  lc=%04d  %s / %s", lc, job.reach_id, job.alt_id)
                total_done += 1
            except Exception as exc:
                log.error("  ✗  lc=%04d  %s / %s  — %s", job.lc, job.reach_id, job.alt_id, exc)
                total_failed += 1

    log.info("Pipeline complete — %d succeeded, %d failed", total_done, total_failed)

    _write_storm_events(storms_df, out_root)

    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=os.path.join(ROOT, "config.json"))
    parser.add_argument("--workers", type=int, default=None,
                        help="Dask worker count (default: n_slots = cpu_count × oversubscription)")
    parser.add_argument("--oversubscription", type=float, default=2.0,
                        help="CSHORE slot multiplier relative to cpu_count (default: 2.0)")
    args = parser.parse_args()
    run(args.config, max_workers=args.workers, oversubscription=args.oversubscription)
