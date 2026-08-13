#!/usr/bin/env python3
"""Multi-reach, multi-alternative, multi-lifecycle pipeline runner.

Reads a config JSON, iterates over every (reach × alternative × lifecycle)
combination found in the storms parquet, and writes structured output under the
configured output root.  Lifecycles are dispatched as Dask futures.

Console entry point ``run-pipeline`` (see pyproject ``[project.scripts]``):

    uv run run-pipeline ex1            # resolves examples/configs/ex1*.json
    uv run run-pipeline path/to.json --workers 4

The Dask dashboard is at http://localhost:8787 while running.

Output layout:
    {output}/
        storm_events.csv              # one row per (lifecycle, storm_id)
        {reach_id}/{alternative_id}/lc_{lifecycle:04d}/
            profiles.parquet
            storm_hazard.parquet
            profile_metrics.parquet
            snapshots.parquet
            placements.csv
            grid.parquet            # common axis (+ georef seam); "postprocess": false skips
            hydro.parquet           # hydro registered onto the common grid
            run_metadata.json
            run_summary.txt
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

import dask
import pandas as pd
from dask.distributed import Client, LocalCluster, as_completed

from .config import (
    _LAYERED_SECTIONS,
    ProfileGeometryConfig,
    ReachConfig,
    _expand_plans,
    _fresh_profiles,
    _load_profiles,
    _merge_sections,
    _parse_run_spec,
    _priority_order,
    _require,
    _resolve_alternatives,
    _resolve_widths,
)
from .postprocess import postprocess_lifecycle
from .profile import Profile
from .reach import Reach
from .results import ParquetResultsSink
from .runner.local import LocalCSHORERunner

# Repo root = two levels up from this file (src/erosion/pipeline.py).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_DIR = os.path.join(ROOT, "examples", "configs")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storm events output
# ---------------------------------------------------------------------------


def _saffir_simpson(wind_ms: float) -> int:
    """Map 1-min sustained wind speed (m/s) to Saffir-Simpson category (0–5).

    Thresholds from NHC kt scale converted to m/s (1 kt = 0.5144 m/s):
        Cat 1: ≥ 64 kt (32.9 m/s)   Cat 2: ≥ 83 kt (42.7 m/s)
        Cat 3: ≥ 96 kt (49.4 m/s)   Cat 4: ≥ 113 kt (58.1 m/s)
        Cat 5: ≥ 137 kt (70.5 m/s)

    Category 0 covers tropical depressions and storms (< 64 kt).  CHS wind is
    local at the save point, not Vmax — categories are approximate (triage only).
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
    """Write {out_root}/storm_events.csv — one row per (lifecycle, storm_id)."""
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

        records.append(
            {
                "lifecycle": int(lc),
                "storm_id": str(storm_id),
                "event_date": event_date,
                "duration_hours": round(duration_hours, 2),
                "saffir_simpson": ss,
                "recurrence_interval_yr": None,  # needs a JPM rates file keyed by storm_id
                "aep": None,  # = 1 / recurrence_interval_yr
            }
        )

    out_df = pd.DataFrame(records)
    csv_path = os.path.join(out_root, "storm_events.csv")
    os.makedirs(out_root, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    log.info("Wrote storm_events.csv → %s  (%d rows)", csv_path, len(out_df))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolve_config(arg: str) -> str:
    """Resolve a config argument to a path.

    Accepts a path to an existing ``.json`` file, or a short name resolved
    against ``examples/configs/`` (e.g. ``ex1`` → ``ex1_single_reach...json``).
    """
    if os.path.isfile(arg):
        return arg
    named = os.path.join(_CONFIG_DIR, arg if arg.endswith(".json") else arg + ".json")
    if os.path.isfile(named):
        return named
    matches = sorted(glob.glob(os.path.join(_CONFIG_DIR, f"{arg}*.json")))
    if matches:
        return matches[0]
    raise SystemExit(f"config not found: {arg!r} (looked in {_CONFIG_DIR})")


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
    postprocess: bool = True  # common-grid registration pass after the flush


def _run_lifecycle(job: _LifecycleJob) -> tuple[str, str, int]:
    """Run one (reach, alternative, lifecycle).  Pure function — no shared state."""
    profiles = _fresh_profiles(job.base_profiles)

    with tempfile.TemporaryDirectory(
        prefix=f"cshore_{job.reach_id}_{job.alt_id}_lc{job.lc}_"
    ) as work_dir:
        sink = ParquetResultsSink(
            job.out_root, job.reach_id, job.alt_id, lifecycle=job.lc, config=job.cfg
        )
        infile_dir = os.path.join(sink.out_dir, "infiles") if job.save_cshore else None
        runner = LocalCSHORERunner(params=job.cfg.cshore, work_dir=work_dir, infile_dir=infile_dir)
        reach = Reach(
            profiles=profiles,
            cfg=job.cfg,
            results=sink,
            runner=runner,
            sim_start=job.sim_start,
            reach_id=job.reach_id,
            alternative_id=job.alt_id,
            lifecycle=job.lc,
            longshore_widths=job.longshore_widths,
        )
        reach.run(job.storms_df, job.sim_end)

    if job.postprocess:
        postprocess_lifecycle(sink.out_dir)

    return job.reach_id, job.alt_id, job.lc


def _build_jobs(
    reaches: dict,
    global_alts: dict,
    global_sections: dict,
    units_context: dict,
    storms_df: pd.DataFrame,
    sim_start: datetime,
    sim_end: float,
    out_root: str,
    save_cshore: bool,
    lifecycles: list,
    run_spec="all",
    base_dir: str = ".",
    postprocess: bool = True,
) -> list[_LifecycleJob]:
    """Expand the config into one ``_LifecycleJob`` per (reach × selected alternative
    × lifecycle).  ``run_spec`` (``"all"`` or a range/list like ``"1-4,8"``) picks
    which alternative ids to run."""
    # Slice storms once per lifecycle so each job carries only its own rows (reach.run
    # filters by lifecycle anyway).  Keeps the per-job payload tiny — essential when
    # workers are processes (no full-table copy per worker / per node).
    lc_slices = {int(lc): grp for lc, grp in storms_df.groupby("lifecycle")}
    all_jobs: list[_LifecycleJob] = []
    for reach_id, reach_data in reaches.items():
        alts = _resolve_alternatives(global_alts, reach_data.get("alternatives", {}))
        selected = _parse_run_spec(run_spec, list(alts))
        alts = {aid: alts[aid] for aid in selected}
        profile_paths = [os.path.join(base_dir, p) for p in reach_data["profiles"]]
        priorities = reach_data.get("profile_priority")

        order = _priority_order(priorities, len(profile_paths))
        longshore_widths = _resolve_widths(
            reach_data.get("longshore_width", 1.0),
            order,
            units_context,
        )

        _be = reach_data.get("berm_elevation")
        geometry = (
            ProfileGeometryConfig.model_validate(
                {"berm_elevation": _be, "datum": reach_data.get("datum", 0.0)},
                context=units_context,
            )
            if _be is not None
            else None
        )

        for alt_id, alt_data in alts.items():
            # A generated run may pin one lifecycle (top-level "lifecycle"); otherwise
            # cross every lifecycle in the storm file (the default / hand-written path).
            # ("name" is a gen-alternatives label; both are non-config keys ignored by
            # _merge_sections.)
            pinned_lc = alt_data.get("lifecycle")
            # Layer storm/cshore/erosion/slc global -> reach -> alt; nourishment stays
            # alt-level (schema-v2 Phase 2 will feed it via the plan lowering).
            sections = _merge_sections(global_sections, reach_data, alt_data)
            nourishment = alt_data.get("nourishment")
            if nourishment is not None:
                sections["nourishment"] = nourishment
            cfg = ReachConfig.model_validate(sections, context=units_context)
            base_profiles = _load_profiles(
                profile_paths, cfg.cshore.d50, reach_id, priorities, geometry
            )

            if pinned_lc is None:
                alt_lcs = lifecycles
            else:
                alt_lcs = [int(pinned_lc)]
                if int(pinned_lc) not in {int(x) for x in lifecycles}:
                    raise ValueError(
                        f"alternative {alt_id!r} pins lifecycle {pinned_lc} not in the storm "
                        f"file (available: {sorted(int(x) for x in lifecycles)})"
                    )

            for lc in alt_lcs:
                all_jobs.append(
                    _LifecycleJob(
                        reach_id=reach_id,
                        alt_id=alt_id,
                        lc=int(lc),
                        base_profiles=base_profiles,
                        cfg=cfg,
                        storms_df=lc_slices[int(lc)],
                        sim_start=sim_start,
                        sim_end=sim_end,
                        out_root=out_root,
                        longshore_widths=longshore_widths,
                        save_cshore=save_cshore,
                        postprocess=postprocess,
                    )
                )
    return all_jobs


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(
    config_path: str,
    max_workers: int | None = None,
    oversubscription: float = 2.0,
    run_select: str | None = None,
) -> None:
    with open(config_path) as f:
        cfg_raw = json.load(f)
    cfg_raw = _expand_plans(cfg_raw)  # schema-v2 nourishment plans -> alternatives (no-op for v1)
    # Every relative path in the config resolves against the config file's own
    # directory (absolute paths pass through), so a config + its data travel as
    # one relocatable bundle
    base_dir = os.path.dirname(os.path.abspath(config_path))

    paths = _require(cfg_raw, "paths", "top level")
    sim = _require(cfg_raw, "simulation", "top level")
    global_sections = {s: cfg_raw[s] for s in _LAYERED_SECTIONS if s in cfg_raw}
    global_alts = cfg_raw.get("alternatives", {})
    reaches = _require(cfg_raw, "reaches", "top level")
    postprocess = bool(cfg_raw.get("postprocess", True))
    # Which alternative ids to run: CLI --run overrides the config "run" (default all).
    run_spec = run_select if run_select is not None else cfg_raw.get("run", "all")

    sim_start = datetime.fromisoformat(_require(sim, "sim_start", 'in "simulation"'))
    sim_end = float(_require(sim, "sim_end_days", 'in "simulation"'))
    out_root = os.path.join(base_dir, _require(paths, "output", 'in "paths"'))
    save_cshore = paths.get("save_cshore", False)

    input_units = cfg_raw.get("units", {}).get("input", "ft")
    units_context = {"input_units": input_units}

    storms_df = pd.read_parquet(os.path.join(base_dir, _require(paths, "storms", 'in "paths"')))
    lifecycles = sorted(storms_df["lifecycle"].unique())

    # Validate every reach/alternative config BEFORE the cluster spins up, so a
    # config typo fails in seconds instead of after workers are provisioned.
    all_jobs = _build_jobs(
        reaches,
        global_alts,
        global_sections,
        units_context,
        storms_df,
        sim_start,
        sim_end,
        out_root,
        save_cshore,
        lifecycles,
        run_spec=run_spec,
        base_dir=base_dir,
        postprocess=postprocess,
    )
    log.info("Alternative selection: run=%s", run_spec)

    cpu_count = os.cpu_count() or 1
    n_slots = max(1, int(cpu_count * oversubscription))
    n_workers = max_workers or n_slots
    log.info(
        "Slots: cpu=%d × %.1f = %d  →  %d Dask worker(s)",
        cpu_count,
        oversubscription,
        n_slots,
        n_workers,
    )

    # In-process threaded cluster: a long synchronous CSHORE subprocess can
    # starve the shared event loop and trip heartbeat timeouts (logged as ERROR
    # but harmless — the task still completes).  Give heartbeats slack, stop the
    # scheduler reaping a busy worker, and quiet routine distributed chatter.
    dask.config.set(
        {
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
            "distributed.scheduler.worker-ttl": None,
            "distributed.admin.tick.limit": "1h",
        }
    )
    logging.getLogger("distributed").setLevel(logging.WARNING)

    # One thread per worker so the GIL never bottlenecks; CSHORE runs as a subprocess.
    # processes=True (not threads): each worker is its own process, so the transient
    # memory CSHORE result-parsing fragments off the heap is returned to the OS when a
    # worker recycles, and the nanny restarts any worker that exceeds memory_limit —
    # bounding memory over a long run.  Per-job payloads are per-lifecycle storm slices
    # (see _build_jobs), so process isolation costs no full-table duplication (scales to
    # many workers / multiple nodes).
    with (
        LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="auto",
            dashboard_address="localhost:8787",
        ) as cluster,
        Client(cluster) as client,
    ):
        log.info(
            "Dask dashboard → http://localhost:8787/status  (%d worker(s), %d lifecycle(s))",
            n_workers,
            len(lifecycles),
        )

        log.info("Submitting %d job(s) total", len(all_jobs))
        # pure=False → random task keys; skips deterministic tokenization of the
        # job payload (Profile dataclasses w/ numpy arrays, storms_df), which
        # newer Dask cannot hash and would otherwise raise TokenizationError on.
        fut_to_job = {
            fut: job for fut, job in zip(client.map(_run_lifecycle, all_jobs, pure=False), all_jobs)
        }

        total_done = total_failed = 0
        for fut in as_completed(fut_to_job):
            job = fut_to_job.pop(fut)
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


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="run-pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default="ex1",
        help="config name (e.g. ex1) resolved against examples/configs/, or a .json path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dask worker count (default: cpu_count × oversubscription)",
    )
    parser.add_argument(
        "--oversubscription",
        type=float,
        default=2.0,
        help="CSHORE slot multiplier relative to cpu_count (default: 2.0)",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="alternative ids to run: 'all' or a range/list e.g. '1-4,8,11-14' "
        "(overrides the config 'run' field)",
    )
    args = parser.parse_args(argv)
    run(
        _resolve_config(args.config),
        max_workers=args.workers,
        oversubscription=args.oversubscription,
        run_select=args.run,
    )


if __name__ == "__main__":
    main()
