#!/usr/bin/env python3
"""Parallel-scaling test matrix: sweep workers × profile_workers.

Creates a temporary storms parquet with N_LIFECYCLES copies of the ex4
storm sequence, then runs run_pipeline.py for every (workers, profile_workers)
combination and reports wall-clock time in a table.

Usage:
    python examples/test_parallel_matrix.py [--lifecycles N] [--config PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Default matrix — rows=workers, cols=profile_workers
# Adjust to taste or pass --workers / --profile-workers on the CLI.
# ---------------------------------------------------------------------------
DEFAULT_WORKERS         = [1, 2, 4, 8]
DEFAULT_PROFILE_WORKERS = [1, 2, 3]

# Number of lifecycle copies to generate so there are enough jobs to expose
# Dask-worker scaling.  2 alts × N_LIFECYCLES = total jobs dispatched.
DEFAULT_LIFECYCLES = 8


def _make_multi_lc_storms(src_parquet: str, n_lifecycles: int, dst_parquet: str) -> None:
    """Duplicate the single-lifecycle storm parquet into N_LIFECYCLES copies."""
    df = pd.read_parquet(src_parquet)
    frames = []
    for lc in range(n_lifecycles):
        dup = df.copy()
        dup["lifecycle"] = lc
        frames.append(dup)
    pd.concat(frames, ignore_index=True).to_parquet(dst_parquet, index=False)
    print(f"  Generated {n_lifecycles}-lifecycle storms → {dst_parquet}")


def _run_once(
    config_path: str,
    workers: int,
    profile_workers: int,
    out_root: str,
) -> float:
    """Run run_pipeline.py with the given settings. Returns elapsed seconds."""
    cmd = [
        sys.executable,
        os.path.join(ROOT, "examples", "run_pipeline.py"),
        config_path,
        "--workers",         str(workers),
        "--profile-workers", str(profile_workers),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        # Print last few lines of stderr so failures are diagnosable
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        print(f"    [FAILED] workers={workers} pw={profile_workers}\n{tail}")
        return float("nan")

    return elapsed


def run(
    base_config: str,
    n_lifecycles: int,
    workers_list: list[int],
    profile_workers_list: list[int],
) -> None:
    with open(base_config) as f:
        cfg = json.load(f)

    src_storms = os.path.join(ROOT, cfg["paths"]["storms"])

    with tempfile.TemporaryDirectory(prefix="beachfx_matrix_") as tmp:
        # --- Build multi-lifecycle storms ---
        multi_storms = os.path.join(tmp, "storms_multi_lc.parquet")
        _make_multi_lc_storms(src_storms, n_lifecycles, multi_storms)

        # --- Patch config to point to new storms and a temp output dir ---
        out_root = os.path.join(tmp, "output")
        patched_cfg = dict(cfg)
        patched_cfg["paths"] = dict(cfg["paths"])
        patched_cfg["paths"]["storms"] = multi_storms
        patched_cfg["paths"]["output"] = out_root
        patched_cfg["paths"].pop("save_cshore", None)

        cfg_path = os.path.join(tmp, "matrix_config.json")
        with open(cfg_path, "w") as f:
            json.dump(patched_cfg, f)

        print(f"\nConfig: {base_config}")
        print(f"Lifecycles: {n_lifecycles}  →  ~{2 * n_lifecycles} jobs (2 alts × {n_lifecycles} lc)")
        print(f"Workers grid:          {workers_list}")
        print(f"Profile-workers grid:  {profile_workers_list}")
        print()

        # --- Header ---
        pw_header = "  ".join(f"pw={pw:>2}" for pw in profile_workers_list)
        print(f"{'workers':>8}  {pw_header}")
        print("-" * (10 + 8 * len(profile_workers_list)))

        results: dict[tuple[int, int], float] = {}

        for w in workers_list:
            row_times = []
            for pw in profile_workers_list:
                print(f"  Running workers={w:>2}, profile_workers={pw:>2} … ", end="", flush=True)
                elapsed = _run_once(cfg_path, w, pw, out_root)
                results[(w, pw)] = elapsed
                if elapsed != elapsed:   # nan
                    row_times.append("  FAIL")
                else:
                    row_times.append(f"{elapsed:6.1f}s")
                print(row_times[-1].strip())

            cells = "  ".join(f"{t:>6}" for t in row_times)
            print(f"{'w='+str(w):>8}  {cells}")

        # --- Summary table ---
        print("\n=== Summary (wall-clock seconds) ===")
        print(f"{'workers':>8}  {pw_header}")
        print("-" * (10 + 8 * len(profile_workers_list)))
        for w in workers_list:
            cells = "  ".join(
                f"{results[(w,pw)]:6.1f}s" if results[(w,pw)] == results[(w,pw)]
                else "  FAIL"
                for pw in profile_workers_list
            )
            print(f"{'w='+str(w):>8}  {cells}")

        # --- Best combo ---
        valid = {k: v for k, v in results.items() if v == v}
        if valid:
            best = min(valid, key=lambda k: valid[k])
            print(f"\nFastest: workers={best[0]}, profile_workers={best[1]}  ({valid[best]:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(ROOT, "examples/configs/ex4_multi_profile_multi_alt.json"),
        help="Base config JSON (storms path will be overridden)",
    )
    parser.add_argument("--lifecycles", type=int, default=DEFAULT_LIFECYCLES,
                        help=f"Number of lifecycle copies to generate (default: {DEFAULT_LIFECYCLES})")
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS,
                        help="Dask worker counts to test")
    parser.add_argument("--profile-workers", dest="profile_workers", type=int, nargs="+",
                        default=DEFAULT_PROFILE_WORKERS,
                        help="Profile-thread counts to test")
    args = parser.parse_args()

    run(
        base_config=args.config,
        n_lifecycles=args.lifecycles,
        workers_list=args.workers,
        profile_workers_list=args.profile_workers,
    )
