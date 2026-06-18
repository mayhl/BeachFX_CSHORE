#!/usr/bin/env python3
"""Plot BeachFX-CSHORE pipeline results.

Usage
-----
# Results viewer — all profiles, all alternatives, single lifecycle:
    python examples/plot_results.py output/ex4 --reach Reach1

# Focus on one profile:
    python examples/plot_results.py output/ex4 --reach Reach1 --profile Reach1_p0

# Save instead of showing:
    python examples/plot_results.py output/ex4 --reach Reach1 --out output/ex4/plots

# Animate one profile:
    python examples/plot_results.py output/ex4 --reach Reach1 --profile Reach1_p0 --video
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from framework.viz import (load_runs, plot_metrics, plot_profile_evolution,
                           generate_profile_frames, generate_event_transition_frames,
                           make_profile_video)

import matplotlib
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BeachFX pipeline results")
    parser.add_argument("output_root",             help="Pipeline output root (e.g. output/ex4)")
    parser.add_argument("--reach",   default=None, help="Reach ID (auto-detected if only one)")
    parser.add_argument("--profile", default=None, help="Profile ID to plot (default: all)")
    parser.add_argument("--lc",      type=int, default=0, help="Lifecycle index (default: 0)")
    parser.add_argument("--out",     default=None, help="Directory to save plots (default: show)")
    parser.add_argument("--video", action="store_true",
                        help="Generate single-line snapshot frame series for ffmpeg")
    parser.add_argument("--events", action="store_true",
                        help="Generate before/after event transition frame series")
    args = parser.parse_args()

    root = os.path.abspath(args.output_root)
    if not os.path.isdir(root):
        print(f"Error: output directory not found: {root}")
        sys.exit(1)

    # Auto-detect reach
    reach_id = args.reach
    if reach_id is None:
        candidates = [d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)) and not d.endswith(".csv")]
        if len(candidates) == 1:
            reach_id = candidates[0]
        else:
            print(f"Multiple reaches found: {candidates}. Specify --reach.")
            sys.exit(1)

    runs = load_runs(root, reach_id, lifecycle=args.lc)
    if not runs:
        print(f"No lifecycle lc_{args.lc:04d} found under {reach_id}")
        sys.exit(1)

    print(f"Reach: {reach_id}   Alternatives: {list(runs.keys())}   lc={args.lc}")

    # Determine which profiles to plot
    first_run = next(iter(runs.values()))
    all_profiles = first_run.profile_ids
    profiles_to_plot = [args.profile] if args.profile else all_profiles
    print(f"Profiles: {profiles_to_plot}")

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        matplotlib.use("Agg")

    def _save_or_show(fig: object, name: str) -> None:
        if out_dir:
            path = os.path.join(out_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved → {path}")
            plt.close(fig)
        else:
            plt.show()

    for pid in profiles_to_plot:
        # --- Profile evolution (one per alternative) ---
        for alt_id, run in runs.items():
            fig = plot_profile_evolution(run, pid)
            _save_or_show(fig, f"{pid}_{alt_id}_profile_evolution.png")

        # --- Metrics comparison (all alternatives overlaid) ---
        has_metrics = any(r.metrics is not None for r in runs.values())
        if has_metrics:
            fig = plot_metrics(runs, pid)
            _save_or_show(fig, f"{pid}_metrics_comparison.png")
        else:
            print(f"  (no profile_metrics.parquet found — skipping metrics plot for {pid})")

        # --- Frame series ---
        if args.video:
            for alt_id, run in runs.items():
                frame_dir = os.path.join(out_dir or ".", f"{pid}_{alt_id}_frames")
                print(f"  generating frames → {frame_dir}/")
                generate_profile_frames(run, pid, frame_dir)

        if args.events:
            for alt_id, run in runs.items():
                frame_dir = os.path.join(out_dir or ".", f"{pid}_{alt_id}_event_frames")
                print(f"  generating event frames → {frame_dir}/")
                generate_event_transition_frames(run, pid, frame_dir)


if __name__ == "__main__":
    main()
