#!/usr/bin/env python3
"""Example: storm + recovery dynamics on a single profile.

Runs 3 storms spaced 30 days apart using MockCSHORERunner and plots
the bed-level profile at each snapshot label (INIT, PreStorm, PostStorm, REC, End).
No CSHORE binary required.

Usage:
    python examples/experiment_storm_recovery.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework.config import CSHOREParams, ReachConfig, StormConfig
from framework.profile import Profile
from framework.queue_builder import build_queue
from framework.reach import Reach, ReachContext
from framework.runner.mock import MockCSHORERunner
from framework.types import ProfileState, ProfileStatus, SnapshotLabel


# ---------------------------------------------------------------------------
# Profile setup — simple linear shoreface
# ---------------------------------------------------------------------------

def make_profile(d50: float = 0.3, pid: str = "p0") -> Profile:
    n = 200
    x = np.linspace(0.0, 400.0, n)
    zb = np.linspace(-6.0, 4.0, n)
    return Profile(
        id=pid,
        x=x,
        zb=zb.copy(),
        zbe=zb.copy(),
        d50=d50,
        status=ProfileStatus(state=ProfileState.QUIESCENT),
    )


# ---------------------------------------------------------------------------
# Build a minimal storms DataFrame from a list of storm start times (days)
# ---------------------------------------------------------------------------

def make_storms_df(
    sim_start: datetime,
    storm_times_days: list[float],
    peak_Hs: float = 2.5,
) -> pd.DataFrame:
    rows = []
    origin = pd.Timestamp(sim_start)
    for i, t_days in enumerate(storm_times_days):
        base = origin + pd.Timedelta(days=t_days)
        Hs_series = [0.5, peak_Hs, peak_Hs * 0.8, 0.3]
        swl_series = [0.0, 0.5, 0.4, 0.05]
        for step in range(4):
            rows.append({
                "storm_id": f"storm_{i:02d}",
                "lifecycle": 0,
                "hydro_tstp": step,
                "date": base + pd.Timedelta(hours=step),
                "wave_height": Hs_series[step],
                "wave_peak_period": [6.0, 10.0, 9.0, 6.0][step],
                "water_elevation": swl_series[step],
                "wave_direction": 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment() -> tuple[Profile, ReachConfig]:
    cfg = ReachConfig(
        storm=StormConfig(T_recover=21.0, catastrophic_jdry_threshold=3),
        cshore=CSHOREParams(d50=0.3, gamma=0.7, effb=0.002, blp=0.001),
    )

    profile = make_profile(d50=cfg.cshore.d50)
    runner = MockCSHORERunner(force_catastrophic=False)

    sim_start = datetime(2025, 1, 1)
    storms_df = make_storms_df(sim_start, storm_times_days=[10.0, 40.0, 70.0])

    queue = build_queue(
        storms_df=storms_df,
        profiles=[profile],
        runner=runner,
        sim_start=sim_start,
        sim_end=100.0,
        cfg=cfg,
    )

    ctx = ReachContext.minimal(cfg=cfg)
    reach = Reach(profiles=[profile], event_queue=queue, ctx=ctx)
    reach.run()

    print(f"Total snapshots: {len(profile.snapshots)}")
    for s in profile.snapshots:
        print(f"  t={s.t:6.1f}d  {s.label.value}")

    return profile, cfg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(profile: Profile) -> None:
    x = profile.x

    label_styles: dict[SnapshotLabel, tuple] = {
        SnapshotLabel.INIT:         ("black",     2.5, "solid",  "INIT"),
        SnapshotLabel.PreStorm:     ("steelblue", 1.0, "dashed", "PreStorm"),
        SnapshotLabel.PostStorm:    ("crimson",   1.0, "solid",  "PostStorm"),
        SnapshotLabel.REC:          ("forestgreen", 1.5, "solid", "Recovery complete"),
        SnapshotLabel.EndIteration: ("purple",    2.5, "solid",  "End"),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Storm + Recovery: bed-level evolution (MockCSHORERunner)\n"
                 "3 storms at t=10, 40, 70 days | T_recover=21 days")

    seen: set = set()
    for snap in profile.snapshots:
        if snap.label not in label_styles:
            continue
        color, lw, ls, lbl = label_styles[snap.label]
        label = lbl if snap.label not in seen else None
        seen.add(snap.label)
        ax.plot(x, snap.zb, color=color, lw=lw, ls=ls, label=label, alpha=0.75)

    ax.axhline(0.0, color="gray", lw=0.5, ls="--", label="SWL")
    ax.set_xlabel("Cross-shore position (m)")
    ax.set_ylabel("Bed elevation (m)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "storm_recovery_experiment.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    profile, cfg = run_experiment()
    plot(profile)
