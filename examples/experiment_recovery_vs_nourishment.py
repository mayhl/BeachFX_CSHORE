#!/usr/bin/env python3
"""Experiment: recovery dynamics under repeated storm stress.

Runs a 5-storm sequence using MockCSHORERunner and plots the bed-level profile
at INIT, each PostStorm snapshot, and EndIteration.  No real binary required.

Usage:
    python examples/experiment_recovery_vs_nourishment.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from framework.events.background import BackgroundErosionEvent
from framework.events.lifecycle import EndOfLifecycleEvent, SnapshotEvent
from framework.events.storm import StormEvent
from framework.profile import Profile
from framework.queue import EventQueue
from framework.reach import Reach, ReachContext
from framework.runner.mock import MockCSHORERunner
from framework.types import ProfileState, ProfileStatus, SnapshotLabel


# ---------------------------------------------------------------------------
# Profile setup — simple linear shoreface
# ---------------------------------------------------------------------------

def make_profile(pid: str = "p0") -> Profile:
    n = 200
    x = np.linspace(0, 400, n)
    zb = np.linspace(-6.0, 4.0, n)
    return Profile(
        id=pid,
        x=x,
        zb=zb.copy(),
        zbe=zb.copy(),
        d50=0.3,
        status=ProfileStatus(state=ProfileState.QUIESCENT),
    )


def storm_forcing(peak_Hs: float = 2.0) -> dict:
    t = np.array([0.0, 3600.0, 7200.0, 10800.0])
    Hs = np.array([0.5, peak_Hs, peak_Hs * 0.8, 0.3])
    return {
        "timebc_wave": t,
        "Hs": Hs,
        "Hrms": Hs * 0.707,
        "Tp": np.array([6.0, 10.0, 9.0, 7.0]),
        "Wsetup": np.zeros(4),
        "swlbc": np.array([0.0, 0.4, 0.3, 0.05]),
        "angle": np.zeros(4),
    }


# ---------------------------------------------------------------------------
# Build queue: INIT → storms (t=10, 25, 40, 55, 70) → background erosion → EOL
# ---------------------------------------------------------------------------

def build_queue(runner: MockCSHORERunner) -> EventQueue:
    queue = EventQueue()
    queue.push(SnapshotEvent(t=0.0, label=SnapshotLabel.INIT))

    storm_times = [10.0, 25.0, 40.0, 55.0, 70.0]
    for t in storm_times:
        queue.push(StormEvent(t=t, storm_forcing=storm_forcing(), runner=runner))

    # Background erosion: 1 mm/day, every 7 days
    queue.push(BackgroundErosionEvent(t=1.0, rate=0.001, interval_days=7.0))

    queue.push(EndOfLifecycleEvent(t=90.0))
    return queue


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_scenario(label: str, force_catastrophic: bool) -> list:
    profile = make_profile()
    runner = MockCSHORERunner(force_catastrophic=force_catastrophic)
    queue = build_queue(runner)
    ctx = ReachContext.minimal()
    segment = Reach(profiles=[profile], event_queue=queue, ctx=ctx)
    segment.run()
    print(f"[{label}] {len(profile.snapshots)} snapshots taken")
    return profile.snapshots


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(snapshots_normal: list, snapshots_catastrophic: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Bed-level evolution under repeated storms\n(MockCSHORERunner)")

    for ax, snapshots, title in zip(
        axes,
        [snapshots_normal, snapshots_catastrophic],
        ["Normal storms", "Catastrophic storms"],
    ):
        x = np.linspace(0, 400, len(snapshots[0].zb))

        for snap in snapshots:
            if snap.label == SnapshotLabel.INIT:
                ax.plot(x, snap.zb, "k-", lw=2, label="INIT")
            elif snap.label == SnapshotLabel.PostStorm:
                ax.plot(x, snap.zb, "r-", lw=0.8, alpha=0.6)
            elif snap.label == SnapshotLabel.EndIteration:
                ax.plot(x, snap.zb, "b--", lw=2, label="End")

        ax.set_title(title)
        ax.set_xlabel("Cross-shore position (m)")
        ax.set_ylabel("Bed elevation (m)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "recovery_experiment.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    snaps_normal = run_scenario("normal", force_catastrophic=False)
    snaps_catastrophic = run_scenario("catastrophic", force_catastrophic=True)
    plot(snaps_normal, snaps_catastrophic)
