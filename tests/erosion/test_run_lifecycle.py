"""Integration-lite tests: run_lifecycle end-to-end with MockCSHORERunner."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.results import ParquetResultsSink
from erosion.types import SnapshotLabel
from tests.builders import profile, run


def _p(pid: str = "p0", n: int = 100):
    return profile(pid=pid, n=n, x_max=200.0, zb="ramp")


def _run(
    n_storms: int = 3,
    n_profiles: int = 2,
    cfg: ReachConfig | None = None,
    sink=None,
    lifecycle: int = 0,
):
    profiles, _ = run(
        [_p(f"p{i}") for i in range(n_profiles)], n_storms, cfg=cfg, sink=sink, lifecycle=lifecycle
    )
    return profiles


class TestSnapshotSequence:
    def test_init_snapshot_first(self):
        profiles = _run(n_storms=3)
        for p in profiles:
            assert p.snapshots[0].label == SnapshotLabel.INIT

    def test_init_snapshot_at_t_zero(self):
        profiles = _run(n_storms=3)
        for p in profiles:
            assert p.snapshots[0].t == pytest.approx(0.0)

    def test_end_iteration_last(self):
        profiles = _run(n_storms=3)
        for p in profiles:
            assert p.snapshots[-1].label == SnapshotLabel.EndIteration

    def test_prestorm_count_matches_storms(self):
        n = 3
        profiles = _run(n_storms=n)
        for p in profiles:
            count = sum(1 for s in p.snapshots if s.label == SnapshotLabel.PreStorm)
            assert count == n

    def test_poststorm_count_matches_storms(self):
        n = 3
        profiles = _run(n_storms=n)
        for p in profiles:
            count = sum(1 for s in p.snapshots if s.label == SnapshotLabel.PostStorm)
            assert count == n

    def test_prestorm_before_poststorm(self):
        profiles = _run(n_storms=2)
        for p in profiles:
            pre_ts = [s.t for s in p.snapshots if s.label == SnapshotLabel.PreStorm]
            post_ts = [s.t for s in p.snapshots if s.label == SnapshotLabel.PostStorm]
            for pre, post in zip(pre_ts, post_ts):
                assert pre <= post

    def test_snapshot_times_monotone(self):
        profiles = _run(n_storms=3)
        for p in profiles:
            ts = [s.t for s in p.snapshots]
            assert ts == sorted(ts)


class TestOutputSchema:
    def test_profiles_parquet_no_zbe(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            _run(sink=sink)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profiles.parquet"))
            assert "zbe" not in df.columns

    def test_storm_hazard_n_unique_storms(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            _run(n_storms=3, n_profiles=2, sink=sink)
            df = pd.read_parquet(os.path.join(sink.out_dir, "storm_hazard.parquet"))
            assert len(df["t_storm"].unique()) == 3


class TestWithErosionConfig:
    def test_periodic_snapshots_created(self):
        cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.005, interval=10.0))
        profiles = _run(n_storms=1, cfg=cfg)
        for p in profiles:
            periodic = [s for s in p.snapshots if s.label == SnapshotLabel.Periodic]
            assert len(periodic) > 0

    def test_erosion_lowers_prestorm_zb_vs_init(self):
        cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, interval=5.0))
        profiles = _run(n_storms=1, cfg=cfg)
        for p in profiles:
            init_zb = p.snapshots[0].zb
            pre_zb = next(s.zb for s in p.snapshots if s.label == SnapshotLabel.PreStorm)
            assert np.mean(pre_zb) < np.mean(init_zb)
