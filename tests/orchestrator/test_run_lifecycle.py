"""Integration-lite tests: run_lifecycle end-to-end with MockCSHORERunner."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from erosion.config import ReachConfig
from erosion.interstorm import UniformErosionConfig
from erosion.results import ParquetResultsSink, read_parquet_footer
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

    def test_events_parquet_matches_profile_logs(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)
            profiles = _run(n_storms=2, n_profiles=2, sink=sink)
            df = pd.read_parquet(os.path.join(sink.out_dir, "events.parquet"))
            # one row per applied event across all profiles
            assert len(df) == sum(len(p.events) for p in profiles)
            for col in ("profile_id", "event_seq", "event_type", "t", "label", "ref_pos"):
                assert col in df.columns
            assert set(df["event_type"]) <= {
                "ErosionTick",
                "StormResponse",
                "Recovery",
                "FullNourishment",
                "PartialNourishment",
                "NourishmentStart",
                "Inundation",
            }
            assert df["ref_pos"].isna().all()  # Phase-C seam still inert

    def test_config_embedded_in_parquet_footer(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, tick_days=5.0))
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0, config=cfg)
            _run(n_storms=2, cfg=cfg, sink=sink)
            for fname in ("events.parquet", "profiles.parquet"):
                footer = read_parquet_footer(os.path.join(sink.out_dir, fname))
                assert "beachfx_reach_id" in footer
                assert footer["beachfx_lifecycle"] == "0"
                assert footer["beachfx_config"] == cfg.model_dump_json()

    def test_footer_without_config_has_identity_only(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "r", "FWOP", lifecycle=0)  # no config passed
            _run(n_storms=2, sink=sink)
            footer = read_parquet_footer(os.path.join(sink.out_dir, "events.parquet"))
            assert "beachfx_config" not in footer
            assert "beachfx_reach_id" in footer


class TestWithErosionConfig:
    def test_erosion_lowers_prestorm_zb_vs_init(self):
        cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, tick_days=5.0))
        profiles = _run(n_storms=1, cfg=cfg)
        for p in profiles:
            init_zb = p.snapshots[0].zb
            pre_zb = next(s.zb for s in p.snapshots if s.label == SnapshotLabel.PreStorm)
            assert np.mean(pre_zb) < np.mean(init_zb)

    def test_erosion_ticks_between_storms(self):
        """Erosion/SLC must accrue in EVERY inter-storm interval, not just before
        the first storm. storms() places storms at t=20, 40 → the [20,40] gap must
        carry Periodic ticks (currently it does not — the interstorm interval is
        empty after run_campaign advances state.t to the next storm)."""
        cfg = ReachConfig(erosion=UniformErosionConfig(rate=0.01, tick_days=5.0))
        profiles = _run(n_storms=2, cfg=cfg)
        for p in profiles:
            gap_ticks = [
                s.t for s in p.snapshots if s.label == SnapshotLabel.Periodic and 20.0 < s.t < 40.0
            ]
            assert gap_ticks, "no erosion ticks in the [20,40] inter-storm gap"
