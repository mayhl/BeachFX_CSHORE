"""Unit tests for ParquetResultsSink and RunMeta."""
import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from framework.config import ReachConfig
from framework.profile import Profile
from framework.reach import ReachContext, run_lifecycle
from framework.results import NullResultsSink, ParquetResultsSink, RunMeta
from framework.runner.mock import MockCSHORERunner

SIM_START = datetime(2030, 1, 1)


def _p(pid: str = "p0", n: int = 50) -> Profile:
    x = np.linspace(0, 100, n)
    return Profile(id=pid, x=x, zb=np.linspace(-1.0, 3.0, n), d50=0.3)


def _storms(n: int = 2) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp(SIM_START)
    for i in range(n):
        day = (i + 1) * 20
        for dt_h in range(0, 13, 6):
            rows.append(dict(lifecycle=0, storm_id=f"S{i}", hydro_tstp=dt_h,
                             date=t0 + pd.Timedelta(days=day, hours=dt_h),
                             wave_height=1.2, wave_peak_period=10.0,
                             water_elevation=0.3, wave_direction=0.0))
    return pd.DataFrame(rows)


def _run(out_root: str, n_storms: int = 2) -> ParquetResultsSink:
    profiles = [_p("p0"), _p("p1")]
    sink = ParquetResultsSink(out_root, "R1", "FWOP", lifecycle=0)
    ctx = ReachContext(reach_id="R1", alternative_id="FWOP", results=sink,
                      sim_start=SIM_START, cfg=ReachConfig())
    run_lifecycle(profiles, _storms(n_storms), SIM_START, (n_storms + 1) * 20 + 10.0,
                  ReachConfig(), MockCSHORERunner(), ctx, lifecycle=0)
    return sink


class TestRunMeta:
    def test_fields(self):
        m = RunMeta(reach_id="R", alternative_id="FWOP",
                    sim_start=datetime(2030, 1, 1), lifecycle=3)
        assert m.reach_id == "R"
        assert m.alternative_id == "FWOP"
        assert m.lifecycle == 3
        assert m.sim_start == datetime(2030, 1, 1)


class TestParquetResultsSinkDirLayout:
    def test_creates_lc_dir(self):
        with tempfile.TemporaryDirectory() as root:
            ParquetResultsSink(root, "R1", "FWOP", lifecycle=0)
            assert os.path.isdir(os.path.join(root, "R1", "FWOP", "lc_0000"))

    def test_lifecycle_zero_padding(self):
        with tempfile.TemporaryDirectory() as root:
            ParquetResultsSink(root, "R1", "FWOP", lifecycle=42)
            assert os.path.isdir(os.path.join(root, "R1", "FWOP", "lc_0042"))

    def test_multiple_alternatives(self):
        with tempfile.TemporaryDirectory() as root:
            for alt in ("FWOP", "FWP-Alt1", "FWP-Alt2"):
                ParquetResultsSink(root, "R1", alt, lifecycle=0)
            for alt in ("FWOP", "FWP-Alt1", "FWP-Alt2"):
                assert os.path.isdir(os.path.join(root, "R1", alt, "lc_0000"))


class TestParquetResultsSinkOutputFiles:
    def test_expected_files_written(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            out = sink.out_dir
            for fname in ("profiles.parquet", "storm_hazard.parquet",
                          "profile_events.parquet", "segment_events.csv",
                          "run_metadata.json", "run_summary.txt"):
                assert os.path.isfile(os.path.join(out, fname)), f"missing {fname}"

    def test_profiles_parquet_no_zbe_column(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profiles.parquet"))
            assert "zbe" not in df.columns

    def test_profiles_parquet_columns(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profiles.parquet"))
            assert set(df.columns) == {"profile_id", "label", "t", "node_idx", "x", "zb"}

    def test_profiles_parquet_both_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profiles.parquet"))
            assert set(df["profile_id"].unique()) == {"p0", "p1"}

    def test_profile_events_columns(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profile_events.parquet"))
            assert set(df.columns) == {"profile_id", "label", "t", "storm_response_type"}

    def test_storm_hazard_no_jdry_column(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root, n_storms=1)
            df = pd.read_parquet(os.path.join(sink.out_dir, "storm_hazard.parquet"))
            assert "jdry" not in df.columns

    def test_profile_events_init_label(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profile_events.parquet"))
            assert "INIT" in df["label"].values

    def test_profile_events_poststorm_label(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_parquet(os.path.join(sink.out_dir, "profile_events.parquet"))
            assert "PostStorm" in df["label"].values

    def test_storm_hazard_both_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root, n_storms=2)
            df = pd.read_parquet(os.path.join(sink.out_dir, "storm_hazard.parquet"))
            assert set(df["profile_id"].unique()) == {"p0", "p1"}

    def test_storm_hazard_n_unique_storms(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root, n_storms=2)
            df = pd.read_parquet(os.path.join(sink.out_dir, "storm_hazard.parquet"))
            assert len(df["t_storm"].unique()) == 2

    def test_segment_events_csv_columns(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_csv(os.path.join(sink.out_dir, "segment_events.csv"))
            assert {"event_type", "profile_id", "t_start", "t_end",
                    "volume_m3", "volume_cy"}.issubset(set(df.columns))

    def test_segment_events_empty_when_no_nourishment(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root)
            df = pd.read_csv(os.path.join(sink.out_dir, "segment_events.csv"))
            assert len(df) == 0

    def test_run_metadata_json_fields(self):
        with tempfile.TemporaryDirectory() as root:
            sink = _run(root, n_storms=2)
            with open(os.path.join(sink.out_dir, "run_metadata.json")) as f:
                meta = json.load(f)
            assert meta["reach_id"] == "R1"
            assert meta["alternative_id"] == "FWOP"
            assert meta["lifecycle"] == 0
            assert meta["n_profiles"] == 2
            assert meta["n_storms"] == 2


class TestRecordNourishment:
    def test_populates_rows(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "R1", "FWOP", lifecycle=0)
            sink.record_nourishment("p0", 10.0, 15.0, 500.0, "FullNourishment")
            assert len(sink._nourishment_rows) == 1
            row = sink._nourishment_rows[0]
            assert row["event_type"] == "FullNourishment"
            assert row["volume_m3"] == pytest.approx(500.0)
            assert row["volume_cy"] == pytest.approx(500.0 * 1.30795)

    def test_null_sink_noop(self):
        sink = NullResultsSink()
        sink.record_nourishment("p0", 0.0, 1.0, 100.0, "FullNourishment")  # must not raise

    def test_segment_events_csv_populated(self):
        with tempfile.TemporaryDirectory() as root:
            sink = ParquetResultsSink(root, "R1", "FWOP", lifecycle=0)
            sink.record_nourishment("p0", 10.0, 15.0, 500.0, "FullNourishment")
            sink.record_nourishment("p1", 15.0, 20.0, 300.0, "PartialNourishment")

            from framework.results import RunMeta
            meta = RunMeta(reach_id="R1", alternative_id="FWOP",
                           sim_start=SIM_START, lifecycle=0)
            profiles = []
            sink.flush(profiles, meta)

            df = pd.read_csv(os.path.join(sink.out_dir, "segment_events.csv"))
            assert len(df) == 2
            assert set(df["event_type"].unique()) == {"FullNourishment", "PartialNourishment"}
