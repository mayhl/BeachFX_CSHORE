"""Unit tests for ParquetResultsSink and RunMeta."""

import json
import os
from datetime import datetime

import pandas as pd
import pytest

from erosion.results import NullResultsSink, ParquetResultsSink, RunMeta
from erosion.runner import MockCSHORERunner
from erosion.types import DecisionKind
from tests.builders import SIM_START, profile, run


def _make_sink(out_root: str, n_storms: int = 2) -> ParquetResultsSink:
    sink = ParquetResultsSink(out_root, "R1", "FWOP", lifecycle=0)
    # Schema tests only -- the mock scoop's bed shape is immaterial, named explicitly
    run(
        [profile("p0", n=50), profile("p1", n=50)],
        n_storms,
        sink=sink,
        reach_id="R1",
        alternative_id="FWOP",
        runner=MockCSHORERunner(),
    )
    return sink


class TestRunMeta:
    def test_fields(self):
        m = RunMeta(
            reach_id="R", alternative_id="FWOP", sim_start=datetime(2030, 1, 1), lifecycle=3
        )
        assert m.reach_id == "R"
        assert m.alternative_id == "FWOP"
        assert m.lifecycle == 3
        assert m.sim_start == datetime(2030, 1, 1)


class TestParquetResultsSinkDirLayout:
    def test_creates_lc_dir(self, tmp_path):
        ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=0)
        assert (tmp_path / "R1" / "FWOP" / "lc_0000").is_dir()

    def test_lifecycle_zero_padding(self, tmp_path):
        ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=42)
        assert (tmp_path / "R1" / "FWOP" / "lc_0042").is_dir()

    def test_multiple_alternatives(self, tmp_path):
        for alt in ("FWOP", "FWP-Alt1", "FWP-Alt2"):
            ParquetResultsSink(str(tmp_path), "R1", alt, lifecycle=0)
        for alt in ("FWOP", "FWP-Alt1", "FWP-Alt2"):
            assert (tmp_path / "R1" / alt / "lc_0000").is_dir()


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> str:
    """Run one 2-storm lifecycle once; all output-file tests read its directory."""
    root = tmp_path_factory.mktemp("results")
    return _make_sink(str(root), n_storms=2).out_dir


class TestParquetResultsSinkOutputFiles:
    def test_expected_files_written(self, out_dir):
        for fname in (
            "profiles.parquet",
            "storm_hazard.parquet",
            "snapshots.parquet",
            "decisions.parquet",
            "placements.csv",
            "run_metadata.json",
            "run_summary.txt",
        ):
            assert os.path.isfile(os.path.join(out_dir, fname)), f"missing {fname}"

    @pytest.mark.parametrize(
        "fname, expected",
        [
            ("profiles.parquet", {"profile_id", "label", "t", "node_idx", "x", "zb"}),
            ("snapshots.parquet", {"profile_id", "label", "t", "storm_response_type"}),
        ],
    )
    def test_exact_columns(self, out_dir, fname, expected):
        df = pd.read_parquet(os.path.join(out_dir, fname))
        assert set(df.columns) == expected

    @pytest.mark.parametrize(
        "fname, absent",
        [
            ("profiles.parquet", "zbe"),
            ("storm_hazard.parquet", "jdry"),
        ],
    )
    def test_absent_columns(self, out_dir, fname, absent):
        df = pd.read_parquet(os.path.join(out_dir, fname))
        assert absent not in df.columns

    @pytest.mark.parametrize("fname", ["profiles.parquet", "storm_hazard.parquet"])
    def test_both_profiles_present(self, out_dir, fname):
        df = pd.read_parquet(os.path.join(out_dir, fname))
        assert set(df["profile_id"].unique()) == {"p0", "p1"}

    @pytest.mark.parametrize("label", ["INIT", "PostStorm"])
    def test_snapshots_label_present(self, out_dir, label):
        df = pd.read_parquet(os.path.join(out_dir, "snapshots.parquet"))
        assert label in df["label"].values

    def test_storm_hazard_n_unique_storms(self, out_dir):
        df = pd.read_parquet(os.path.join(out_dir, "storm_hazard.parquet"))
        assert len(df["t_storm"].unique()) == 2

    def test_placements_csv_columns(self, out_dir):
        df = pd.read_csv(os.path.join(out_dir, "placements.csv"))
        assert {"event_type", "profile_id", "t_start", "t_end", "placed_m3", "placed_cy"}.issubset(
            set(df.columns)
        )

    def test_placements_empty_when_no_nourishment(self, out_dir):
        df = pd.read_csv(os.path.join(out_dir, "placements.csv"))
        assert len(df) == 0

    def test_run_metadata_json_fields(self, out_dir):
        with open(os.path.join(out_dir, "run_metadata.json")) as f:
            meta = json.load(f)
        assert meta["reach_id"] == "R1"
        assert meta["alternative_id"] == "FWOP"
        assert meta["lifecycle"] == 0
        assert meta["n_profiles"] == 2
        assert meta["n_storms"] == 2


class TestRecordNourishment:
    def test_populates_rows(self, tmp_path):
        sink = ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=0)
        sink.record_nourishment("p0", 10.0, 15.0, 500.0, "FullNourishment")
        assert len(sink._nourishment_rows) == 1
        row = sink._nourishment_rows[0]
        assert row["event_type"] == "FullNourishment"
        assert row["placed_m3"] == pytest.approx(500.0)
        assert row["placed_cy"] == pytest.approx(500.0 * 1.30795)

    def test_null_sink_noop(self):
        sink = NullResultsSink()
        sink.record_nourishment("p0", 0.0, 1.0, 100.0, "FullNourishment")  # must not raise

    def test_placements_csv_populated(self, tmp_path):
        sink = ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=0)
        sink.record_nourishment("p0", 10.0, 15.0, 500.0, "FullNourishment")
        sink.record_nourishment("p1", 15.0, 20.0, 300.0, "PartialNourishment")
        meta = RunMeta(reach_id="R1", alternative_id="FWOP", sim_start=SIM_START, lifecycle=0)
        sink.flush([], meta)
        df = pd.read_csv(os.path.join(sink.out_dir, "placements.csv"))
        assert len(df) == 2
        assert set(df["event_type"].unique()) == {"FullNourishment", "PartialNourishment"}


class TestRecordDecision:
    """The reach-scope audit trail: why the crew mobilized, deferred, or was cut short."""

    def _flushed(self, tmp_path) -> pd.DataFrame:
        sink = ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=0)
        sink.record_decision(DecisionKind.NOURISH_TRIGGER, 20.5, deficit_m3=155.0, trigger=30.0)
        # Logged at the time it concerns (the next storm), not the time it was taken —
        # which is why the file carries an emission sequence as well as a time.
        sink.record_decision(DecisionKind.INTERRUPT, 34.0, profile_id="p0", placed_fraction=0.33)
        sink.flush([], RunMeta("R1", "FWOP", SIM_START, 0))
        return pd.read_parquet(os.path.join(sink.out_dir, "decisions.parquet"))

    def test_rows_carry_kind_time_and_emission_order(self, tmp_path):
        df = self._flushed(tmp_path)
        assert list(df["kind"]) == ["NOURISH_TRIGGER", "INTERRUPT"]
        assert list(df["decision_seq"]) == [0, 1]
        assert list(df["t"]) == pytest.approx([20.5, 34.0])

    def test_payload_keys_become_columns_and_absent_ones_are_null(self, tmp_path):
        df = self._flushed(tmp_path)
        assert df.loc[0, "deficit_m3"] == pytest.approx(155.0)
        assert df.loc[1, "placed_fraction"] == pytest.approx(0.33)
        assert pd.isna(df.loc[1, "deficit_m3"])  # INTERRUPT carries no deficit
        assert pd.isna(df.loc[0, "profile_id"])  # a reach-scope decision names no profile

    def test_written_empty_when_nothing_was_decided(self, tmp_path):
        sink = ParquetResultsSink(str(tmp_path), "R1", "FWOP", lifecycle=0)
        sink.flush([], RunMeta("R1", "FWOP", SIM_START, 0))
        df = pd.read_parquet(os.path.join(sink.out_dir, "decisions.parquet"))
        assert len(df) == 0
        assert set(df.columns) == {"decision_seq", "kind", "t", "profile_id"}

    def test_null_sink_noop(self):
        NullResultsSink().record_decision(DecisionKind.NOURISH_SKIP, 1.0)  # must not raise
