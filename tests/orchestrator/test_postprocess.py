"""Post-processing tests: common-grid establishment + hydro registration.

Runs a lifecycle through ``ScriptedRunner``, flushes, then exercises the distinct
``postprocess_lifecycle`` pass — physics-independent.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from erosion.config import ReachConfig
from erosion.postprocess import (
    check_edge_proximity,
    postprocess_lifecycle,
    register_hydro,
    transect_lonlat,
)
from erosion.profile import Georef
from erosion.results import ParquetResultsSink, read_parquet_footer
from tests.builders import ncfg, run, storms_at, template_profile
from tests.doubles import SEVERE, ScriptedRunner


def _flush_lifecycle(root):
    cfg = ReachConfig(
        storm={"z_berm": 2.0},
        nourishment=ncfg(volume_trigger=30.0, production_rate=100.0, assessor="volume"),
    )
    sink = ParquetResultsSink(root, "reach1", "FWP", lifecycle=0, config=cfg)
    profiles, _ = run(
        [template_profile("p0")],
        storms_df=storms_at([20]),
        runner=ScriptedRunner(SEVERE),
        cfg=cfg,
        sink=sink,
    )
    return sink.out_dir, profiles


def test_grid_matches_profile_x():
    with tempfile.TemporaryDirectory() as root:
        out_dir, profiles = _flush_lifecycle(root)
        postprocess_lifecycle(out_dir)
        grid = pd.read_parquet(os.path.join(out_dir, "grid.parquet"))
        assert list(grid.columns) == [
            "profile_id",
            "node_idx",
            "x",
        ]  # no georef -> local coords only
        gx = grid[grid.profile_id == "p0"].sort_values("node_idx")["x"].to_numpy()
        np.testing.assert_allclose(gx, profiles[0].x)
        assert not os.path.exists(
            os.path.join(out_dir, "reach.parquet")
        )  # no polygon without coords


def test_hydro_registered_onto_common_grid():
    with tempfile.TemporaryDirectory() as root:
        out_dir, _profiles = _flush_lifecycle(root)
        postprocess_lifecycle(out_dir)
        grid = pd.read_parquet(os.path.join(out_dir, "grid.parquet"))
        hydro = pd.read_parquet(os.path.join(out_dir, "hydro.parquet"))
        assert "x" not in hydro.columns  # x lives only in grid.parquet (dimension/fact split)
        n_grid = int((grid.profile_id == "p0").sum())
        for _ts, h in hydro.groupby("t_storm"):
            assert len(h) == n_grid  # hydro spans the full common grid
            assert list(h.sort_values("node_idx")["node_idx"]) == list(range(n_grid))
        # node_idx joins hydro back to the grid axis
        joined = hydro.merge(grid, on=["profile_id", "node_idx"])
        assert len(joined) == len(hydro) and "x" in joined.columns


def test_postprocess_propagates_config_footer():
    with tempfile.TemporaryDirectory() as root:
        out_dir, _ = _flush_lifecycle(root)
        src = read_parquet_footer(os.path.join(out_dir, "profiles.parquet"))
        postprocess_lifecycle(out_dir)
        for fname in ("grid.parquet", "hydro.parquet"):
            assert read_parquet_footer(os.path.join(out_dir, fname)) == src


def test_postprocess_idempotent():
    with tempfile.TemporaryDirectory() as root:
        out_dir, _ = _flush_lifecycle(root)
        postprocess_lifecycle(out_dir)
        g1 = pd.read_parquet(os.path.join(out_dir, "grid.parquet"))
        postprocess_lifecycle(out_dir)  # reprocess, no CSHORE
        g2 = pd.read_parquet(os.path.join(out_dir, "grid.parquet"))
        pd.testing.assert_frame_equal(g1, g2)


def test_edge_proximity_flags_shoreline_near_landward_edge():
    grid = pd.DataFrame(
        {"profile_id": ["a"] * 3 + ["b"] * 3, "node_idx": [0, 1, 2, 0, 1, 2], "x": [0, 50, 100] * 2}
    )
    # a: shoreline retreated to x=90 -> 10 m from the x=100 edge (< 25 buffer) -> flagged
    # b: shoreline at x=40 -> 60 m margin -> clear
    metrics = pd.DataFrame(
        {"profile_id": ["a", "a", "b", "b"], "shoreline_x": [30.0, 90.0, 20.0, 40.0]}
    )
    flagged = check_edge_proximity(metrics, grid, buffer_m=25.0)
    assert [pid for pid, _ in flagged] == ["a"]
    assert flagged[0][1] == 10.0


def test_edge_proximity_skips_profiles_without_shoreline():
    grid = pd.DataFrame({"profile_id": ["a"] * 2, "node_idx": [0, 1], "x": [0, 100]})
    metrics = pd.DataFrame({"profile_id": ["a", "a"], "shoreline_x": [np.nan, np.nan]})
    assert check_edge_proximity(metrics, grid) == []


def test_transect_lonlat_projects_eastward():
    # azimuth 90 (due east) -> lat constant, lon increases with x
    lon, lat = transect_lonlat(
        np.array([0.0, 100.0]), origin_lon=-78.0, origin_lat=34.0, azimuth_deg=90.0
    )
    np.testing.assert_allclose(lat, [34.0, 34.0], atol=1e-9)
    assert lon[1] > lon[0]
    assert lon[0] == -78.0


def test_georef_produces_lonlat_grid_and_reach_polygon():
    with tempfile.TemporaryDirectory() as root:
        cfg = ReachConfig(
            nourishment=ncfg(volume_trigger=30.0, production_rate=100.0, assessor="volume")
        )
        p = template_profile("p0")
        p.georef = Georef(origin_lon=-78.0, origin_lat=34.0, azimuth_deg=90.0)
        sink = ParquetResultsSink(root, "reach1", "FWP", lifecycle=0, config=cfg)
        run([p], storms_df=storms_at([20]), runner=ScriptedRunner(SEVERE), cfg=cfg, sink=sink)
        postprocess_lifecycle(sink.out_dir)
        grid = pd.read_parquet(os.path.join(sink.out_dir, "grid.parquet"))
        assert {"lon", "lat"} <= set(grid.columns)
        assert grid["lon"].notna().all()
        reach = pd.read_parquet(os.path.join(sink.out_dir, "reach.parquet"))
        assert len(reach) == 1
        assert reach["geometry"].iloc[0].startswith("POLYGON")
        assert "reach_id" in reach.columns


def test_register_hydro_nan_landward_of_wet_zone():
    # Hydro valid only over the seaward wet zone (first 2 nodes); registering onto the
    # full common grid must leave landward nodes NaN (no hydro there), runup carried.
    grid = pd.DataFrame(
        {
            "profile_id": ["p"] * 4,
            "node_idx": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "georef": None,
        }
    )
    hazard = pd.DataFrame(
        {
            "profile_id": ["p"] * 4,
            "t_storm": [10.0] * 4,
            "node_idx": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "mwl": [0.5, 0.4, np.nan, np.nan],
            "Hs": [1.0, 0.8, np.nan, np.nan],
            "runup_m": [1.2] * 4,
        }
    )
    reg = register_hydro(hazard, grid).sort_values("node_idx")
    assert not np.isnan(reg["mwl"].iloc[0])  # seaward node interpolated
    assert np.isnan(reg["mwl"].iloc[3])  # landward of wet zone -> NaN
    assert (reg["runup_m"] == 1.2).all()
