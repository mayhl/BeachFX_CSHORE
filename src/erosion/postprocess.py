"""Post-processing: establish the per-profile common grid and register outputs onto it.

A DISTINCT pass over a flushed lifecycle output dir (the locked Phase-A decision):
the run logs raw native-extent results; this step reads them back and produces the
global-grid products.  Phase A (no mid-run growth) means every bed snapshot already
shares ``profile.x``, so establishing the common grid is (a) storing that axis once
(``grid.parquet``) and (b) the one genuine registration — pulling hydro (written on
CSHORE's native ``result.x``, wet-zone-only) onto the common bed axis (``hydro.parquet``).

Reprocessable: rerun against the same dir with no CSHORE. The ``georef`` column is a
seam for the GeoParquet hook (null until wired).

Also hosts the offline summary helpers over ``profile_metrics.parquet``:
:func:`profile_extremes` (per-profile temporal envelope + net INIT->final change)
and :func:`storm_deltas` (per-storm pre/post change, INUNDATION carried through).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .results import read_parquet_footer, write_parquet_with_footer

log = logging.getLogger(__name__)


def establish_common_grid(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Per-profile common x-vector = the maximal-extent snapshot grid.

    Under append-only landward growth the largest snapshot's x is a superset of every
    other's, so it is the union axis; in Phase-A (no growth) every snapshot shares one
    grid and this just picks it.  Returns rows ``(profile_id, node_idx, x)``; ``lon``/``lat``
    are added later by ``apply_georef`` when the profile carries a georeference.
    """
    out = []
    for pid, g in profiles_df.groupby("profile_id", sort=False):
        # each (label, t) is one snapshot; take the one with the most nodes
        lbl, t = g.groupby(["label", "t"]).size().idxmax()
        snap = g[(g["label"] == lbl) & (g["t"] == t)].sort_values("node_idx")
        for i, xi in enumerate(snap["x"].to_numpy()):
            out.append({"profile_id": pid, "node_idx": i, "x": float(xi)})
    return pd.DataFrame(out, columns=["profile_id", "node_idx", "x"])


def register_hydro(hazard_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate each (profile, storm) hydro field onto the profile's common grid.

    Hydro (``mwl``/``Hs``) is written over CSHORE's wet zone only (NaN landward of
    ``JR``); registering onto the full common grid leaves those landward nodes NaN —
    the correct "no hydro here" signal.  ``runup_m`` is a per-storm scalar, carried through.

    Output is indexed by ``node_idx`` and carries NO ``x`` — join to ``grid.parquet`` on
    ``(profile_id, node_idx)`` to recover position (the common-grid normalization payoff).
    """
    gx = {
        pid: grp.sort_values("node_idx")["x"].to_numpy()
        for pid, grp in grid_df.groupby("profile_id")
    }
    out = []
    for (pid, ts), h in hazard_df.groupby(["profile_id", "t_storm"]):
        h = h.sort_values("node_idx")
        src_x = h["x"].to_numpy()
        cx = gx[pid]
        runup = float(h["runup_m"].iloc[0])
        reg = {}
        for field in ("mwl", "Hs"):
            fp = h[field].to_numpy()
            valid = ~np.isnan(fp)
            if valid.any():
                # left = seaward edge value; right = NaN (no hydro landward of the wet zone)
                reg[field] = np.interp(cx, src_x[valid], fp[valid], left=fp[valid][0], right=np.nan)
            else:
                reg[field] = np.full(len(cx), np.nan)
        for i in range(len(cx)):
            out.append(
                {
                    "profile_id": pid,
                    "t_storm": float(ts),
                    "node_idx": i,
                    "mwl": float(reg["mwl"][i]),
                    "Hs": float(reg["Hs"][i]),
                    "runup_m": runup,
                }
            )
    cols = ["profile_id", "t_storm", "node_idx", "mwl", "Hs", "runup_m"]
    return pd.DataFrame(out, columns=cols)


def transect_lonlat(
    x: np.ndarray, origin_lon: float, origin_lat: float, azimuth_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project cross-shore offsets ``x`` (m, increasing landward) along a transect of compass
    bearing ``azimuth_deg`` from its ``origin`` (lon/lat) to per-node ``(lon, lat)``.
    Equirectangular local approximation — adequate at transect scale."""
    az = np.radians(azimuth_deg)
    lat = origin_lat + (x * np.cos(az)) / 111320.0
    lon = origin_lon + (x * np.sin(az)) / (111320.0 * np.cos(np.radians(origin_lat)))
    return lon, lat


def apply_georef(grid_df: pd.DataFrame, georef_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``lon``/``lat`` to the common grid for georeferenced profiles (the GeoParquet
    hook).  Non-georeferenced profiles keep NaN coordinates.  Returns a new frame."""
    gr = georef_df.set_index("profile_id")
    lon = np.full(len(grid_df), np.nan)
    lat = np.full(len(grid_df), np.nan)
    for pid, g in grid_df.groupby("profile_id"):
        if pid not in gr.index:
            continue
        r = gr.loc[pid]
        lo, la = transect_lonlat(g["x"].to_numpy(), r.origin_lon, r.origin_lat, r.azimuth_deg)
        lon[g.index], lat[g.index] = lo, la
    out = grid_df.copy()
    out["lon"], out["lat"] = lon, lat
    return out


def reach_polygon_wkt(grid_geo_df: pd.DataFrame) -> str | None:
    """Coverage-only bounding-envelope polygon (WKT) over all georeferenced nodes — the
    reach's plan-view footprint for GIS overlay, no data attached.  ``None`` when no node
    has coordinates."""
    lon, lat = grid_geo_df["lon"].to_numpy(), grid_geo_df["lat"].to_numpy()
    m = np.isfinite(lon) & np.isfinite(lat)
    if not m.any():
        return None
    lo0, lo1 = float(lon[m].min()), float(lon[m].max())
    la0, la1 = float(lat[m].min()), float(lat[m].max())
    return f"POLYGON (({lo0} {la0}, {lo1} {la0}, {lo1} {la1}, {lo0} {la1}, {lo0} {la0}))"


def check_edge_proximity(
    metrics_df: pd.DataFrame, grid_df: pd.DataFrame, buffer_m: float = 25.0
) -> list[tuple[str, float]]:
    """Flag profiles whose shoreline migrated within ``buffer_m`` of the landward grid edge.

    The fixed grid is adequate only while the active zone stays clear of its landward end;
    once the shoreline (CSHORE-frame, x increasing landward) closes on ``x[-1]`` the beach
    risks running off the domain and being constant-extrapolated.  Warn-only — the remedy is
    a longer initial grid and re-run (mid-run growth is a deferred fallback).  Returns
    ``(profile_id, margin_m)`` for each flagged profile.
    """
    flagged = []
    for pid, g in grid_df.groupby("profile_id"):
        x_landward = float(g["x"].max())
        m = metrics_df[metrics_df["profile_id"] == pid]
        if m.empty or m["shoreline_x"].isna().all():
            continue  # no shoreline read (e.g. no geometry) -> can't assess
        margin = x_landward - float(m["shoreline_x"].max())
        if margin < buffer_m:
            flagged.append((pid, margin))
    return flagged


def postprocess_lifecycle(out_dir: str) -> None:
    """Establish the common grid + register hydro onto it for one flushed lifecycle dir.

    Reads ``profiles.parquet`` (bed store) and, when present, ``storm_hazard.parquet``
    (native-grid hydro); writes ``grid.parquet`` (common axis + georef seam) and
    ``hydro.parquet`` (hydro on the common grid).  No-op on the axis if profiles are absent.
    """
    profiles_path = os.path.join(out_dir, "profiles.parquet")
    if not os.path.exists(profiles_path):
        return
    # Propagate the run's config-in-footer from the source so derived products stay
    # self-describing / reproducible from the file alone.
    footer = read_parquet_footer(profiles_path)
    grid = establish_common_grid(pd.read_parquet(profiles_path))

    # GeoParquet hook: when profiles carry a georeference, project the grid to lon/lat and
    # emit the coverage-only reach polygon.  Absent georef -> local coordinates only.
    georef_path = os.path.join(out_dir, "profile_georef.parquet")
    if os.path.exists(georef_path):
        grid = apply_georef(grid, pd.read_parquet(georef_path))
        poly = reach_polygon_wkt(grid)
        if poly is not None:
            reach = pd.DataFrame([{"reach_id": footer.get("beachfx_reach_id"), "geometry": poly}])
            write_parquet_with_footer(reach, os.path.join(out_dir, "reach.parquet"), footer)
    write_parquet_with_footer(grid, os.path.join(out_dir, "grid.parquet"), footer)

    hazard_path = os.path.join(out_dir, "storm_hazard.parquet")
    if os.path.exists(hazard_path):
        hydro = register_hydro(pd.read_parquet(hazard_path), grid)
        write_parquet_with_footer(hydro, os.path.join(out_dir, "hydro.parquet"), footer)

    metrics_path = os.path.join(out_dir, "profile_metrics.parquet")
    if os.path.exists(metrics_path):
        for pid, margin in check_edge_proximity(pd.read_parquet(metrics_path), grid):
            log.warning(
                "edge-proximity: profile %s shoreline within %.1f m of the landward grid edge; "
                "active zone risks extrapolation — lengthen the grid and re-run",
                pid,
                margin,
            )


# ---------------------------------------------------------------------------
# Derived summary metrics (offline, over a finished run's metrics parquet)
# ---------------------------------------------------------------------------

# Height/width/volume metrics whose per-profile temporal envelope is meaningful.
_ENVELOPE_COLS = [
    "dune_crest_elevation",
    "dune_front_relief",  # dune height above the berm
    "dune_back_relief",  # dune height above the upland
    "berm_elevation",  # berm height
    "berm_width",
    "dune_width",
    "volume_above_datum",
]
# Metrics whose net INIT→final change tells an erosion/accretion story.
_NET_COLS = [
    "dune_crest_elevation",
    "berm_elevation",
    "berm_width",
    "volume_above_datum",
]


def load_metrics(path: str) -> pd.DataFrame:
    """Read a run's ``profile_metrics.parquet`` into a DataFrame."""
    return pd.read_parquet(path)


def profile_extremes(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-profile temporal envelope over the full snapshot series.

    One row per ``profile_id`` with ``<col>_min`` / ``_max`` / ``_range`` for the
    height/width/volume metrics, the net change ``<col>_net`` (last minus first,
    ordered by ``t``) for the erosion-story metrics, and ``scarp_height_max``.  NaN
    snapshots (e.g. HIGH_UPLAND with no dune) are skipped by min/max/first/last, so
    a profile that carries a dune only part of the run still reports its dune
    envelope over the snapshots where one exists."""
    if metrics.empty:
        return pd.DataFrame()
    g = metrics.sort_values("t").groupby("profile_id", sort=True)
    out: dict[str, pd.Series] = {}
    for col in _ENVELOPE_COLS:
        lo, hi = g[col].min(), g[col].max()
        out[f"{col}_min"] = lo
        out[f"{col}_max"] = hi
        out[f"{col}_range"] = hi - lo
    for col in _NET_COLS:
        out[f"{col}_net"] = g[col].last() - g[col].first()
    out["scarp_height_max"] = g["scarp_height"].max()
    return pd.DataFrame(out).reset_index()


def storm_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-storm change for each profile: PostStorm (or INUNDATION) minus PreStorm
    at the same storm time.

    One row per (``profile_id``, ``t``) storm, using the *drop* convention
    (pre − post) so erosion reads positive:

    * ``crest_drop``      — dune-crest lowering,
    * ``berm_elev_drop``  — berm-height lowering,
    * ``berm_width_loss`` — berm narrowing,
    * ``eroded_volume``   — subaerial volume loss (aligned pre/post, shared grid),
    * ``dune_scarp`` / ``scarp_height`` — from the post snapshot,
    * ``inundation`` — CSHORE failed and the profile passed through unchanged.
    """
    pre = metrics[metrics["label"] == "PreStorm"]
    post = metrics[metrics["label"].isin(["PostStorm", "INUNDATION"])]
    if pre.empty or post.empty:
        return pd.DataFrame()
    keys = ["profile_id", "t"]
    m = pre.merge(post, on=keys, suffixes=("_pre", "_post"))
    out = pd.DataFrame(
        {
            "profile_id": m["profile_id"],
            "t": m["t"],
            "crest_drop": m["dune_crest_elevation_pre"] - m["dune_crest_elevation_post"],
            "berm_elev_drop": m["berm_elevation_pre"] - m["berm_elevation_post"],
            "berm_width_loss": m["berm_width_pre"] - m["berm_width_post"],
            "eroded_volume": m["volume_above_datum_pre"] - m["volume_above_datum_post"],
            "dune_scarp": m["dune_scarp_post"],
            "scarp_height": m["scarp_height_post"],
            "inundation": m["label_post"] == "INUNDATION",
        }
    )
    return out.sort_values(keys).reset_index(drop=True)
