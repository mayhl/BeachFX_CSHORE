"""Post-hoc derived metrics over a run's ``profile_metrics.parquet``.

The pipeline writes one 0-D ``ProfileMetrics`` row per (profile, snapshot); these
helpers aggregate that table *after the fact* without touching the run pipeline:

* :func:`profile_extremes` — per-profile temporal envelope (min / max / range of
  the dune-crest and berm heights/widths and subaerial volume over the whole
  snapshot series), the net INIT→final change, and the peak scarp.
* :func:`storm_deltas` — per (profile, storm) change across each storm, pairing the
  ``PreStorm`` snapshot with its ``PostStorm`` (or ``INUNDATION``, a CSHORE failure
  that carries the profile through unchanged).  ``eroded_volume`` is the aligned
  pre−post subaerial volume loss; snapshots share one fixed x-grid, so the
  difference is well defined without re-registration.

Both take the metrics DataFrame (or a path via :func:`load_metrics`) and return a
tidy DataFrame — one row per profile / per storm — for analysis code.
"""

from __future__ import annotations

import pandas as pd

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
