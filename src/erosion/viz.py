"""Visualization utilities for BeachFX-CSHORE pipeline output.

Public API
----------
load_runs(output_root, reach_id, lifecycle=0) -> dict[str, RunResult]
plot_profile_evolution(run, profile_id, ...)  -> Figure
plot_metrics(runs, profile_id, ...)           -> Figure
make_profile_video(run, profile_id, out_path) -> None
plot_idealized_fit(x, zb, design_berm_elevation, ...) -> Figure   # fit debug overlay
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Pipeline output for one (reach, alternative, lifecycle) directory."""
    out_dir:   str
    reach_id:  str
    alt_id:    str
    lifecycle: int

    _profiles: pd.DataFrame | None = field(default=None, repr=False)
    _metrics:  pd.DataFrame | None = field(default=None, repr=False)
    _hazard:   pd.DataFrame | None = field(default=None, repr=False)
    _events:   pd.DataFrame | None = field(default=None, repr=False)

    def _load(self, name: str) -> pd.DataFrame | None:
        path = os.path.join(self.out_dir, name)
        return pd.read_parquet(path) if os.path.exists(path) else None

    @property
    def profiles(self) -> pd.DataFrame:
        if self._profiles is None:
            self._profiles = self._load("profiles.parquet")
        return self._profiles

    @property
    def metrics(self) -> pd.DataFrame | None:
        if self._metrics is None:
            self._metrics = self._load("profile_metrics.parquet")
        return self._metrics

    @property
    def hazard(self) -> pd.DataFrame | None:
        if self._hazard is None:
            self._hazard = self._load("storm_hazard.parquet")
        return self._hazard

    @property
    def events(self) -> pd.DataFrame | None:
        if self._events is None:
            self._events = self._load("profile_events.parquet")
        return self._events

    @property
    def profile_ids(self) -> list[str]:
        return sorted(self.profiles["profile_id"].unique())

    @property
    def storm_times(self) -> np.ndarray:
        """Unique storm times from storm_hazard.parquet."""
        if self.hazard is None:
            return np.array([])
        return np.sort(self.hazard["t_storm"].unique())

    @property
    def erosion_times(self) -> np.ndarray:
        """Unique erosion tick times from profiles.parquet (Periodic snapshots)."""
        if self.profiles is None:
            return np.array([])
        ev = self.profiles
        times = ev.loc[ev["label"] == "Periodic", "t"].unique()
        return np.sort(times)


def load_runs(
    output_root: str,
    reach_id: str,
    lifecycle: int = 0,
) -> dict[str, RunResult]:
    """Discover all alternatives under reach_id; return {alt_id: RunResult}."""
    reach_dir = os.path.join(output_root, reach_id)
    runs: dict[str, RunResult] = {}
    if not os.path.isdir(reach_dir):
        raise FileNotFoundError(f"Reach directory not found: {reach_dir}")
    for alt_id in sorted(os.listdir(reach_dir)):
        lc_dir = os.path.join(reach_dir, alt_id, f"lc_{lifecycle:04d}")
        if os.path.isdir(lc_dir) and os.path.exists(os.path.join(lc_dir, "profiles.parquet")):
            runs[alt_id] = RunResult(
                out_dir=lc_dir,
                reach_id=reach_id,
                alt_id=alt_id,
                lifecycle=lifecycle,
            )
    return runs


# ---------------------------------------------------------------------------
# Colours and style helpers
# ---------------------------------------------------------------------------

_ALT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

def _alt_color(runs: dict[str, RunResult], alt_id: str) -> str:
    keys = list(runs.keys())
    return _ALT_COLORS[keys.index(alt_id) % len(_ALT_COLORS)]

_LABEL_STYLE: dict[str, dict] = {
    "INIT":         {"color": "black",   "lw": 2.0,  "ls": "-",  "zorder": 5},
    "PreStorm":     {"color": "#888888", "lw": 0.8,  "ls": "--", "zorder": 2},
    "PostStorm":    {"color": "#d62728", "lw": 1.0,  "ls": "-",  "zorder": 3},
    "REC":          {"color": "#2ca02c", "lw": 0.8,  "ls": ":",  "zorder": 2},
    "EndIteration": {"color": "#1f77b4", "lw": 2.0,  "ls": "-",  "zorder": 5},
}


def _get_profile_snapshot(
    df: pd.DataFrame, profile_id: str, label: str, t: float
) -> tuple[np.ndarray, np.ndarray] | None:
    sub = df[(df["profile_id"] == profile_id) & (df["label"] == label) & (np.isclose(df["t"], t))]
    if sub.empty:
        return None
    sub = sub.sort_values("node_idx")
    return sub["x"].to_numpy(), sub["zb"].to_numpy()


# ---------------------------------------------------------------------------
# Plot 1 — profile cross-section evolution
# ---------------------------------------------------------------------------

def _beach_zoom(sub: pd.DataFrame, margin_m: float = 150.0) -> tuple[float, float]:
    """Auto-detect beach zone: from (shoreline - margin) to profile landward end."""
    # Use the snapshot with the most seaward shoreline as the left bound
    sub_sorted = sub.sort_values("node_idx")
    # shoreline = last node where zb transitions from below 0 to above 0 (wet→dry)
    all_x = []
    for (label, t), grp in sub_sorted.groupby(["label", "t"], sort=False):
        x = grp["x"].to_numpy()
        z = grp["zb"].to_numpy()
        cross = np.where(np.diff((z > 0).astype(int)) > 0)[0]
        if len(cross):
            all_x.append(x[cross[-1]])
    x_shore_min = min(all_x) if all_x else sub["x"].min()
    x_max = sub["x"].max()
    return max(0.0, x_shore_min - margin_m), x_max


def plot_profile_evolution(
    run: RunResult,
    profile_id: str,
    labels: Sequence[str] | None = None,
    zoom_x: tuple[float, float] | None = None,
    zoom_beach: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Cross-section snapshots for one profile, all timesteps coloured by time.

    INIT is excluded — it lives on the raw loaded grid (different x spacing)
    while all post-CSHORE snapshots share the CSHORE 2 m grid.  All included
    snapshots are drawn with a plasma colormap (early=yellow, late=purple).
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    df = run.profiles
    sub = df[df["profile_id"] == profile_id].copy()

    # Exclude INIT — incompatible x grid; include everything else
    all_labels = labels or ["PreStorm", "PostStorm", "REC", "EndIteration", "Periodic"]
    snaps = (
        sub[sub["label"].isin(all_labels)][["label", "t"]]
        .drop_duplicates()
        .sort_values("t")
        .reset_index(drop=True)
    )

    if snaps.empty:
        return fig or plt.figure()

    # Determine the canonical x_max from CSHORE-expanded snapshots.
    # The first PreStorm lives on the raw loaded grid (shorter x range) and
    # must be excluded so all plotted profiles share the same coordinate system.
    snap_xmax = {
        (row["label"], row["t"]): sub[
            (sub["label"] == row["label"]) & np.isclose(sub["t"], row["t"])
        ]["x"].max()
        for _, row in snaps.iterrows()
    }
    ref_xmax = max(snap_xmax.values())
    snaps = snaps[
        snaps.apply(lambda r: snap_xmax[(r["label"], r["t"])] >= 0.8 * ref_xmax, axis=1)
    ].reset_index(drop=True)

    if snaps.empty:
        return fig or plt.figure()

    t_min, t_max = snaps["t"].min(), snaps["t"].max()
    cmap = cm.plasma_r
    norm = mcolors.Normalize(vmin=t_min, vmax=max(t_max, t_min + 1))

    for _, row in snaps.iterrows():
        xy = _get_profile_snapshot(sub, profile_id, row["label"], row["t"])
        if xy is None:
            continue
        color = cmap(norm(row["t"]))
        lw = 1.8 if row["label"] == "EndIteration" else 0.7
        alpha = 1.0 if row["label"] == "EndIteration" else 0.75
        ax.plot(xy[0], xy[1], color=color, lw=lw, alpha=alpha, zorder=2)

    # Colour bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time (days)", fraction=0.02, pad=0.01)

    # Zoom to beach zone using only post-CSHORE snapshots
    xlim = zoom_x or (_beach_zoom(sub[sub["label"].isin(all_labels)]) if zoom_beach
                      else (sub["x"].min(), sub["x"].max()))
    ax.set_xlim(xlim)

    ax.fill_between(xlim, -200, 0, color="#cce5ff", alpha=0.25, zorder=0)
    ax.axhline(0, color="#4a90d9", lw=0.7, alpha=0.7)

    z_in_view = sub[sub["label"].isin(all_labels)]
    z_in_view = z_in_view[(z_in_view["x"] >= xlim[0]) & (z_in_view["x"] <= xlim[1])]["zb"]
    y_lo = max(z_in_view.min() - 0.5, -5.0) if not z_in_view.empty else -5.0
    y_hi = z_in_view.max() + 0.5 if not z_in_view.empty else 5.0
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Cross-shore position (m, offshore→landward)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{run.reach_id} / {run.alt_id} — {profile_id} profile evolution")
    ax.grid(True, alpha=0.25)
    if fig is not None:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — metrics time series (comparison across alternatives)
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "berm_width":           "Berm width (m)",
    "dune_crest_elevation": "Dune crest elevation (m)",
    "shoreline_x":          "Shoreline position (m)",
    "volume_above_datum":   "Volume above datum (m²/m)",
}

# Causal order within each simulation step
_LABEL_ORDER = {"INIT": 0, "PreStorm": 1, "PostStorm": 2, "REC": 3,
                "EndIteration": 4, "Periodic": 5}

# Marker style per snapshot label
_LABEL_MARKER = {
    "PreStorm":  dict(marker="o", s=12, zorder=3, alpha=0.6),   # circle
    "PostStorm": dict(marker="v", s=25, zorder=5, alpha=0.9),   # triangle-down
    "REC":       dict(marker="^", s=12, zorder=3, alpha=0.6),   # triangle-up
    "Periodic":  dict(marker=".", s=8,  zorder=2, alpha=0.4),   # dot
}


def plot_metrics(
    runs: dict[str, RunResult],
    profile_id: str,
    metrics: Sequence[str] = (
        "berm_width", "dune_crest_elevation", "shoreline_x", "volume_above_datum"
    ),
    labels: Sequence[str] | None = None,
) -> Figure:
    """Time-series of 0-D morphology metrics; one line per alternative."""
    n = len(metrics)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for alt_id, run in runs.items():
        if run.metrics is None:
            continue
        df = run.metrics[run.metrics["profile_id"] == profile_id].copy()
        if labels is not None:
            df = df[df["label"].isin(labels)]
        # Sort in causal order: by t first, then by label position within each step
        df["_order"] = df["label"].map(_LABEL_ORDER).fillna(99)
        df = df.sort_values(["t", "_order"]).drop(columns="_order")
        color = _alt_color(runs, alt_id)

        for ax, metric in zip(flat_axes, metrics):
            if metric not in df.columns:
                continue
            # Single connected line through all snapshots in causal order
            ax.plot(df["t"], df[metric], color=color, lw=1.1, label=alt_id,
                    zorder=2, alpha=0.7)
            # Markers per label — first alt only adds label text to legend
            first_alt = list(runs.keys())[0] == alt_id
            for lbl, style in _LABEL_MARKER.items():
                sub = df[df["label"] == lbl]
                if sub.empty:
                    continue
                legend_lbl = lbl if first_alt else None
                ax.scatter(sub["t"], sub[metric], color=color,
                           label=legend_lbl, **style)


    for ax, metric in zip(flat_axes, metrics):
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.25)
        handles, lbls = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, l in zip(handles, lbls):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), fontsize=7,
                  ncol=2, loc="best", framealpha=0.85)

    for ax in flat_axes[n:]:
        ax.set_visible(False)

    reach_ids = list({r.reach_id for r in runs.values()})
    fig.suptitle(f"{', '.join(reach_ids)} — {profile_id} metrics", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3a — before/after event transition frames
# ---------------------------------------------------------------------------

_TRANSITION_STYLE = {
    "Storm":    dict(color="#d62728", label="PostStorm", fill="#ffcccc"),
    "Recovery": dict(color="#2ca02c", label="PreStorm",  fill="#ccf5cc"),
    "Erosion":  dict(color="#5b7aad", label="PreStorm",  fill="#d0ddf0"),
}


def _build_transitions(
    snaps: pd.DataFrame,
    include_recovery: bool = True,
    include_erosion: bool = True,
) -> list[dict]:
    """Build ordered list of (event_type, before_row, after_row) transition dicts.

    Pairs (in causal order per storm cycle):
      Erosion  : INIT → PreStorm_1  (pre-storm background erosion)
                 REC_t → PreStorm_{t_next}  (inter-storm background erosion)
      Storm    : PreStorm_t  → PostStorm_t
      Recovery : PostStorm_t → PreStorm_{t_next}
    """
    transitions: list[dict] = []
    pre_rows  = snaps[snaps["label"] == "PreStorm"].sort_values("t").reset_index(drop=True)
    post_rows = snaps[snaps["label"] == "PostStorm"].sort_values("t").reset_index(drop=True)
    rec_rows  = snaps[snaps["label"] == "REC"].sort_values("t").reset_index(drop=True)
    init_rows = snaps[snaps["label"] == "INIT"].sort_values("t").reset_index(drop=True)
    end_rows  = snaps[snaps["label"] == "EndIteration"].sort_values("t").reset_index(drop=True)

    # --- Erosion transitions ---
    if include_erosion and not pre_rows.empty:
        # INIT → first PreStorm (both on the raw-profile grid)
        if not init_rows.empty:
            transitions.append({"type": "Erosion",
                                 "before": init_rows.iloc[0], "after": pre_rows.iloc[0]})

        # REC_t → next PreStorm (both on CSHORE grid)
        for _, rec in rec_rows.iterrows():
            later_pre = pre_rows[pre_rows["t"] > rec["t"]]
            if not later_pre.empty:
                transitions.append({"type": "Erosion",
                                     "before": rec, "after": later_pre.iloc[0]})

    # --- Storm + Recovery transitions ---
    for i, pre in pre_rows.iterrows():
        post_match = post_rows[np.isclose(post_rows["t"], pre["t"])]
        if post_match.empty:
            continue
        post = post_match.iloc[0]
        transitions.append({"type": "Storm", "before": pre, "after": post})

        if include_recovery and i + 1 < len(pre_rows):
            next_pre = pre_rows.iloc[i + 1]
            transitions.append({"type": "Recovery", "before": post, "after": next_pre})

    # Final EndIteration as last recovery if present
    if include_recovery and not post_rows.empty and not end_rows.empty:
        transitions.append({
            "type": "Recovery",
            "before": post_rows.iloc[-1],
            "after":  end_rows.iloc[-1],
        })

    return transitions


def generate_event_transition_frames(
    run: RunResult,
    profile_id: str,
    out_dir: str,
    *,
    include_recovery: bool = True,
    include_erosion: bool = True,
    zoom_x: tuple[float, float] | None = None,
    zoom_beach: bool = True,
    dpi: int = 120,
    figsize: tuple[float, float] = (10, 4),
) -> int:
    """One frame per event transition: erosion, storm, and recovery before/after pairs.

    Frame types (fill colour):
      Erosion  (blue)  : INIT → PreStorm_1, REC_t → PreStorm_{t+1}
      Storm    (red)   : PreStorm_t → PostStorm_t
      Recovery (green) : PostStorm_t → PreStorm_{t+1} / EndIteration
    """
    os.makedirs(out_dir, exist_ok=True)

    df = run.profiles[run.profiles["profile_id"] == profile_id].copy()

    # Collect all snapshot labels (INIT + REC needed for erosion transitions)
    all_labels = ["INIT", "PreStorm", "PostStorm", "REC", "EndIteration"]
    snaps = (
        df[df["label"].isin(all_labels)][["label", "t"]]
        .drop_duplicates()
        .sort_values("t")
        .reset_index(drop=True)
    )

    if snaps.empty:
        print("No snapshots found.")
        return 0

    transitions = _build_transitions(
        snaps, include_recovery=include_recovery, include_erosion=include_erosion
    )
    if not transitions:
        print("No transitions found.")
        return 0

    # Axis limits: use post-CSHORE labels for zoom detection (they share the
    # same physical coordinate system), then apply to all frame types.
    zoom_labels = ["PostStorm", "REC", "PreStorm", "EndIteration"]
    zoom_df = df[df["label"].isin(zoom_labels)]
    if zoom_x:
        xlim = zoom_x
    elif zoom_beach:
        xlim = _beach_zoom(zoom_df) if not zoom_df.empty else (df["x"].min(), df["x"].max())
    else:
        xlim = (df["x"].min(), df["x"].max())

    all_df = df[df["label"].isin(all_labels)]
    z_in_view = all_df[(all_df["x"] >= xlim[0]) & (all_df["x"] <= xlim[1])]["zb"]
    ylim = (max(z_in_view.min() - 0.5, -5.0), z_in_view.max() + 0.5)

    # Pre-fetch unique (label, t) → xy for all transitions
    needed: set[tuple] = set()
    for tr in transitions:
        needed.add((tr["before"]["label"], tr["before"]["t"]))
        needed.add((tr["after"]["label"],  tr["after"]["t"]))
    xy_cache: dict[tuple, tuple[np.ndarray, np.ndarray] | None] = {
        key: _get_profile_snapshot(df, profile_id, key[0], key[1])
        for key in needed
    }

    # Build figure with static elements
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.fill_between(xlim, -200, 0, color="#cce5ff", alpha=0.25, zorder=0)
    ax.axhline(0, color="#4a90d9", lw=0.7, alpha=0.7)
    ax.set_xlabel("Cross-shore position (m, offshore→landward)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.25)

    before_line, = ax.plot([], [], color="#888888", lw=1.2, ls="--", zorder=3, label="before")
    after_line,  = ax.plot([], [], lw=2.2, zorder=5, label="after")
    fill_poly    = [None]  # mutable container so inner loop can replace it
    title_text   = ax.set_title("")
    annot        = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )
    ax.legend(fontsize=8, loc="upper right")

    for i, tr in enumerate(transitions):
        style  = _TRANSITION_STYLE[tr["type"]]
        b_row, a_row = tr["before"], tr["after"]
        xy_b = xy_cache.get((b_row["label"], b_row["t"]))
        xy_a = xy_cache.get((a_row["label"], a_row["t"]))

        if xy_b is not None:
            before_line.set_data(xy_b[0], xy_b[1])
        if xy_a is not None:
            after_line.set_data(xy_a[0], xy_a[1])
            after_line.set_color(style["color"])

        # Remove previous fill and redraw
        if fill_poly[0] is not None:
            fill_poly[0].remove()
            fill_poly[0] = None
        if xy_b is not None and xy_a is not None:
            # Interpolate before onto after x-grid for fill
            z_b_interp = np.interp(xy_a[0], xy_b[0], xy_b[1])
            fill_poly[0] = ax.fill_between(
                xy_a[0], z_b_interp, xy_a[1],
                where=(xy_a[0] >= xlim[0]) & (xy_a[0] <= xlim[1]),
                color=style["fill"], alpha=0.4, zorder=2,
            )

        title_text.set_text(
            f"{run.reach_id} / {run.alt_id} — {profile_id}  |  {tr['type']}"
        )
        annot.set_text(
            f"{tr['type']:<10}\n"
            f"t: {b_row['t']:.0f}d → {a_row['t']:.0f}d\n"
            f"{b_row['label']} → {a_row['label']}"
        )

        fig.savefig(os.path.join(out_dir, f"frame_{i:05d}.png"),
                    dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    n = len(transitions)
    print(f"Event frames: {n} → {out_dir}/frame_NNNNN.png")
    print(f"  ffmpeg -r 1 -i '{out_dir}/frame_%05d.png' "
          f"-vcodec libx264 -pix_fmt yuv420p '{out_dir}/../{profile_id}_{run.alt_id}_events.mp4'")
    return n


# ---------------------------------------------------------------------------
# Plot 3b — single-line snapshot series for ffmpeg
# ---------------------------------------------------------------------------

def generate_profile_frames(
    run: RunResult,
    profile_id: str,
    out_dir: str,
    *,
    labels: Sequence[str] = ("PreStorm", "PostStorm", "EndIteration"),
    line_color: str = "#1f77b4",
    zoom_x: tuple[float, float] | None = None,
    zoom_beach: bool = True,
    dpi: int = 120,
    figsize: tuple[float, float] = (10, 4),
) -> int:
    """Write one PNG per event snapshot for ffmpeg assembly.

    One frame per snapshot in ``labels`` order.  REC excluded by default —
    it is an analytical interpolation, not a CSHORE run.  Use ffmpeg ``-r``
    to control playback speed.

    Returns total frames written and prints the ffmpeg command.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = run.profiles[run.profiles["profile_id"] == profile_id].copy()

    snaps = (
        df[df["label"].isin(labels)][["label", "t"]]
        .drop_duplicates()
        .sort_values("t")
        .reset_index(drop=True)
    )

    # Drop snapshots on incompatible (pre-CSHORE) x grid
    snap_xmax = {
        (r["label"], r["t"]): df[
            (df["label"] == r["label"]) & np.isclose(df["t"], r["t"])
        ]["x"].max()
        for _, r in snaps.iterrows()
    }
    if not snap_xmax:
        print("No compatible snapshots found.")
        return 0
    ref_xmax = max(snap_xmax.values())
    snaps = snaps[
        snaps.apply(lambda r: snap_xmax[(r["label"], r["t"])] >= 0.8 * ref_xmax, axis=1)
    ].reset_index(drop=True)

    if snaps.empty:
        print("No compatible snapshots found.")
        return 0

    # Axis limits (fixed across all frames)
    compat_df = df[df["label"].isin(labels)]
    if zoom_x:
        xlim = zoom_x
    elif zoom_beach:
        xlim = _beach_zoom(compat_df[compat_df.apply(
            lambda r: snap_xmax.get((r["label"], r["t"]), 0) >= 0.8 * ref_xmax, axis=1)])
    else:
        xlim = (compat_df["x"].min(), compat_df["x"].max())

    z_in_view = compat_df[(compat_df["x"] >= xlim[0]) & (compat_df["x"] <= xlim[1])]["zb"]
    ylim = (max(z_in_view.min() - 0.5, -5.0), z_in_view.max() + 0.5)

    # Pre-fetch snapshot arrays
    snap_xy: list[tuple[np.ndarray, np.ndarray] | None] = [
        _get_profile_snapshot(df, profile_id, r["label"], r["t"])
        for _, r in snaps.iterrows()
    ]
    ref_xy = next((xy for xy in snap_xy if xy is not None), None)
    ref_t  = snaps.iloc[0]["t"]

    # Build figure — static elements drawn once
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.fill_between(xlim, -200, 0, color="#cce5ff", alpha=0.25, zorder=0)
    ax.axhline(0, color="#4a90d9", lw=0.7, alpha=0.7)
    ax.set_xlabel("Cross-shore position (m, offshore→landward)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{run.reach_id} / {run.alt_id} — {profile_id}")
    ax.grid(True, alpha=0.25)

    if ref_xy is not None:
        ax.plot(ref_xy[0], ref_xy[1], color="#aaaaaa", lw=1.0, ls="--", zorder=1)

    current_line, = ax.plot([], [], color=line_color, lw=2.2, zorder=5)
    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    # Render — one file per snapshot
    for i, (_, row) in enumerate(snaps.iterrows()):
        xy = snap_xy[i]
        if xy is not None:
            current_line.set_data(xy[0], xy[1])
        time_text.set_text(
            f"t = {row['t']:>6.1f} d\n"
            f"{row['label']:<14}\n"
            f"ref: t={ref_t:.0f}d  ---"
        )
        fig.savefig(os.path.join(out_dir, f"frame_{i:05d}.png"),
                    dpi=dpi, bbox_inches="tight")

    plt.close(fig)

    n = len(snaps)
    print(f"Frames written: {n} → {out_dir}/frame_NNNNN.png")
    print(f"ffmpeg (adjust -r for playback speed):")
    print(f"  ffmpeg -r 2 -i '{out_dir}/frame_%05d.png' "
          f"-vcodec libx264 -pix_fmt yuv420p '{out_dir}/../{profile_id}_{run.alt_id}.mp4'")
    return n


def make_profile_video(
    run: RunResult,
    profile_id: str,
    out_path: str,
    fps: int = 2,
    zoom_x: tuple[float, float] | None = None,
    dpi: int = 120,
    labels: Sequence[str] = ("PreStorm", "PostStorm", "EndIteration"),
) -> None:
    """Generate frames then assemble with ffmpeg if available."""
    import subprocess
    frame_dir = os.path.splitext(out_path)[0] + "_frames"
    n = generate_profile_frames(
        run, profile_id, frame_dir,
        labels=labels, zoom_x=zoom_x, dpi=dpi,
    )
    if n == 0:
        return
    default_fps = fps
    cmd = [
        "ffmpeg", "-y", "-r", str(default_fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved → {out_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"ffmpeg not available or failed — frames are in {frame_dir}")


# ---------------------------------------------------------------------------
# Debug — idealized fit overlaid on the raw profile
# ---------------------------------------------------------------------------

def plot_idealized_fit(
    x: np.ndarray,
    zb: np.ndarray,
    design_berm_elevation: float,
    datum: float = 0.0,
    ref=None,
    ax=None,
) -> Figure:
    """Overlay the idealized (fitted) profile on the raw profile for debugging.

    Fits ``(x, zb)`` with ``erosion.metrics.fit_profile`` and plots: raw bed,
    the piecewise-linear idealization, the datum, and the fit residual on a twin
    axis.  Landmark elevations and the morphology type are annotated.
    """
    from .metrics import fit_profile

    x = np.asarray(x, dtype=float)
    zb = np.asarray(zb, dtype=float)
    m, ideal = fit_profile(x, zb, design_berm_elevation, datum, ref=ref)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.plot(x, zb, color="0.4", lw=1.0, label="raw zb")
    ax.axhline(datum, color="steelblue", lw=0.8, ls=":", label="datum")

    title = f"morph={m.morph_type}"
    if ideal is not None:
        zi = ideal.evaluate(x)
        ax.plot(x, zi, color="C3", lw=1.8, label="idealized")
        ax.scatter(ideal.knots_x, ideal.knots_z, color="C3", s=25, zorder=5)
        resid = zb - zi
        axr = ax.twinx()
        axr.plot(x, resid, color="C0", lw=0.7, alpha=0.5)
        axr.set_ylabel("residual (m)", color="C0")
        axr.axhline(0.0, color="C0", lw=0.5, ls=":", alpha=0.4)
        title += f"   fit_quality={m.fit_quality:.3f}"
        if m.berm_scarp or m.dune_scarp:
            tags = [t for t, on in (("berm", m.berm_scarp), ("dune", m.dune_scarp)) if on]
            title += f"   scarp: {', '.join(tags)} (h={m.scarp_height:.2f})"

    if not np.isnan(m.dune_crest_x):
        ax.scatter([m.dune_crest_x], [m.dune_crest_elevation], marker="^",
                   color="C2", s=60, zorder=6, label="dune crest")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
