from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .profile import Profile
    from .runner.base import CSHOREResult


def _fit_to_n(a: np.ndarray, n: int) -> np.ndarray:
    """Truncate or NaN-pad a 1-D array to length ``n`` (float dtype)."""
    a = np.asarray(a, dtype=float)
    if len(a) >= n:
        return a[:n]
    out = np.full(n, np.nan)
    out[: len(a)] = a
    return out


def _concat_chunks(chunks: list[dict], cols: list[str]) -> pd.DataFrame:
    """Build a DataFrame by column-wise concatenation of per-record array chunks."""
    return pd.DataFrame({c: np.concatenate([ch[c] for ch in chunks]) for c in cols})


@dataclass
class RunMeta:
    reach_id: str
    alternative_id: str
    sim_start: datetime
    lifecycle: int = 0


class ResultsSink:
    def record_storm_hazard(self, profile_id: str, t: float, result: CSHOREResult) -> None: ...

    def record_nourishment(
        self,
        profile_id: str,
        t_start: float,
        t_end: float,
        volume_m3: float,
        event_type: str,
    ) -> None: ...

    def record_warning(self, profile_id: str, t: float, message: str) -> None: ...

    def record_decision(self, kind, t: float, profile_id: str | None = None, **payload) -> None:
        """Record a reach/SIM-scope orchestrator decision (NOURISH_TRIGGER, …).

        No-op in the base sink; a capturing/recording sink collects them, and the
        (future) event-log sink persists them as ``scope=REACH/SIM`` rows.
        """

    def flush(self, profiles: list[Profile], meta: RunMeta) -> None: ...


class NullResultsSink(ResultsSink):
    pass


class ParquetResultsSink(ResultsSink):
    """Writes profile snapshots, storm hazard, nourishment events, and run metadata.

    Output layout::

        {out_root}/{reach_id}/{alternative_id}/lc_{lifecycle:04d}/
            profiles.parquet        # all labeled snapshots — one row per (profile, label, node)
            storm_hazard.parquet    # CSHORE spatial output — one row per (profile, storm, node)
            profile_metrics.parquet # 0-D morphology metrics per (profile, snapshot)
            profile_events.parquet  # lightweight snapshot log — one row per (profile, snapshot)
            segment_events.csv      # nourishment placement events
            run_metadata.json
            run_summary.txt
    """

    def __init__(self, out_root: str, reach_id: str, alternative_id: str, lifecycle: int = 0):
        self.lifecycle = lifecycle
        self.out_dir = os.path.join(out_root, reach_id, alternative_id, f"lc_{lifecycle:04d}")
        os.makedirs(self.out_dir, exist_ok=True)
        self._hazard_chunks: list[dict] = []  # one array-chunk per (profile, storm)
        self._storm_times: set[float] = set()
        self._nourishment_rows: list[dict] = []
        self._warning_rows: list[dict] = []

    def record_storm_hazard(self, profile_id: str, t: float, result: CSHOREResult) -> None:
        n = len(result.x)
        self._storm_times.add(float(t))
        self._hazard_chunks.append(
            {
                "profile_id": np.full(n, profile_id, dtype=object),
                "t_storm": np.full(n, float(t)),
                "node_idx": np.arange(n),
                "x": np.asarray(result.x, dtype=float),
                "mwl": _fit_to_n(result.eta, n),
                "Hs": _fit_to_n(result.Hs, n),
                "runup_m": np.full(n, float(result.runup_m)),
            }
        )

    def record_nourishment(
        self,
        profile_id: str,
        t_start: float,
        t_end: float,
        volume_m3: float,
        event_type: str,
    ) -> None:
        CY_PER_M3 = 1.30795
        self._nourishment_rows.append(
            {
                "event_type": event_type,
                "profile_id": profile_id,
                "t_start": t_start,
                "t_end": t_end,
                "volume_m3": volume_m3,
                "volume_cy": volume_m3 * CY_PER_M3,
            }
        )

    def record_warning(self, profile_id: str, t: float, message: str) -> None:
        """Record a non-fatal run warning (e.g. a skipped/inundated storm).

        Interim channel: how INUNDATION is ultimately handled is undecided, so
        for now the storm is skipped, the profile reused, and the event surfaced
        here (``warnings.csv`` + the run summary) rather than silently logged.
        """
        self._warning_rows.append({"profile_id": profile_id, "t": float(t), "message": message})

    def flush(self, profiles: list[Profile], meta: RunMeta) -> None:
        self._write_profiles(profiles)
        self._write_storm_hazard()
        self._write_profile_metrics(profiles)
        self._write_profile_events(profiles)
        self._write_segment_events()
        self._write_warnings()
        self._write_metadata(meta, profiles)
        self._write_summary(meta, profiles)

    # ------------------------------------------------------------------

    def _write_profiles(self, profiles: list[Profile]) -> None:
        chunks = []
        for p in profiles:
            for snap in p.snapshots:
                n = len(snap.x)
                chunks.append(
                    {
                        "profile_id": np.full(n, p.id, dtype=object),
                        "label": np.full(n, snap.label.value, dtype=object),
                        "t": np.full(n, float(snap.t)),
                        "node_idx": np.arange(n),
                        "x": np.asarray(snap.x, dtype=float),
                        "zb": np.asarray(snap.zb, dtype=float),
                    }
                )
        if chunks:
            cols = ["profile_id", "label", "t", "node_idx", "x", "zb"]
            _concat_chunks(chunks, cols).to_parquet(
                os.path.join(self.out_dir, "profiles.parquet"), index=False
            )

    def _write_storm_hazard(self) -> None:
        if self._hazard_chunks:
            cols = ["profile_id", "t_storm", "node_idx", "x", "mwl", "Hs", "runup_m"]
            _concat_chunks(self._hazard_chunks, cols).to_parquet(
                os.path.join(self.out_dir, "storm_hazard.parquet"), index=False
            )

    def _write_profile_metrics(self, profiles: list[Profile]) -> None:
        rows = []
        for p in profiles:
            for snap in p.snapshots:
                if snap.metrics is None:
                    continue
                m = snap.metrics
                rows.append(
                    {
                        "profile_id": p.id,
                        "label": snap.label.value,
                        "t": snap.t,
                        "morph_type": m.morph_type,
                        "shoreline_x": m.shoreline_x,
                        "foreshore_slope": m.foreshore_slope,
                        "berm_elevation": m.berm_elevation,
                        "berm_width": m.berm_width,
                        "dune_crest_elevation": m.dune_crest_elevation,
                        "dune_crest_x": m.dune_crest_x,
                        "dune_width": m.dune_width,
                        "dune_front_width": m.dune_front_width,
                        "dune_back_width": m.dune_back_width,
                        "dune_top_width": m.dune_top_width,
                        "dune_front_relief": m.dune_front_relief,
                        "dune_back_relief": m.dune_back_relief,
                        "dune_front_slope": m.dune_front_slope,
                        "dune_back_slope": m.dune_back_slope,
                        "upland_elevation": m.upland_elevation,
                        "volume_above_datum": m.volume_above_datum,
                        "berm_scarp": m.berm_scarp,
                        "dune_scarp": m.dune_scarp,
                        "scarp_height": m.scarp_height,
                        "fit_quality": m.fit_quality,
                    }
                )
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(self.out_dir, "profile_metrics.parquet"), index=False
            )

    def _write_profile_events(self, profiles: list[Profile]) -> None:
        """Lightweight snapshot log — one row per (profile, snapshot). No node data."""
        rows = []
        for p in profiles:
            for snap in p.snapshots:
                rows.append(
                    {
                        "profile_id": p.id,
                        "label": snap.label.value,
                        "t": snap.t,
                        "storm_response_type": (
                            snap.storm_response_type.value
                            if snap.storm_response_type is not None
                            else None
                        ),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(self.out_dir, "profile_events.parquet"), index=False
            )

    def _write_segment_events(self) -> None:
        cols = ["event_type", "profile_id", "t_start", "t_end", "volume_m3", "volume_cy"]
        df = (
            pd.DataFrame(self._nourishment_rows)
            if self._nourishment_rows
            else pd.DataFrame(columns=cols)
        )
        df.to_csv(os.path.join(self.out_dir, "segment_events.csv"), index=False)

    def _write_warnings(self) -> None:
        cols = ["profile_id", "t", "message"]
        df = (
            pd.DataFrame(self._warning_rows)
            if self._warning_rows
            else pd.DataFrame(columns=cols)
        )
        df.to_csv(os.path.join(self.out_dir, "warnings.csv"), index=False)

    def _write_metadata(self, meta: RunMeta, profiles: list[Profile]) -> None:
        data = {
            "reach_id": meta.reach_id,
            "alternative_id": meta.alternative_id,
            "lifecycle": meta.lifecycle,
            "sim_start": meta.sim_start.isoformat(),
            "n_profiles": len(profiles),
            "n_storms": len(self._storm_times),
            "n_nourishment": len(self._nourishment_rows),
            "n_warnings": len(self._warning_rows),
        }
        with open(os.path.join(self.out_dir, "run_metadata.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _write_summary(self, meta: RunMeta, profiles: list[Profile]) -> None:
        n_hazard_rows = sum(len(ch["node_idx"]) for ch in self._hazard_chunks)
        n_metrics = sum(1 for p in profiles for s in p.snapshots if s.metrics is not None)
        with open(os.path.join(self.out_dir, "run_summary.txt"), "w") as f:
            f.write("BeachFX Simulation Run Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Reach:        {meta.reach_id}\n")
            f.write(f"Alternative:  {meta.alternative_id}\n")
            f.write(f"Lifecycle:    {meta.lifecycle}\n")
            f.write(f"Sim start:    {meta.sim_start.isoformat()}\n")
            f.write(f"Storms:       {len(self._storm_times)}\n")
            f.write(f"Profiles:     {len(profiles)}\n")
            f.write(f"Hazard rows:  {n_hazard_rows}\n")
            f.write(f"Metric rows:  {n_metrics}\n")
            f.write(f"Nourishment:  {len(self._nourishment_rows)}\n")
            f.write(f"Warnings:     {len(self._warning_rows)}\n")
            for w in self._warning_rows:
                f.write(f"  ! t={w['t']:.1f}d  {w['profile_id']}: {w['message']}\n")
