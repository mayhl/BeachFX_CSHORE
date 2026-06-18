from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import SnapshotLabel

if TYPE_CHECKING:
    from .profile import Profile
    from .runner.base import CSHOREResult


@dataclass
class RunMeta:
    reach_id:       str
    alternative_id: str
    sim_start:      datetime
    lifecycle:      int = 0


class ResultsSink:
    def record_storm_hazard(
        self, profile_id: str, t: float, result: CSHOREResult
    ) -> None: ...

    def record_nourishment(
        self,
        profile_id: str,
        t_start: float,
        t_end: float,
        volume_m3: float,
        event_type: str,
    ) -> None: ...

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
        self._hazard_rows: list[dict] = []
        self._nourishment_rows: list[dict] = []

    def record_storm_hazard(
        self, profile_id: str, t: float, result: CSHOREResult
    ) -> None:
        n = len(result.x)
        eta = np.asarray(result.eta)
        Hs  = np.asarray(result.Hs)
        for i in range(n):
            self._hazard_rows.append({
                "profile_id": profile_id,
                "t_storm":    t,
                "node_idx":   i,
                "x":          float(result.x[i]),
                "mwl":        float(eta[i]) if i < len(eta) else float("nan"),
                "Hs":         float(Hs[i])  if i < len(Hs)  else float("nan"),
                "runup_m":    float(result.runup_m),
            })

    def record_nourishment(
        self,
        profile_id: str,
        t_start: float,
        t_end: float,
        volume_m3: float,
        event_type: str,
    ) -> None:
        CY_PER_M3 = 1.30795
        self._nourishment_rows.append({
            "event_type": event_type,
            "profile_id": profile_id,
            "t_start":    t_start,
            "t_end":      t_end,
            "volume_m3":  volume_m3,
            "volume_cy":  volume_m3 * CY_PER_M3,
        })

    def flush(self, profiles: list[Profile], meta: RunMeta) -> None:
        self._write_profiles(profiles)
        self._write_storm_hazard()
        self._write_profile_metrics(profiles)
        self._write_profile_events(profiles)
        self._write_segment_events()
        self._write_metadata(meta, profiles)
        self._write_summary(meta, profiles)

    # ------------------------------------------------------------------

    def _write_profiles(self, profiles: list[Profile]) -> None:
        rows = []
        for p in profiles:
            for snap in p.snapshots:
                for idx, (x, zb) in enumerate(zip(snap.x, snap.zb)):
                    rows.append({
                        "profile_id": p.id,
                        "label":      snap.label.value,
                        "t":          snap.t,
                        "node_idx":   idx,
                        "x":          float(x),
                        "zb":         float(zb),
                    })
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(self.out_dir, "profiles.parquet"), index=False
            )

    def _write_storm_hazard(self) -> None:
        if self._hazard_rows:
            pd.DataFrame(self._hazard_rows).to_parquet(
                os.path.join(self.out_dir, "storm_hazard.parquet"), index=False
            )

    def _write_profile_metrics(self, profiles: list[Profile]) -> None:
        rows = []
        for p in profiles:
            for snap in p.snapshots:
                if snap.metrics is None:
                    continue
                m = snap.metrics
                rows.append({
                    "profile_id":         p.id,
                    "label":              snap.label.value,
                    "t":                  snap.t,
                    "morph_type":         m.morph_type,
                    "shoreline_x":        m.shoreline_x,
                    "berm_width":         m.berm_width,
                    "foreshore_slope":    m.foreshore_slope,
                    "dune_height":        m.dune_height,
                    "dune_x":             m.dune_x,
                    "dune_width":         m.dune_width,
                    "dune_front_slope":   m.dune_front_slope,
                    "dune_back_slope":    m.dune_back_slope,
                    "upland_elevation":   m.upland_elevation,
                    "volume_above_datum": m.volume_above_datum,
                    "scarp_present":      m.scarp_present,
                    "max_beach_slope":    m.max_beach_slope,
                    "dune_front_resid":   m.dune_front_resid,
                })
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(self.out_dir, "profile_metrics.parquet"), index=False
            )

    def _write_profile_events(self, profiles: list[Profile]) -> None:
        """Lightweight snapshot log — one row per (profile, snapshot). No node data."""
        rows = []
        for p in profiles:
            for snap in p.snapshots:
                rows.append({
                    "profile_id":          p.id,
                    "label":               snap.label.value,
                    "t":                   snap.t,
                    "storm_response_type": (
                        snap.storm_response_type.value
                        if snap.storm_response_type is not None else None
                    ),
                })
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

    def _write_metadata(self, meta: RunMeta, profiles: list[Profile]) -> None:
        storm_times = {r["t_storm"] for r in self._hazard_rows}
        data = {
            "reach_id":       meta.reach_id,
            "alternative_id": meta.alternative_id,
            "lifecycle":      meta.lifecycle,
            "sim_start":      meta.sim_start.isoformat(),
            "n_profiles":     len(profiles),
            "n_storms":       len(storm_times),
            "n_nourishment":  len(self._nourishment_rows),
        }
        with open(os.path.join(self.out_dir, "run_metadata.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _write_summary(self, meta: RunMeta, profiles: list[Profile]) -> None:
        storm_times = {r["t_storm"] for r in self._hazard_rows}
        n_metrics = sum(
            1 for p in profiles for s in p.snapshots if s.metrics is not None
        )
        with open(os.path.join(self.out_dir, "run_summary.txt"), "w") as f:
            f.write("BeachFX Simulation Run Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Reach:        {meta.reach_id}\n")
            f.write(f"Alternative:  {meta.alternative_id}\n")
            f.write(f"Lifecycle:    {meta.lifecycle}\n")
            f.write(f"Sim start:    {meta.sim_start.isoformat()}\n")
            f.write(f"Storms:       {len(storm_times)}\n")
            f.write(f"Profiles:     {len(profiles)}\n")
            f.write(f"Hazard rows:  {len(self._hazard_rows)}\n")
            f.write(f"Metric rows:  {n_metrics}\n")
            f.write(f"Nourishment:  {len(self._nourishment_rows)}\n")
