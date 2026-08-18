from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .units import M3_TO_CY

if TYPE_CHECKING:
    from .config import ReachConfig
    from .profile import Profile
    from .runner.base import CSHOREResult


def write_parquet_with_footer(df: pd.DataFrame, path: str, footer: dict | None) -> None:
    """Write ``df`` to parquet, embedding ``footer`` (str->str) in the file's key-value
    metadata so the output is self-describing / reproducible.  Shared with postprocess."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    if footer:
        existing = table.schema.metadata or {}
        merged = {**existing, **{k.encode(): v.encode() for k, v in footer.items()}}
        table = table.replace_schema_metadata(merged)
    pq.write_table(table, path)


def read_parquet_footer(path: str) -> dict:
    """Decode the ``beachfx_*`` key-value footer written by ``write_parquet_with_footer``."""
    md = pq.read_metadata(path).metadata or {}
    return {k.decode(): v.decode() for k, v in md.items() if k.decode().startswith("beachfx_")}


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


# ProfileMetrics fields emitted to profile_metrics.parquet, in column order.
# Deliberately omits n_upland_nodes (an internal count, not a reported metric),
# so this is an explicit list rather than dataclasses.asdict(m).
_METRIC_COLUMNS = (
    "morph_type",
    "shoreline_x",
    "foreshore_slope",
    "berm_elevation",
    "berm_width",
    "berm_x",
    "dune_crest_elevation",
    "dune_crest_x",
    "dune_width",
    "dune_front_width",
    "dune_back_width",
    "dune_top_width",
    "dune_front_relief",
    "dune_back_relief",
    "dune_front_slope",
    "dune_back_slope",
    "upland_elevation",
    "volume_above_datum",
    "berm_scarp",
    "dune_scarp",
    "scarp_height",
    "fit_quality",
)


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
        placed_m3: float,
        event_type: str,
        borrow_m3: float = 0.0,
    ) -> None: ...

    def record_warning(self, profile_id: str, t: float, message: str) -> None: ...

    def record_decision(self, kind, t: float, profile_id: str | None = None, **payload) -> None:
        """Record a reach/SIM-scope decision (NOURISH_TRIGGER, …).

        No-op in the base sink; the parquet sink persists them to
        ``decisions.parquet`` (the reach-scope counterpart of ``events.parquet``).
        """

    def flush(self, profiles: list[Profile], meta: RunMeta) -> None: ...


class NullResultsSink(ResultsSink):
    pass


class ParquetResultsSink(ResultsSink):
    """Writes profile snapshots, storm hazard, nourishment events, and run metadata.

    Output layout (the file list ``flush`` writes; the suite asserts the real
    directory against it, so keep the two together)::

        {out_root}/{reach_id}/{alternative_id}/lc_{lifecycle:04d}/
            events.parquet          # append-only applied-event log (the audit spine)
            profiles.parquet        # all labeled snapshots — one row per (profile, label, node)
            storm_hazard.parquet    # CSHORE spatial output — one row per (profile, storm, node)
            profile_metrics.parquet # 0-D metrics per fitted snapshot (only when any exist)
            snapshots.parquet       # lightweight snapshot log — one row per (profile, snapshot)
            decisions.parquet       # reach-scope decisions — one row each
            placements.csv          # nourishment placement events
            warnings.csv            # per-profile run warnings
            profile_georef.parquet  # transect georeference (only when a profile has one)
            run_metadata.json
            run_summary.txt
    """

    def __init__(
        self,
        out_root: str,
        reach_id: str,
        alternative_id: str,
        lifecycle: int = 0,
        config: ReachConfig | None = None,
    ):
        self.lifecycle = lifecycle
        self.out_dir = os.path.join(out_root, reach_id, alternative_id, f"lc_{lifecycle:04d}")
        os.makedirs(self.out_dir, exist_ok=True)
        self._hazard_chunks: list[dict] = []  # one array-chunk per (profile, storm)
        self._storm_times: set[float] = set()
        self._nourishment_rows: list[dict] = []
        self._warning_rows: list[dict] = []
        self._decision_rows: list[dict] = []
        # Run config embedded in every parquet footer (self-describing outputs).
        self._config_json = config.model_dump_json() if config is not None else None
        self._footer: dict | None = None

    def _build_footer(self, meta: RunMeta) -> dict:
        """Key-value footer stamped into every parquet: run identity + full config JSON."""
        footer = {
            "beachfx_reach_id": meta.reach_id,
            "beachfx_alternative_id": meta.alternative_id,
            "beachfx_lifecycle": str(meta.lifecycle),
            "beachfx_sim_start": meta.sim_start.isoformat(),
        }
        if self._config_json is not None:
            footer["beachfx_config"] = self._config_json
        return footer

    def _to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        write_parquet_with_footer(df, os.path.join(self.out_dir, filename), self._footer)

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
        placed_m3: float,
        event_type: str,
        borrow_m3: float = 0.0,
    ) -> None:

        # placed_m3 = placement (geometry-effective, on the beach); borrow_m3 =
        # dredged volume (placement × ratio) — the basis for duration and cost.
        self._nourishment_rows.append(
            {
                "event_type": event_type,
                "profile_id": profile_id,
                "t_start": t_start,
                "t_end": t_end,
                "placed_m3": placed_m3,
                "placed_cy": placed_m3 * M3_TO_CY,
                "borrow_m3": borrow_m3,
                "borrow_cy": borrow_m3 * M3_TO_CY,
            }
        )

    def record_decision(self, kind, t: float, profile_id: str | None = None, **payload) -> None:
        """Persist a reach-scope decision (why the crew mobilized, deferred,
        or placed a partial fill) — the audit trail behind the profile-scope events.

        ``decision_seq`` is the emission order, and it is the only ordering that holds:
        a decision is logged at the time it *concerns*, which is not the time it was
        taken (INTERRUPT carries the next storm's date, BLACKOUT_DEFER the window's), so
        ``t`` alone is not monotone.  Payload keys vary by kind, so absent keys land null.
        """
        self._decision_rows.append(
            {
                "decision_seq": len(self._decision_rows),
                "kind": getattr(kind, "value", kind),
                "t": float(t),
                "profile_id": profile_id,
                **payload,
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
        self._footer = self._build_footer(meta)
        self._write_events(profiles)
        self._write_georef(profiles)
        self._write_profiles(profiles)
        self._write_storm_hazard()
        self._write_profile_metrics(profiles)
        self._write_snapshots(profiles)
        self._write_decisions()
        self._write_placements()
        self._write_warnings()
        self._write_metadata(meta, profiles)
        self._write_summary(meta, profiles)

    # ------------------------------------------------------------------

    def _write_events(self, profiles: list[Profile]) -> None:
        """Append-only event log — one row per applied event (Phase A source of truth).

        The bed each event produced lives in ``profiles.parquet`` (referenced by
        ``label`` + ``t``), so the log never duplicates node arrays; it carries the
        event type, its scalar payload, and the ``ref_pos`` seam (inert until Phase C).
        ``event_seq`` orders events within a profile.  Payload keys vary by event type,
        so absent keys land null in the columnar frame.
        """
        base_cols = ["profile_id", "event_seq", "event_type", "t", "label", "ref_pos"]
        rows = [
            {
                "profile_id": p.id,
                "event_seq": seq,
                "event_type": e.event_type,
                "t": e.t,
                "label": e.label,
                "ref_pos": e.ref_pos,
                **e.payload,
            }
            for p in profiles
            for seq, e in enumerate(p.events)
        ]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=base_cols)
        self._to_parquet(df, "events.parquet")

    def _write_georef(self, profiles: list[Profile]) -> None:
        """Per-profile transect georeference (the GeoParquet hook), when set.  One row per
        georeferenced profile; postprocess reads it to project the grid to lon/lat and build
        the reach polygon.  No file when no profile carries a georef (today's case)."""
        rows = [
            {
                "profile_id": p.id,
                "origin_lon": p.georef.origin_lon,
                "origin_lat": p.georef.origin_lat,
                "azimuth_deg": p.georef.azimuth_deg,
            }
            for p in profiles
            if p.georef is not None
        ]
        if rows:
            self._to_parquet(pd.DataFrame(rows), "profile_georef.parquet")

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
            self._to_parquet(_concat_chunks(chunks, cols), "profiles.parquet")

    def _write_storm_hazard(self) -> None:
        if self._hazard_chunks:
            cols = ["profile_id", "t_storm", "node_idx", "x", "mwl", "Hs", "runup_m"]
            self._to_parquet(_concat_chunks(self._hazard_chunks, cols), "storm_hazard.parquet")

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
                        **{col: getattr(m, col) for col in _METRIC_COLUMNS},
                    }
                )
        if rows:
            self._to_parquet(pd.DataFrame(rows), "profile_metrics.parquet")

    def _write_snapshots(self, profiles: list[Profile]) -> None:
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
            self._to_parquet(pd.DataFrame(rows), "snapshots.parquet")

    def _write_decisions(self) -> None:
        """Reach-scope decision log — the counterpart of ``events.parquet``.  Always
        written (an empty frame is the honest record of a run that decided nothing)."""
        base_cols = ["decision_seq", "kind", "t", "profile_id"]
        df = (
            pd.DataFrame(self._decision_rows)
            if self._decision_rows
            else pd.DataFrame(columns=base_cols)
        )
        self._to_parquet(df, "decisions.parquet")

    def _write_placements(self) -> None:
        cols = [
            "event_type",
            "profile_id",
            "t_start",
            "t_end",
            "placed_m3",
            "placed_cy",
            "borrow_m3",
            "borrow_cy",
        ]
        df = (
            pd.DataFrame(self._nourishment_rows)
            if self._nourishment_rows
            else pd.DataFrame(columns=cols)
        )
        df.to_csv(os.path.join(self.out_dir, "placements.csv"), index=False)

    def _write_warnings(self) -> None:
        cols = ["profile_id", "t", "message"]
        df = pd.DataFrame(self._warning_rows) if self._warning_rows else pd.DataFrame(columns=cols)
        df.to_csv(os.path.join(self.out_dir, "warnings.csv"), index=False)

    def _write_metadata(self, meta: RunMeta, profiles: list[Profile]) -> None:
        data = {
            "reach_id": meta.reach_id,
            "alternative_id": meta.alternative_id,
            "lifecycle": meta.lifecycle,
            "sim_start": meta.sim_start.isoformat(),
            "n_profiles": len(profiles),
            "n_storms": len(self._storm_times),
            "n_events": sum(len(p.events) for p in profiles),
            "n_nourishment": len(self._nourishment_rows),
            "n_decisions": len(self._decision_rows),
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
            f.write(f"Events:       {sum(len(p.events) for p in profiles)}\n")
            f.write(f"Hazard rows:  {n_hazard_rows}\n")
            f.write(f"Metric rows:  {n_metrics}\n")
            f.write(f"Nourishment:  {len(self._nourishment_rows)}\n")
            f.write(f"Decisions:    {len(self._decision_rows)}\n")
            f.write(f"Warnings:     {len(self._warning_rows)}\n")
            for w in self._warning_rows:
                f.write(f"  ! t={w['t']:.1f}d  {w['profile_id']}: {w['message']}\n")
