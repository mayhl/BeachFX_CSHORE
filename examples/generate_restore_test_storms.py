#!/usr/bin/env python3
"""Select a non-inundating strong-storm set for the restore-test harness.

Screens a CHS storm catalog down to the strongest storms whose peak still water
level (SWL) stays below the reach's controlling profile crest — with a wave-setup
margin so a storm that would overtop via setup (not SWL alone) is excluded — and
writes each selected storm as its own lifecycle.  This is the reproducible
generator behind ``data/storms/restore_test_6lc.parquet`` (the ad-hoc selection
that first built it is retired).

Screen rule (per storm)::

    keep iff  peak_SWL + setup_margin < min_profile_crest

The setup margin (default 0.5 m) accounts for wave setup raising the effective
water level above the reported SWL at the crest — a storm inside the margin can
wash the profile out even though its SWL alone clears the crest, so it is dropped.
Surviving storms are ranked by peak SWL (descending = strongest first) and the
top ``--n`` are written, one per lifecycle, so the restore harness runs
``(N profiles) x (N storms)`` — each storm damaging a fresh profile.

The profile crest is the maximum bed elevation in the CSHORE frame (landward-
positive, metres, via ``load_raw_profile``) — the elevation surge must exceed to
wash the profile out.  Screening against the MIN crest across the reach profiles
keeps every profile non-inundating.

Each selected storm keeps its own hydrograph (rows, intra-storm spacing) but is
re-based to start at ``--storm-start`` and re-indexed to its own lifecycle, so the
lifecycles are independent and directly comparable (the pipeline runs each one
separately from the shared ``sim_start``).

Usage::

    python examples/generate_restore_test_storms.py
    python examples/generate_restore_test_storms.py \
        --source data/chs_300storms.parquet \
        --profiles 'data/profiles_test/*.csv' \
        --n 6 --setup-margin 0.5 --storm-start 2025-01-06 \
        --out data/storms/restore_test_6lc.parquet
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from erosion.geometry import load_raw_profile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def min_profile_crest(profile_paths: list[str], d50: float = 0.3) -> float:
    """Lowest max-bed-elevation (CSHORE frame, metres) across the reach profiles.

    A storm inundates the weakest profile first, so screening against the minimum
    crest keeps every profile non-inundating.
    """
    crests = [float(np.nanmax(load_raw_profile(p, d50)["z"])) for p in profile_paths]
    if not crests:
        raise SystemExit("no profiles found to derive a crest from")
    return min(crests)


def select_storms(source: pd.DataFrame, crest: float, setup_margin: float, n: int) -> list[str]:
    """The ``n`` strongest storm ids whose ``peak_SWL + setup_margin < crest``.

    Strongest = highest peak SWL.  Raises if fewer than ``n`` storms survive the
    screen (no silent short set — widen the source or the crest instead)."""
    peak = source.groupby("storm_id")["water_elevation"].max()
    kept = peak[peak + setup_margin < crest].sort_values(ascending=False)
    if len(kept) < n:
        raise SystemExit(
            f"only {len(kept)} storm(s) pass the screen (SWL < {crest - setup_margin:.2f} m); "
            f"need {n}. Widen the source catalog or lower the setup margin."
        )
    return list(kept.index[:n])


def build(source: pd.DataFrame, storm_ids: list[str], storm_start: pd.Timestamp) -> pd.DataFrame:
    """Assemble the per-lifecycle table: each selected storm -> its own lifecycle,
    ranked order preserved (lifecycle 0 = strongest), each re-based to ``storm_start``
    with its intra-storm spacing intact."""
    frames = []
    for lc, sid in enumerate(storm_ids):
        s = source[source["storm_id"] == sid].sort_values("hydro_tstp").copy()
        s["lifecycle"] = lc
        s["date"] = storm_start + (s["date"] - s["date"].min())
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source", default="data/chs_300storms.parquet", help="source CHS storm catalog (parquet)"
    )
    ap.add_argument(
        "--profiles",
        default="data/profiles_test/*.csv",
        help="glob of reach profile CSVs (crest reference)",
    )
    ap.add_argument("--n", type=int, default=6, help="number of storms / lifecycles to select")
    ap.add_argument(
        "--setup-margin", type=float, default=0.5, help="wave-setup margin (m) below the crest"
    )
    ap.add_argument(
        "--d50", type=float, default=0.3, help="d50 for the profile loader (crest-independent)"
    )
    ap.add_argument(
        "--storm-start", default="2025-01-06", help="date each lifecycle's storm re-based to start"
    )
    ap.add_argument(
        "--out", default="data/storms/restore_test_6lc.parquet", help="output parquet path"
    )
    args = ap.parse_args()

    source_path = os.path.join(ROOT, args.source)
    profile_paths = sorted(glob.glob(os.path.join(ROOT, args.profiles)))
    out_path = os.path.join(ROOT, args.out)

    src = pd.read_parquet(source_path)
    src["storm_id"] = src["storm_id"].astype(str)
    crest = min_profile_crest(profile_paths, args.d50)
    storm_ids = select_storms(src, crest, args.setup_margin, args.n)
    out = build(src, storm_ids, pd.Timestamp(args.storm_start))
    out.to_parquet(out_path, index=False)

    peak = src.groupby("storm_id")["water_elevation"].max()
    screen = crest - args.setup_margin
    print(f"min profile crest = {crest:.2f} m  ->  screen: peak SWL < {screen:.2f} m")
    print(f"selected {len(storm_ids)} storm(s) (lifecycle 0 = strongest):")
    for lc, sid in enumerate(storm_ids):
        print(f"  lc{lc}: storm {sid:>5}  peak SWL = {peak[sid]:.2f} m")
    print(f"wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
