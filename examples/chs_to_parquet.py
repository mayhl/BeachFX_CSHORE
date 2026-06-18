"""
Convert CHS timeseries CSV to the parquet format expected by run_cshore.py.

Selection options (can be combined; all filters applied before --n trim):

    --n 10                     first 10 storms in chronological order
    --top-surge 10             top 10 storms by peak surge
    --top-wave 10              top 10 storms by peak Hm0
    --min-surge 0.5            storms with peak surge >= 0.5 m
    --max-surge 3.0            storms with peak surge <= 3.0 m (avoids CSHORE failure)
    --min-wave 1.0             storms with peak Hm0 >= 1.0 m
    --ids 65,71,204            specific storm IDs (comma-separated)

Examples:
    uv run examples/chs_to_parquet.py --top-surge 20 --out data/top20_surge.parquet
    uv run examples/chs_to_parquet.py --min-surge 0.5 --min-wave 1.0
    uv run examples/chs_to_parquet.py --ids 65,66,71
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_CSV = os.path.join(ROOT, "CHS-SA_TS_SimB_Post0_SP1416_STWAVE14_Timeseries.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "EventDate_LC.parquet")


def load_chs(csv_path):
    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Storm ID"])
    df["Storm ID"] = df["Storm ID"].astype(int)
    return df


def storm_stats(df):
    """Per-storm summary: peak surge, peak Hm0, start date, row count."""
    g = df.groupby("Storm ID")
    stats = pd.DataFrame({
        "peak_surge": g["Water Elevation"].max(),
        "peak_hmo":   g["Zero Moment Wave Height"].max(),
        "start_date": g["yyyymmddHHMM"].min().astype(str).str.zfill(12),
        "rows":       g.size(),
    })
    stats["start_date"] = pd.to_datetime(
        stats["start_date"].astype(float).astype(int).astype(str).str.zfill(12),
        format="%Y%m%d%H%M",
    )
    return stats


def select_storms(df, args):
    stats = storm_stats(df)

    # Start with all storm IDs
    ids = set(stats.index.tolist())

    # Filter by specific IDs
    if args.ids:
        requested = {int(i) for i in args.ids.split(",")}
        missing = requested - set(stats.index)
        if missing:
            print(f"WARNING: storm IDs not found in CSV: {sorted(missing)}")
        ids &= requested

    # Filter by minimum thresholds
    if args.min_surge is not None:
        ids &= set(stats[stats["peak_surge"] >= args.min_surge].index)
    if args.max_surge is not None:
        ids &= set(stats[stats["peak_surge"] <= args.max_surge].index)
    if args.min_wave is not None:
        ids &= set(stats[stats["peak_hmo"] >= args.min_wave].index)

    # Top-N by surge or wave height (applied after threshold filters)
    if args.top_surge is not None:
        top = stats.loc[list(ids)].nlargest(args.top_surge, "peak_surge").index
        ids &= set(top)
    if args.top_wave is not None:
        top = stats.loc[list(ids)].nlargest(args.top_wave, "peak_hmo").index
        ids &= set(top)

    # Trim to first N (chronological) after all other filters
    if not ids:
        raise ValueError("No storms remain after applying filters.")

    selected_stats = stats.loc[sorted(ids)].sort_values("start_date")

    if args.min_spacing is not None:
        selected_stats = selected_stats.sort_values("start_date")
        kept, last_date = [], None
        for sid, row in selected_stats.iterrows():
            if last_date is None or (row["start_date"] - last_date).days >= args.min_spacing:
                kept.append(sid)
                last_date = row["start_date"]
        selected_stats = selected_stats.loc[kept]

    if args.n is not None:
        selected_stats = selected_stats.head(args.n)

    final_ids = selected_stats.index.tolist()

    # Print summary table
    print(f"\nSelected {len(final_ids)} storms:\n")
    print(f"  {'ID':>6}  {'Start date':<20}  {'Peak surge (m)':>14}  {'Peak Hm0 (m)':>12}  {'Rows':>5}")
    print(f"  {'──':>6}  {'──────────':<20}  {'──────────────':>14}  {'────────────':>12}  {'────':>5}")
    for sid in final_ids:
        s = selected_stats.loc[sid]
        print(f"  {sid:>6}  {str(s['start_date']):<20}  {s['peak_surge']:>14.3f}  {s['peak_hmo']:>12.3f}  {int(s['rows']):>5}")
    print()

    return df[df["Storm ID"].isin(final_ids)]


def convert(df, sim_start=None, spacing_days=None):
    """Convert selected CHS storms to the parquet schema.

    If *sim_start* and *spacing_days* are both provided, storm hydrographs are
    re-timed onto synthetic evenly-spaced dates::

        storm_i_start = sim_start + (i+1) * spacing_days

    The relative timing within each storm's hydrograph is preserved.
    This is required when the source data is a concurrent CHS ensemble
    (all storms simulated in the same short window) rather than sequential
    historical events.
    """
    origin = pd.Timestamp(sim_start) if sim_start else None
    rows = []
    for storm_idx, (storm_id, grp) in enumerate(df.groupby("Storm ID", sort=True)):
        grp = grp.sort_values("yyyymmddHHMM").reset_index(drop=True)
        orig_dates = pd.to_datetime(
            grp["yyyymmddHHMM"].astype(int).astype(str).str.zfill(12),
            format="%Y%m%d%H%M",
        )
        if origin is not None and spacing_days is not None:
            storm_start = origin + pd.Timedelta(days=(storm_idx + 1) * spacing_days)
            dates = storm_start + (orig_dates - orig_dates.iloc[0])
        else:
            dates = orig_dates

        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append({
                "lifecycle":        0,
                "storm_id":         str(storm_id),
                "hydro_tstp":       i,
                "date":             dates.iloc[i],
                "wave_height":      row["Zero Moment Wave Height"],
                "wave_peak_period": row["Peak Period"],
                "wave_direction":   row["Mean Wave Direction"],
                "water_elevation":  row["Water Elevation"],
                # Wind speed retained so storm_events.csv can derive Saffir-Simpson.
                # NOTE: recurrence_interval_yr and aep are NOT available in this
                # timeseries CSV — they require a separate JPM rates file keyed by
                # storm_id.  Populate those storm_events.csv columns from that file.
                "wind_speed_ms":    row.get("Wind Magnitude", float("nan")),
            })

    out = pd.DataFrame(rows)
    return out[[
        "lifecycle", "wave_peak_period", "wave_direction", "hydro_tstp",
        "storm_id", "water_elevation", "wave_height", "date", "wind_speed_ms",
    ]]


def main():
    parser = argparse.ArgumentParser(
        description="Convert CHS CSV to CSHORE parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",       default=DEFAULT_CSV,  help="CHS timeseries CSV")
    parser.add_argument("--out",       default=DEFAULT_OUT,  help="Output parquet path")
    parser.add_argument("--n",         type=int,             help="Keep first N after other filters")
    parser.add_argument("--top-surge", type=int, dest="top_surge", help="Top N by peak surge")
    parser.add_argument("--top-wave",  type=int, dest="top_wave",  help="Top N by peak Hm0")
    parser.add_argument("--min-surge",   type=float, dest="min_surge",   help="Min peak surge (m)")
    parser.add_argument("--max-surge",   type=float, dest="max_surge",   help="Max peak surge (m)")
    parser.add_argument("--min-wave",    type=float, dest="min_wave",    help="Min peak Hm0 (m)")
    parser.add_argument("--min-spacing",  type=int,   dest="min_spacing",  help="Min days between storm starts in source data (greedy; not useful for concurrent CHS ensembles)")
    parser.add_argument("--spacing-days", type=float, dest="spacing_days", help="Re-time storms onto synthetic dates: storm_i starts at sim_start + (i+1)*spacing_days")
    parser.add_argument("--sim-start",    default="2025-01-01", dest="sim_start", help="Simulation origin for --spacing-days (default: 2025-01-01)")
    parser.add_argument("--ids",          help="Comma-separated storm IDs e.g. 65,71,204")
    args = parser.parse_args()

    print(f"Input : {args.csv}")
    print(f"Output: {args.out}")
    if args.spacing_days:
        print(f"Re-timing: sim_start={args.sim_start}, spacing={args.spacing_days} days/storm")

    df  = load_chs(args.csv)
    sub = select_storms(df, args)
    out = convert(sub, sim_start=args.sim_start if args.spacing_days else None,
                  spacing_days=args.spacing_days)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")


if __name__ == "__main__":
    main()
