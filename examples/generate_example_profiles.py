#!/usr/bin/env python3
"""Generate synthetic beach profile CSVs for example runs.

Profiles are written in Beach-FX / CHS convention:
  - No header row
  - Two columns: x_ft (cross-shore, seaward-positive), z_ft (elevation NAVD88)
  - x=0 is the landward edge; x increases offshore

Profile structure (landward → seaward in CHS convention)::

    [upland flat] → [dune back] → [dune crest] → [dune front] → [berm flat]
    → [beach face] → [shoreline] → [nearshore+bar] → [offshore]

Three BeachFX morphology types are generated (Tech Ref §7.3.1):

  LOW_UPLAND  — DE > BE > UE   typical barrier island with dune above berm above upland
  LOW_BERM    — DE > UE ≥ BE   dune present, but berm at or below upland elevation
  HIGH_UPLAND — UE ≥ DE        high upland (bluff); no dune relief above the upland

Usage:
    python examples/generate_example_profiles.py
    python examples/generate_example_profiles.py --out-dir data/profiles
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "data", "profiles")

FT_TO_M = 0.3048


def make_profile(
    *,
    # Upland
    upland_z_ft: float,
    upland_width_ft: float = 400.0,
    # Dune  (set dune_back_width_ft=0 and dune_height_ft=upland_z_ft for HIGH_UPLAND)
    dune_height_ft: float,
    dune_back_width_ft: float = 80.0,
    dune_front_width_ft: float = 50.0,
    # Berm
    berm_z_ft: float,
    berm_width_ft: float = 100.0,
    # Beach and offshore
    beach_face_slope: float = 0.10,
    nearshore_slope: float = 0.025,
    offshore_slope: float = 0.012,
    bar_z_ft: float = 2.0,
    bar_width_ft: float = 80.0,
    bar_offset_ft: float = 250.0,
    total_length_ft: float = 5200.0,
    dx_ft: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_ft, z_ft) in CHS convention (x=0 landward, x increases seaward)."""
    x = np.arange(0.0, total_length_ft + dx_ft, dx_ft)
    z = np.zeros_like(x)
    c = 0.0  # cursor (seaward position)

    # 1. Upland flat
    z[x <= upland_width_ft] = upland_z_ft
    c = upland_width_ft

    # 2. Dune back slope: upland_z → dune_height  (skipped for HIGH_UPLAND)
    if dune_back_width_ft > 0 and dune_height_ft > upland_z_ft:
        mask = (x > c) & (x <= c + dune_back_width_ft)
        z[mask] = upland_z_ft + (dune_height_ft - upland_z_ft) * (x[mask] - c) / dune_back_width_ft
        c += dune_back_width_ft

    # 3. Dune front: dune_height → berm_z
    dune_top = dune_height_ft if dune_height_ft > upland_z_ft else upland_z_ft
    mask = (x > c) & (x <= c + dune_front_width_ft)
    z[mask] = dune_top - (dune_top - berm_z_ft) * (x[mask] - c) / dune_front_width_ft
    c += dune_front_width_ft

    # 4. Berm flat
    mask = (x > c) & (x <= c + berm_width_ft)
    z[mask] = berm_z_ft
    c += berm_width_ft

    # 5. Beach face: berm_z → 0
    beach_len = berm_z_ft / beach_face_slope
    mask = (x > c) & (x <= c + beach_len)
    z[mask] = berm_z_ft - beach_face_slope * (x[mask] - c)
    sl_x = c + beach_len  # shoreline position
    c = sl_x

    # 6. Nearshore: gentle slope + Gaussian bar
    bar_crest_x = c + bar_offset_ft
    ns_end = bar_crest_x + bar_width_ft
    mask = (x > c) & (x <= ns_end)
    x_ns = x[mask]
    z_base = -nearshore_slope * (x_ns - c)
    bar = bar_z_ft * np.exp(-0.5 * ((x_ns - bar_crest_x) / (bar_width_ft / 3.0)) ** 2)
    z[mask] = z_base + bar
    c = ns_end

    # 7. Offshore: continue slope
    z_off0 = z[x <= c][-1]
    mask = x > c
    z[mask] = z_off0 - offshore_slope * (x[mask] - c)

    return x, z


def write_csv(path: str, x: np.ndarray, z: np.ndarray, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for xi, zi in zip(x, z):
            f.write(f"{xi:.2f},{zi:.2f}\n")
    print(f"    {label:35s}  {len(x):4d} nodes  → {os.path.relpath(path)}")


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    print(
        f"  {'Profile':<12}  {'Type':<14}  {'UE':>6}  {'DE':>6}  {'BE':>6}  {'BW':>6}  {'berm_elevation (m)':>20}"
    )
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 20}")

    def row(pid, morph, ue, de, be, bw):
        print(
            f"  {pid:<12}  {morph:<14}  {ue:>5.1f}'  {de:>5.1f}'  {be:>5.1f}'  {bw:>5.0f}'  {be * FT_TO_M:>17.3f} m"
        )

    # ------------------------------------------------------------------
    # Reach 1 — one profile per morphology type
    # ------------------------------------------------------------------
    print("\nReach 1 — one profile per BeachFX morphology type")

    # p0: LOW_UPLAND  (DE=14ft > BE=8ft > UE=6ft)
    x, z = make_profile(
        upland_z_ft=6.0,
        dune_height_ft=14.0,
        dune_back_width_ft=80.0,
        dune_front_width_ft=50.0,
        berm_z_ft=8.0,
        berm_width_ft=120.0,
        beach_face_slope=0.10,
    )
    write_csv(os.path.join(out_dir, "reach1_p0.csv"), x, z, "reach1_p0  LOW_UPLAND")
    row("reach1_p0", "LOW_UPLAND", 6.0, 14.0, 8.0, 120)

    # p1: LOW_BERM  (DE=14ft > UE=9ft, BE=6ft ≤ UE)
    x, z = make_profile(
        upland_z_ft=9.0,
        dune_height_ft=14.0,
        dune_back_width_ft=60.0,
        dune_front_width_ft=40.0,
        berm_z_ft=6.0,
        berm_width_ft=80.0,
        beach_face_slope=0.10,
    )
    write_csv(os.path.join(out_dir, "reach1_p1.csv"), x, z, "reach1_p1  LOW_BERM")
    row("reach1_p1", "LOW_BERM", 9.0, 14.0, 6.0, 80)

    # p2: HIGH_UPLAND  (UE=15ft ≥ DE=15ft — bluff, no distinct dune above upland)
    x, z = make_profile(
        upland_z_ft=15.0,
        dune_height_ft=15.0,
        dune_back_width_ft=0.0,
        dune_front_width_ft=60.0,
        berm_z_ft=8.0,
        berm_width_ft=100.0,
        beach_face_slope=0.10,
    )
    write_csv(os.path.join(out_dir, "reach1_p2.csv"), x, z, "reach1_p2  HIGH_UPLAND")
    row("reach1_p2", "HIGH_UPLAND", 15.0, 15.0, 8.0, 100)

    # ------------------------------------------------------------------
    # Reach 2 — LOW_UPLAND variants, steeper beach (higher wave energy)
    # ------------------------------------------------------------------
    print("\nReach 2 — LOW_UPLAND, steeper beach face")

    for pid, bw, slope in [("p0", 120, 0.14), ("p1", 90, 0.13), ("p2", 60, 0.12)]:
        x, z = make_profile(
            upland_z_ft=6.0,
            dune_height_ft=13.0,
            dune_back_width_ft=70.0,
            dune_front_width_ft=45.0,
            berm_z_ft=7.5,
            berm_width_ft=bw,
            beach_face_slope=slope,
        )
        write_csv(os.path.join(out_dir, f"reach2_{pid}.csv"), x, z, f"reach2_{pid}  LOW_UPLAND")
        row(f"reach2_{pid}", "LOW_UPLAND", 6.0, 13.0, 7.5, bw)

    # ------------------------------------------------------------------
    # Reach 3 — LOW_BERM variants, dissipative (gentle slopes)
    # ------------------------------------------------------------------
    print("\nReach 3 — LOW_BERM, dissipative beach")

    for pid, bw, ue, de in [("p0", 80, 8.0, 12.0), ("p1", 60, 9.0, 11.0)]:
        x, z = make_profile(
            upland_z_ft=ue,
            dune_height_ft=de,
            dune_back_width_ft=60.0,
            dune_front_width_ft=40.0,
            berm_z_ft=6.0,
            berm_width_ft=bw,
            beach_face_slope=0.07,
            offshore_slope=0.009,
        )
        write_csv(os.path.join(out_dir, f"reach3_{pid}.csv"), x, z, f"reach3_{pid}  LOW_BERM")
        row(f"reach3_{pid}", "LOW_BERM", ue, de, 6.0, bw)

    n_total = 3 + 3 + 2
    print(f"\nDone — {n_total} profile CSVs written to {out_dir}")
    print("\nSuggested berm_elevation values for configs (use units: {input: m}):")
    print("  LOW_UPLAND  reach1_p0, reach2_*  →  berm_elevation: 2.4   (≈ 8 ft)")
    print("  LOW_BERM    reach1_p1, reach3_*  →  berm_elevation: 1.8   (≈ 6 ft)")
    print("  HIGH_UPLAND reach1_p2            →  berm_elevation: 2.4   (≈ 8 ft)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out_dir)
