# BeachFX-CSHORE

Python simulation framework for coastal Beach-FX studies using CSHORE as the cross-shore morphology engine. Supports multi-reach, multi-alternative (FWOP/FWP), and multi-lifecycle Monte Carlo simulations driven by a structured event queue.

## Structure

```
src/
  framework/       # event-driven simulation engine
    events/        # StormEvent, RecoveryEvent, NourishmentEvent, BackgroundErosion, SLC
    runner/        # LocalCSHORERunner, MockCSHORERunner
    config.py      # ReachConfig, CSHOREParams, NourishmentConfig, ...
    queue_builder.py
    reach.py
    results.py
  models/          # legacy ChainedPolicy / CshoreAdapter (run_cshore.py)
  utils/           # profile loading, geometry, sediment helpers
  executables/     # CSHORE binaries (macOS, Linux, Windows)
data/
  profiles/        # cross-shore profile CSVs (x ft, z ft, NAVD88)
  storms/          # storm forcing parquets (one row per hydrograph timestep)
examples/
  run_pipeline.py  # main entry point — multi-reach × alternative × lifecycle
  configs/         # example config files (ex1–ex4)
tests/             # unit + integration test suite
```

## Quick Start

```bash
# Ex1 — single reach, single profile, FWOP
uv run examples/run_pipeline.py examples/configs/ex1_single_reach_single_profile.json

# Ex2 — single reach, 3 profiles with priority ordering, FWOP
uv run examples/run_pipeline.py examples/configs/ex2_single_reach_multi_profile.json

# Ex3 — 3 reaches, 2-3 profiles each, FWOP
uv run examples/run_pipeline.py examples/configs/ex3_multi_reach_multi_profile.json

# Ex4 — single reach, 3 profiles, FWOP vs FWP (with nourishment)
uv run examples/run_pipeline.py examples/configs/ex4_multi_profile_multi_alt.json

# Parallel workers (default: cpu_count, capped at lifecycle count)
uv run examples/run_pipeline.py examples/configs/ex3_multi_reach_multi_profile.json --workers 4
```

Output is written to `output/{reach}/{alternative}/lc_{lifecycle:04d}/`.

## Config Format

```json
{
  "paths":      { "storms": "data/storms/ex_storms_10s_45d.parquet", "output": "output/ex1" },
  "simulation": { "sim_start": "2025-01-01", "sim_end_days": 500 },
  "cshore":     { "d50": 0.3, "gamma": 0.7, "dx": 2.0, "iprofl": 1.1 },
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0, "hydro_dt": 1.0 },
      "erosion": { "type": "uniform", "rate": 8.2e-4, "interval": 30 },
      "slc":     { "rate": 1.1e-5, "interval": 30 }
    },
    "FWP": {
      "nourishment": {
        "template_x": [0, 100, 120, 140, 200, 370, 800, 1600],
        "template_z": [1.8, 1.8, 2.4, 2.4, 1.2, -1.5, -5.5, -9.3],
        "volume_trigger": 25000,
        "production_rate": 5000,
        "evaluation_interval": 365
      }
    }
  },
  "reaches": {
    "Reach1": {
      "profiles": ["data/profiles/reach1_p0.csv"],
      "cshore":   { "d50": 0.3 }
    }
  }
}
```

### Parameter override hierarchy

CSHORE parameters merge in three layers — each layer overrides only the keys it specifies:

```
global "cshore"          ← shared defaults (dx, gamma, iprofl, ...)
  └── reach "cshore"     ← per-reach overrides (e.g. d50, effb)
        └── alt "cshore" ← per-alternative overrides (rare)
```

Alternative sub-dicts (`storm`, `erosion`, `slc`, `nourishment`) follow the same pattern independently — FWP only needs to declare what differs from FWOP. Keys absent at a lower level fall through to the level above.

## Unit Awareness

The framework defaults to **feet** for all user-facing length and volume fields. CSHORE's internal grid (dx, d50) always uses its native SI units regardless of the global setting.

### Minimal config — all defaults (feet)

No `units` key needed. All dimensional length fields are interpreted as feet. Nourishment volumes are **3-D** (cy for the reach segment total), not per-unit-width — the scaling from CSHORE's 2-D output to 3-D uses `longshore_width`.

```json
{
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0 },
      "erosion": { "type": "uniform", "rate": 8.2e-4, "interval": 30 }
    },
    "FWP": {
      "nourishment": {
        "template_x":      [0, 328, 394, 459, 656, 1214, 2625, 5249],
        "template_z":      [5.9, 5.9, 7.9, 7.9, 3.9, -4.9, -18.0, -30.5],
        "volume_trigger":  90000,
        "production_rate": 1500000,
        "evaluation_interval": 365
      }
    }
  },
  "reaches": {
    "Reach1": {
      "longshore_width": 1000,
      "profiles": ["data/profiles/reach1_p0.csv"]
    }
  }
}
```

`template_x`/`template_z` in feet. `volume_trigger` = 90,000 cy (3-D reach total). `production_rate` = 1,500,000 cy/yr. `longshore_width` = 1,000 ft.

### Switching to meters globally

Add `"units": {"input": "m"}` at the top level. Length fields interpret bare numbers as meters; nourishment volumes become m³ (3-D reach total) and rates become m³/day.

```json
{
  "units": { "input": "m" },
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0 },
      "erosion": { "type": "uniform", "rate": 2.5e-4, "interval": 30 }
    },
    "FWP": {
      "nourishment": {
        "template_x":      [0, 100, 120, 140, 200, 370, 800, 1600],
        "template_z":      [1.8, 1.8, 2.4, 2.4, 1.2, -1.5, -5.5, -9.3],
        "volume_trigger":  75000,
        "production_rate": 1200000,
        "evaluation_interval": 365
      }
    }
  },
  "reaches": {
    "Reach1": {
      "longshore_width": 305,
      "profiles": ["data/profiles/reach1_p0.csv"]
    }
  }
}
```

`volume_trigger` = 75,000 m³. `production_rate` = 1,200,000 m³/yr (bare float interpreted as m³/yr when `input_units=m`). `longshore_width` = 305 m.

### Explicit per-field units (override global)

Any field can take `{"value": ..., "units": "..."}` to override the global setting for that field alone. Useful when mixing data sources.

```json
{
  "units": { "input": "m" },
  "nourishment": {
    "template_x":      [0, 100, 120, 200, 800],
    "template_z":      [1.8, 1.8, 2.4, 1.2, -5.5],
    "volume_trigger":  { "value": 90000,   "units": "cy" },
    "production_rate": { "value": 1500000, "units": "cy/yr" },
    "evaluation_interval": 365
  },
  "reaches": {
    "Reach1": {
      "longshore_width": { "value": 1000, "units": "ft" }
    }
  }
}
```

### CSHORE parameters are context-immune

`d50` (mm) and `dx` (m) always use CSHORE native units — the global `input` setting has no effect on them:

```json
{
  "units": { "input": "ft" },
  "cshore": {
    "d50": 0.3,
    "dx":  2.0
  }
}
```

`d50 = 0.3 mm`, `dx = 2.0 m` regardless of unit context.

### Reach-level longshore width

`longshore_width` on a reach converts 2D CSHORE volume (m²/m or cy/ft) to 3D totals (m³ or cy) in reports. A scalar divides evenly across profiles; a list assigns widths explicitly.

```json
"reaches": {
  "Reach1": {
    "longshore_width": 1000,
    "profiles": ["data/profiles/reach1_p0.csv", "data/profiles/reach1_p1.csv"]
  }
}
```

```json
"reaches": {
  "Reach1": {
    "longshore_width": [500, 500],
    "profiles": ["data/profiles/reach1_p0.csv", "data/profiles/reach1_p1.csv"]
  }
}
```

## Storm Parquet Format

One row per hydrograph timestep:

| column | type | description |
|--------|------|-------------|
| `lifecycle` | int | Monte Carlo lifecycle index |
| `storm_id` | str | unique storm identifier |
| `hydro_tstp` | int | timestep index within storm |
| `date` | datetime | UTC timestamp |
| `wave_height` | float | Hm0 (m) |
| `wave_peak_period` | float | Tp (s) |
| `wave_direction` | float | degrees |
| `water_elevation` | float | still water level (m NAVD) |

Convert a CHS ensemble CSV to parquet:

```bash
uv run examples/chs_to_parquet.py \
  --top-surge 10 --max-surge 2.0 \
  --spacing-days 45 --sim-start 2025-01-01 \
  --out data/storms/my_storms.parquet
```

## Output Layout

```
output/{reach}/{alternative}/lc_{lc:04d}/
  snapshots.parquet      # profile_id, label, t, node_idx, x, zb, zbe
  profiles.parquet       # final profile state per profile
  profile_events.parquet # per-profile event log
  reach_events.csv       # reach-level event log
  run_metadata.json
  run_summary.txt
```

Snapshot labels follow the Beach-FX convention: `INIT`, `PreStorm`, `PostStorm`, `REC`, `SSN`, `ESN`, `SEN`, `EEN`, `EndIteration`.

## Parallelism

Lifecycles within each (reach, alternative) pair run in parallel via Dask:

```bash
uv run examples/run_pipeline.py config.json --workers 4
```

Workers share the storms DataFrame in memory (thread mode — no serialisation overhead). Dask dashboard available at `http://localhost:8787` during the run.

## Dependencies

Managed by `uv`. Key packages: `numpy`, `pandas`, `pyarrow`, `pydantic`, `dask[distributed]`, `matplotlib`.

```bash
uv sync
```

## Running Tests

```bash
uv run pytest tests/                   # unit tests only
uv run pytest tests/ -m integration   # requires CSHORE binary in src/executables/
```
