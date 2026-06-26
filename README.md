# erosion

The **erosion module** of CHART FEAT — a Python simulation framework for storm-driven erosion of ocean-facing sandy beaches, using CSHORE as the 1D cross-shore morphology engine. It couples a multi-decade lifecycle orchestrator with CSHORE per-storm response to support Coastal Storm Risk Management (CSRM) feasibility studies, and supports multi-reach, multi-alternative (FWOP/FWP), and multi-lifecycle Monte Carlo simulations.

## Structure

```
src/
  erosion/             # the erosion module (interval-loop simulation engine)
    reach.py           # run_lifecycle() — the lifecycle orchestrator
    interstorm.py      # Phase 1: background erosion + sea-level-change ticks
    storm.py           # Phase 2: parallel CSHORE driver + storm scheduling
    nourishment.py     # Phase 3: berm recovery + nourishment campaigns
    profile.py         # Profile data + profile events (StormResponse, Recovery, ...)
    metrics.py         # BeachFX morphology classification + 0-D metrics
    config.py          # ReachConfig, CSHOREParams, NourishmentConfig, ...
    units.py           # unit-aware config fields (ft/m, cy/m3)
    results.py         # ParquetResultsSink (output writer)
    geometry.py        # profile CSV loading
    viz.py             # plotting helpers
    runner/            # CSHORE execution boundary
      base.py          # CSHORERunner ABC, CSHOREResult
      local.py         # LocalCSHORERunner (subprocess)
      mock.py          # MockCSHORERunner (tests)
      cshore_io.py     # CSHORE infile generation + ODOC/OBPROF/OSETUP parsing
      vfall.py         # sediment fall-velocity (CSHORE wf input)
    pipeline.py      # main entry point (run-pipeline console script)
  executables/         # CSHORE binaries (macOS, Linux, Windows)
data/
  profiles/            # cross-shore profile CSVs (x ft, z ft, NAVD88)
  storms/              # storm forcing parquets (one row per hydrograph timestep)
examples/
  configs/             # example config files (ex1–ex4)
tests/                 # unit + integration test suite
```

## Architecture

Each lifecycle runs as an **interval loop** over the storm schedule
(`erosion/reach.py::run_lifecycle`). For every storm, three phases execute in order:

1. **Inter-storm** (`run_interstorm`) — apply background erosion and sea-level-change ticks from the previous storm up to this one.
2. **Storm response** (`run_parallel_cshore`) — run CSHORE for every profile in parallel (one subprocess per profile, capped at CPU count); interpolate each result back onto the profile's fixed grid.
3. **Campaign** (`run_campaign`) — post-storm berm recovery, then any triggered nourishment, scheduled by priority across profiles.

Profiles are pure data; physics is applied through small `ProfileEvent` types
(`StormResponse`, `Recovery`, `FullNourishment`, `PartialNourishment`, `ErosionTick`).
CSHORE is pluggable behind the `CSHORERunner` interface (`LocalCSHORERunner` for the
real binary, `MockCSHORERunner` for tests).

## Quick Start

The `run-pipeline` console script takes a config by **short name** (resolved
against `examples/configs/`) or by path:

```bash
uv run run-pipeline ex1    # single reach, single profile, FWOP
uv run run-pipeline ex2    # single reach, 3 profiles with priority ordering
uv run run-pipeline ex3    # 3 reaches, 2-3 profiles each
uv run run-pipeline ex4    # single reach, FWOP vs FWP (with nourishment)

# By path, and with a worker cap:
uv run run-pipeline examples/configs/ex3_multi_reach_multi_profile.json --workers 4

# Equivalent module form:
uv run python -m erosion ex1
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
  profiles.parquet        # all labeled snapshots — profile_id, label, t, node_idx, x, zb
  storm_hazard.parquet    # per-storm CSHORE output — profile_id, t_storm, node_idx, x, mwl, Hs, runup_m
  profile_metrics.parquet # 0-D morphology metrics per (profile, snapshot)
  profile_events.parquet  # per-(profile, snapshot) log + storm_response_type
  segment_events.csv      # nourishment events — event_type, profile_id, t_start, t_end, volume_m3, volume_cy
  run_metadata.json
  run_summary.txt
```

Snapshot labels follow the Beach-fx convention: `INIT`, `PreStorm`, `PostStorm`, `INUNDATION`, `RECS`, `REC`, `Pre-PDI`, `Post-PDI`, `SSN`, `ESN`, `SEN`, `EEN`, `Periodic`, `EndIteration`.

## Parallelism

Lifecycles within each (reach, alternative) pair run in parallel via Dask:

```bash
uv run run-pipeline ex3 --workers 4
```

Workers share the storms DataFrame in memory (thread mode — no serialisation overhead). Dask dashboard available at `http://localhost:8787` during the run.

## Dependencies

Managed by `uv`. Key packages: `numpy`, `pandas`, `pyarrow`, `pydantic`, `dask[distributed]`, `matplotlib`.

```bash
uv sync
```

## Running Tests

```bash
uv run pytest tests/                       # full suite (unit + integration)
uv run pytest tests/ -m "not integration"  # unit tests only
uv run pytest tests/ -m integration        # integration only — needs CSHORE binary in src/executables/
```
