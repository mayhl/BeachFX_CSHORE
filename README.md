# erosion

The **erosion module** of CHART FEAT — a Python simulation framework for storm-driven erosion of ocean-facing sandy beaches, using CSHORE as the 1D cross-shore morphology engine. It couples a multi-decade lifecycle orchestrator with CSHORE per-storm response to support Coastal Storm Risk Management (CSRM) feasibility studies, and supports multi-reach, multi-alternative (FWOP/FWP), and multi-lifecycle Monte Carlo simulations.

## Orientation

One run expands into independent jobs, one per (reach × alternative × lifecycle), dispatched on a Dask process pool. Each job replays its lifecycle's storm schedule through an interval loop (`reach.py::Reach.run`): between storms the bed is lowered by background erosion and sea-level ticks; at each storm every profile gets its own CSHORE subprocess, and the resulting bed is interpolated back onto the profile's fixed grid; after each storm a campaign assesses each profile's sand deficit against its restore template, a reach-level decider compares the total to the volume trigger (or emergency dune thresholds), and a single serial crew places fill profile-by-profile at the production rate — everything the crew doesn't reach relaxes back toward its pre-storm shape over `T_recover` days. Calendar-driven nourishment cycles can also fire in quiet gaps. Every bed change appends a labeled snapshot; each job flushes parquet tables (beds, hydro, metrics, events, decisions) into `output/{reach}/{alt}/lc_NNNN/`.

## Structure

```
src/
  erosion/               # the erosion module (interval-loop simulation engine)
    pipeline.py          # config → jobs → Dask dispatch; the run-pipeline CLI
    reach.py             # Reach.run() — the interval loop; decides with `decision`, executes physics
    storm.py             # storm scheduling + parallel CSHORE driver + response classification
    interstorm.py        # background erosion + sea-level-change ticks between storms
    profile.py           # Profile data + profile events (StormResponse, Recovery, ...)
    decision/            # pure management-calendar planning (no physics imports)
      calendar.py        #   persistent state: cycle backlog + campaign carry-forward
      model.py           #   the decision vocabulary (frozen dataclasses)
      planner.py         #   the interval planner: campaign gate, crew-clock fold, cycle logic
      emit.py            #   the one decision→sink funnel
    nourishment/         # assess deficits, execute campaigns (the physics side)
      config.py          #   NourishmentConfig + geometry thresholds
      assess.py          #   Tier-1 assessors + parametric template synthesis
      campaign.py        #   work bundles + recovery application
      execute.py         #   the campaign executor (plays the planner's decisions)
    metrics/             # morphology fitting (berm/dune detection + idealized forms)
    runner/              # CSHORE execution boundary
      base.py            #   CSHORERunner ABC, CSHOREResult
      local.py           #   LocalCSHORERunner (subprocess)
      mock.py            #   MockCSHORERunner (tests)
      cshore_io.py       #   CSHORE infile generation + ODOC/OBPROF/OSETUP parsing (vendored)
      vfall.py           #   sediment fall-velocity (CSHORE wf input)
    config.py            # ReachConfig root model
    types.py             # shared enums: snapshot labels, campaign/decision kinds
    units.py             # unit-aware config fields (ft/m, cy/m3)
    results.py           # ParquetResultsSink (output writer)
    postprocess.py       # after-the-run registration pass (common grid + hydro)
    summary.py           # after-the-run derived metrics
    sweep.py             # gen-alternatives CLI (offline cartesian plan generator)
    geometry.py          # profile CSV loading
    viz.py               # plotting helpers
  executables/           # CSHORE binaries (macOS, Linux, Windows)
data/
  profiles/              # cross-shore profile CSVs (x ft, z ft, NAVD88)
  storms/                # storm forcing parquets (one row per hydrograph timestep)
examples/
  configs/               # example config files (ex1–ex4)
tests/                   # unit + integration test suite
```

## Architecture

Each lifecycle runs as an **interval loop** over the storm schedule
(`erosion/reach.py::Reach.run`). For every storm, the phases execute in order:

1. **Phase 1 — pre-storm erosion/SLC** (`run_interstorm`) — background erosion and sea-level-change ticks from the previous storm up to this one.
2. **Phase 1.5 — planned cycle** — a calendar-driven nourishment cycle due in the pre-storm gap.
3. **Phase 2 — storm response** (`run_parallel_cshore`) — run CSHORE for every profile in parallel (one subprocess per profile, capped at CPU count); interpolate each result back onto the profile's fixed grid.
4. **Phase 2.5 — gap erosion** (`run_interstorm` over [storm, next]) — pauses at a due cycle's date so the cycle assesses the beach as of its own date.
5. **Phase 3 — campaign** (`run_campaign`) — post-storm recovery plus any triggered nourishment, placed by a serial crew in priority order.
6. **Phase 3.5 — planned cycle** — a cycle due in the post-storm gap.

Decision-making and physics are separate layers. The `decision` package is a pure,
picklable planner — it sees only scalars, dates, and config, and returns decision
dataclasses (launch/skip/defer/interrupt); it never imports physics. The physics
side (`nourishment`, `storm`, `interstorm`, `profile`) executes those decisions.
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
uv run run-pipeline ex4    # single reach, FWOP vs FWP nourishment plans

# By path, with a worker cap, or running a subset of alternatives:
uv run run-pipeline examples/configs/ex3_multi_reach_multi_profile.json --workers 4
uv run run-pipeline ex4 --run FWP

# Equivalent module form:
uv run python -m erosion ex1
```

Output is written to `output/{reach}/{alternative}/lc_{lifecycle:04d}/`.

> **NOTE:** every relative path in a config (`paths.storms`, `paths.output`,
> profile CSVs) resolves against the config file's own directory, so a config
> and its data travel as one relocatable bundle; absolute paths pass through.

## Config Format

```json
{
  "paths":      { "storms": "../../data/storms/ex_storms_10s_45d.parquet", "output": "../../output/ex1" },
  "simulation": { "sim_start": "2025-01-01", "sim_end_days": 500 },
  "cshore":     { "d50": 0.3, "gamma": 0.7, "dx": 2.0, "iprofl": 1.1 },
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0, "hydro_dt": 1.0 },
      "erosion": { "type": "uniform", "rate": 8.2e-4, "tick_days": 30 },
      "slc":     { "rate": 1.1e-5, "tick_days": 30 }
    },
    "FWP": {
      "nourishment": {
        "volume_trigger": 25000,
        "production_rate": 500000,
        "template_geometry": { "berm_width": 200, "dune_height": 5.0 }
      }
    }
  },
  "reaches": {
    "Reach1": {
      "profiles": ["../../data/profiles/reach1_p0.csv"],
      "cshore":   { "d50": 0.3 }
    }
  }
}
```

The restore target is **parametric** (`template_geometry`: berm width/height,
foreshore slope, dune height/width/form) and is synthesized per profile on its
own grid; any field left unset falls back to the profile's measured as-built
geometry, so an empty `template_geometry` means restore-to-as-built. Nourishment
triggers: `volume_trigger` (reach-level deficit gate), `emergency_volume`
(per-profile forced mobilization), and `emergency_geometry` dune thresholds
(dune-aware assessors); at least one must be active. Calendar-driven cycles
(`cycle_interval_years`, `cycle_start_date`) propose on schedule and the volume
gate disposes, so a healthy beach skips its cycle.

### Nourishment plan table (schema v2)

Instead of spelling alternatives out, a top-level `nourishment` table keyed by
plan id can be referenced per reach — each referenced plan becomes an
alternative, and shared `storm`/`erosion`/`slc` sections are hoisted to the
global level (see `ex4`):

```json
{
  "storm":   { "T_recover": 21.0 },
  "nourishment": {
    "FWOP": { "volume_trigger": 50000, "production_rate": 500000 },
    "FWP":  { "volume_trigger": 25000, "production_rate": 500000 }
  },
  "reaches": {
    "Reach1": { "profiles": ["../../data/profiles/reach1_p0.csv"], "nourishment": ["FWOP", "FWP"] }
  }
}
```

A top-level `"run"` key (or the `--run` CLI flag, e.g. `--run FWP` or
`--run '1-4,8'`) selects which alternative ids to execute; the default runs all.
The `gen-alternatives` console script (`sweep.py`) generates cartesian plan
tables offline.

### Parameter override hierarchy

CSHORE parameters merge in three layers — each layer overrides only the keys it specifies:

```
global "cshore"          ← shared defaults (dx, gamma, iprofl, ...)
  └── reach "cshore"     ← per-reach overrides (e.g. d50, effb)
        └── alt "cshore" ← per-alternative overrides (rare)
```

The other sections (`storm`, `erosion`, `slc`, `nourishment`) follow the same
pattern independently, whether declared globally or per alternative — FWP only
needs to declare what differs from FWOP. Keys absent at a lower level fall
through to the level above.

## Unit Awareness

The framework defaults to **feet** for all user-facing length and volume fields. CSHORE's internal grid (dx, d50) always uses its native SI units regardless of the global setting.

### Minimal config — all defaults (feet)

No `units` key needed. All dimensional length fields are interpreted as feet. Nourishment volumes are **3-D** (cy for the reach segment total), not per-unit-width — the scaling from CSHORE's 2-D output to 3-D uses `longshore_width`.

```json
{
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0 },
      "erosion": { "type": "uniform", "rate": 8.2e-4, "tick_days": 30 }
    },
    "FWP": {
      "nourishment": {
        "volume_trigger":  90000,
        "production_rate": 1500000,
        "template_geometry": { "berm_width": 200, "dune_height": 6.5 }
      }
    }
  },
  "reaches": {
    "Reach1": {
      "longshore_width": 1000,
      "profiles": ["../../data/profiles/reach1_p0.csv"]
    }
  }
}
```

`template_geometry` lengths in feet. `volume_trigger` = 90,000 cy (3-D reach total). `production_rate` = 1,500,000 cy/yr. `longshore_width` = 1,000 ft.

### Switching to meters globally

Add `"units": {"input": "m"}` at the top level. Length fields interpret bare numbers as meters; nourishment volumes become m³ (3-D reach total) and rates become m³/day.

```json
{
  "units": { "input": "m" },
  "alternatives": {
    "FWOP": {
      "storm":   { "T_recover": 21.0 },
      "erosion": { "type": "uniform", "rate": 2.5e-4, "tick_days": 30 }
    },
    "FWP": {
      "nourishment": {
        "volume_trigger":  75000,
        "production_rate": 3300,
        "template_geometry": { "berm_width": 60, "dune_height": 2.0 }
      }
    }
  },
  "reaches": {
    "Reach1": {
      "longshore_width": 305,
      "profiles": ["../../data/profiles/reach1_p0.csv"]
    }
  }
}
```

`volume_trigger` = 75,000 m³. `production_rate` = 3,300 m³/day (bare float interpreted as m³/day when `input_units=m`). `longshore_width` = 305 m.

### Explicit per-field units (override global)

Any field can take `{"value": ..., "units": "..."}` to override the global setting for that field alone. Useful when mixing data sources.

```json
{
  "units": { "input": "m" },
  "nourishment": {
    "volume_trigger":  { "value": 90000,   "units": "cy" },
    "production_rate": { "value": 1500000, "units": "cy/yr" },
    "template_geometry": {
      "berm_width": { "value": 200, "units": "ft" }
    }
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
    "profiles": ["../../data/profiles/reach1_p0.csv", "../../data/profiles/reach1_p1.csv"]
  }
}
```

```json
"reaches": {
  "Reach1": {
    "longshore_width": [500, 500],
    "profiles": ["../../data/profiles/reach1_p0.csv", "../../data/profiles/reach1_p1.csv"]
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
output/
  storm_events.csv          # run-level storm log across lifecycles
  {reach}/{alternative}/lc_{lc:04d}/
    profiles.parquet        # all labeled snapshots — profile_id, label, t, node_idx, x, zb
    storm_hazard.parquet    # per-storm CSHORE output — profile_id, t_storm, node_idx, x, mwl, Hs, runup_m
    profile_metrics.parquet # 0-D morphology metrics per (profile, snapshot)
    snapshots.parquet  # lightweight snapshot log — one row per (profile, snapshot)
    events.parquet          # append-only profile event log (the audit trail of applied physics)
    decisions.parquet       # reach-scope decisions — launches, skips, defers, interrupts
    placements.csv      # nourishment placement events — profile_id, t_start, t_end, volumes
    warnings.csv            # surfaced run warnings (e.g. CSHORE failures)
    run_metadata.json
    run_summary.txt
```

Every parquet carries the full `ReachConfig` and run identity in its key-value
footer (`results.read_parquet_footer`), so outputs are self-describing.

Snapshot labels follow the Beach-fx convention: `INIT`, `PreStorm`, `PostStorm`,
`INUNDATION`, `RECS`, `REC`, `RECN`, `Pre-PDI`, `Post-PDI`, `SSN`, `ESN`, `SEN`,
`EEN`, `EENS`, `ESNS`, `Periodic`, `EndIteration`. The nourishment pairs mark
segment start/end: `SEN`/`EEN` = storm-triggered campaign, `SSN`/`ESN` = planned
cycle, with a trailing `S` (`EENS`, `ESNS`) when a storm cut the placement short.

## Parallelism

Jobs — one per (reach, alternative, lifecycle) — run in parallel on a Dask
process pool; each worker receives only its own lifecycle's storm slice:

```bash
uv run run-pipeline ex3 --workers 4
```

Within each storm, CSHORE subprocesses run per profile on a thread pool capped
at CPU count. Dask dashboard available at `http://localhost:8787` during the run.

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
