# BeachFX-CSHORE

Python pipeline for running [CSHORE](https://github.com/CSHORE) as the morphology model in Beach-FX. Supports chained storm simulation, where each storm's final bed profile becomes the initial condition for the next.

## Structure

```
examples/        # entry point scripts
src/
  models/        # CshoreAdapter, ChainedPolicy, IO
  utils/         # profile loading, storm parsing, sediment
  executables/   # CSHORE binaries (Linux, macOS, Windows)
data/            # input profiles, storm forcing, outputs
```

## Quick Start

### Native (macOS / Linux / Windows)

```bash
uv run examples/run_cshore.py
```

### Docker

```bash
docker compose up --build
```

## Configuration

All inputs are declared in `config.json`:

```json
{
  "paths": {
    "storms": "data/EventDate_LC.parquet",
    "outfiles": "data/outfiles"
  },
  "profile": {
    "Reach1": { "d50": 0.3, "file": "data/Profile.csv" }
  }
}
```

## Outputs

Results are written to `data/outfiles/<reach>/`:
- `<reach>_master.parquet` — storm chain results (initial/final profile per storm)
- `*_chain_profiles.png` — profile evolution plot
- `*_chain_differences.png` — per-storm bed level change plot

## Dependencies

Managed by `uv`. Key packages: `numpy`, `pandas`, `h5py`, `pyarrow`, `matplotlib`.

```bash
uv sync
```
