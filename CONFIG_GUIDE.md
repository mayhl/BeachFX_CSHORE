# CSHORE Parameter Reference

Run configuration (paths, simulation window, alternatives, reaches, units) is
documented in the [README](README.md#config-format). This page is the reference
for the `cshore` section only — every key lives flat under `cshore` (globally,
per reach, or per alternative; see the README's override hierarchy) and maps
onto `CSHOREParams` (`src/erosion/runner/local.py`), which carries the defaults
listed here.

The CSHORE binary is selected automatically by platform from `src/executables/`
— no config needed.

## Calibration parameters

| Key | Default | Description |
|---|---|---|
| `d50` | 0.3 | Median sediment grain size (mm); also the reach-wide default for profile init |
| `blp` | 0.001 | Bedload parameter |
| `effb` | 0.002 | Suspension efficiency from wave breaking |
| `gamma` | 0.7 | Breaking wave height-to-depth ratio |
| `efff` | 0.005 | Suspension efficiency from bottom friction |
| `slp` | 0.5 | Suspended load parameter |
| `slpot` | 0.1 | Overtopping suspended load parameter |
| `tanphi` | 0.63 | Tangent of sediment friction angle |
| `dx` | 1.0 | Cross-shore grid spacing (m) — CSHORE-native units, immune to the global `units` setting |
| `rwh` | 0.02 | Numerical runup wire height (m) |
| `fw` | 0.015 | Bed friction factor applied at every node |

## Physical constants (rarely changed)

| Key | Default | Description |
|---|---|---|
| `sg` | 2.65 | Specific gravity of sand (quartz) |
| `sporo` | 0.4 | Sediment porosity |
| `temp` | 20.0 | Water temperature (°C) for fall-velocity calculation |
| `salin` | 0.0 | Salinity (ppt) |

## Model logic flags

CSHORE Fortran engine control flags — same section, same flat keys.

| Key | Default | Description |
|---|---|---|
| `iprofl` | 1.1 | Morphology update: `1.1` = active erosion/accretion, `0` = fixed bed |
| `iline` | 1 | Wave transformation mode |
| `isedav` | 0 | Sediment availability: `0` = unlimited, `1` = hard bottom |
| `iperm` | 0 | Permeability: `0` = impermeable, `1` = permeable |
| `iover` | 1 | Overtopping: `1` = on, `0` = off |
| `infilt` | 0 | Infiltration landward of dune crest: `1` = on |
| `iwtran` | 0 | Wave transmission through overtopping |
| `ipond` | 0 | Ponding landward of structure |
| `iwcint` | 0 | Wave-current interaction |
| `iroll` | 0 | Wave roller physics |
| `iwind` | 0 | Wind effects |
| `itide` | 0 | Tidal effect on currents |
| `ilab` | 0 | Boundary condition timing — **must be `0` for field conditions** |

## Profile CSVs

Profile files are listed per reach (`reaches.<name>.profiles`). Format: two
columns `x, z` (no header), values in feet, Beach-FX seaward-positive
convention. The loader converts to meters and reverses to CSHORE's
landward-positive convention automatically.

> **NOTE:** vegetation is not currently exposed through config — the runner
> hardcodes it off (`_build_cshore_config`, `src/erosion/runner/local.py`).
