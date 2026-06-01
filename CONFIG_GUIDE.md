# BeachFX-CSHORE Configuration Guide

All runtime inputs are declared in `config.json` at the project root.

## 1. Paths (`paths`)

| Key | Description |
|---|---|
| `data` | Root folder for input data files |
| `storms` | Path to storm forcing parquet file |
| `infiles` | Where generated CSHORE input files are written |
| `outfiles` | Where results and plots are written |

The CSHORE binary is selected automatically by platform from `src/executables/` — no config needed.

## 2. Profile (`profile`)

Each reach is a named entry with two fields:

```json
"profile": {
    "Reach1": { "d50": 0.3, "file": "data/Profile.csv" }
}
```

| Key | Description |
|---|---|
| `d50` | Median sediment grain size (mm) |
| `file` | Path to cross-shore profile CSV (feet, Beach-FX seaward-positive convention) |

Profile CSV format: two columns `x, z` (no header), values in feet. The loader converts to meters and reverses to CSHORE's landward-positive convention automatically. To add a new reach, add an entry here — no code changes needed.

## 3. CSHORE Physics (`cshore`)

| Key | Description |
|---|---|
| `dx` | Cross-shore grid spacing (m) |
| `gamma` | Breaking wave height-to-depth ratio (default 0.7) |
| `effb` | Suspension efficiency from wave breaking (default 0.002) |
| `efff` | Suspension efficiency from bottom friction (default 0.005) |
| `slp` | Suspended load parameter |
| `slpot` | Overtopping suspended load parameter |
| `tanphi` | Tangent of sediment friction angle |
| `blp` | Bedload parameter |
| `rwh` | Wave runup height parameter |
| `sporo` | Sediment porosity (typically 0.4) |
| `sg` | Specific gravity of sand (typically 2.65 for quartz) |
| `temp` | Water temperature (°C) for fall velocity calculation |
| `salin` | Salinity (ppt) |
| `fw` | Bed friction factor applied at every node |

## 4. Model Logic (`model_logic`)

CSHORE Fortran engine control flags.

| Key | Description |
|---|---|
| `iprofl` | Morphology update: `1.1` = active erosion/accretion, `0` = fixed bed |
| `iline` | Wave transformation mode |
| `isedav` | Sediment availability: `0` = unlimited, `1` = hard bottom |
| `iperm` | Permeability: `0` = impermeable, `1` = permeable |
| `iover` | Overtopping: `1` = on, `0` = off |
| `infilt` | Infiltration landward of dune crest: `1` = on |
| `iwtran` | Wave transmission through overtopping |
| `ipond` | Ponding landward of structure |
| `iwcint` | Wave-current interaction |
| `iroll` | Wave roller physics |
| `iwind` | Wind effects |
| `itide` | Tidal effect on currents |
| `ilab` | Boundary condition timing — **must be `0` for field conditions** |

## 5. Vegetation (`vegetation`)

| Key | Description |
|---|---|
| `enabled` | `false` disables vegetation entirely (default) |

When enabling vegetation, add the following fields:

```json
"vegetation": {
    "enabled": true,
    "Cd": 1.0,
    "n": 100.0,
    "dia": 0.01,
    "ht": 0.2,
    "rod": 0.1,
    "extent": [0.7, 1.0]
}
```

| Key | Description |
|---|---|
| `Cd` | Drag coefficient |
| `n` | Stem density (stems/m²) |
| `dia` | Stem diameter (m) |
| `ht` | Canopy height (m) |
| `rod` | Erosion depth below sand for stem failure (m) |
| `extent` | Fractional cross-shore extent of vegetation `[start, end]` |
