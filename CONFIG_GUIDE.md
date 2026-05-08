# BeachFX-CSHORE Configuration Guide

This guide explains the parameters found in `config.json`. Most values are converted from Imperial (US) to Metric (SI) during processing, but inputs are generally specified in the units familiar to coastal engineers.

## 1. Paths (`paths`)
Defines the directory structure for the workflow.
- **`data`**: Root folder for input profiles and storm data.
- **`infiles`**: Where the generated CSHORE input files are stored.
- **`outfiles`**: Where the HDF5 and final `.dat` result files are stored.
- **`executables`**: Location of the CSHORE binary (`.out` or `.exe`).

## 2. Profile Parameters (`profile`)
Defines the beach geometry.
- **`names`**: Unique identifiers for each reach (e.g., "Reach1").
- **`height_dune`**: Elevation of the dune crest (ft).
- **`width_dune`**: Width of the flat dune crest (ft).
- **`width_berm`**: Width of the berm (ft).
- **`width_upland`**: Width of the upland area (ft).
- **`height_upland`**: Elevation of the upland area (ft).
- **`slope_dune`**: Slope of the dune face (rise/run).
- **`height_berm`**: Elevation of the berm (ft).
- **`slope_foreshore`**: Slope of the foreshore/beach face (rise/run).
- **`d50`**: Median sediment grain size (mm).

## 3. CSHORE Physics (`cshore`)
Technical constants for the numerical model.
- **`dx`**: Constant grid spacing between calculation nodes (m).
- **`gamma`**: Shallow water ratio of wave height to water depth (breaking parameter). Default is 0.7.
- **`effb`**: Suspension efficiency due to breaking wave energy (eB). Standard USACE value is 0.002.
- **`efff`**: Suspension efficiency due to bottom friction (ef). Default is 0.005.
- **`slp`**: Suspended load parameter.
- **`slpot`**: Overtopping suspended load parameter.
- **`tanphi`**: Tangent of the sediment friction angle.
- **`blp`**: Bedload parameter.
- **`sporo`**: Sediment porosity (typically 0.4).
- **`sg`**: Specific gravity of sand grains (typically 2.65 for quartz).
- **`temp`**: Water temperature (Celsius) used for fall velocity calculation.
- **`salin`**: Salinity (ppt).
- **`fw`**: Bed friction factor applied at every node.

## 4. Tidal Configuration (`tide`)
- **`amp`**: Tidal amplitude (m).
- **`T`**: Tidal period (hours). Default 12.5 for semi-diurnal.
- **`phases`**: Numerical code for phase shift (1=High, 2=Falling, 3=Low, 4=Rising).

## 5. Model Logic (`model_logic`)
Control toggles for CSHORE's FORTRAN engine.
- **`iprofl`**: Toggle for morphology (1.1 = Run with erosion/accretion, 0 = Fixed).
- **`isedav`**: Sand availability (0 = Unlimited, 1 = Hard bottom/Limited).
- **`iover`**: Enable overtopping calculations (1 = On, 0 = Off).
- **`iperm`**: Permeability (1 = Permeable bottom, 0 = Impermeable).
- **`infilt`**: Include infiltration landward of dune crest (1 = Yes, 0 = No).
- **`iwtran`**: Wave transmission due to overtopping.
- **`iroll`**: Include wave roller physics (1 = Yes, 0 = No).
- **`iwind`**: Include wind effects.
- **`itide`**: Include tidal effect on currents (0 = No).
- **`ilab`**: Controls boundary condition timing. **Must be 0 for this workflow.**

## 6. Vegetation (`vegetation`)
- **`enabled`**: Boolean to toggle vegetation effects.
- **`Cd`**: Vegetation drag coefficient.
- **`n`**: Vegetation density.
- **`dia`**: Vegetation stem diameter (m).
- **`ht`**: Vegetation height (m).
- **`rod`**: Erosion limit below sand for vegetation failure.
- **`extent`**: Fraction of the domain covered by vegetation (e.g., [0.7, 1.0]).
