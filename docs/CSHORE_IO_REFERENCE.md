# CSHORE Input/Output Reference

This document catalogs the parameters used by the CSHORE engine and how they are handled by the Python I/O scripts (`cshoreIO.py`).

## 1. Input Parameters (`infile`)

CSHORE is hardcoded to read a file named `infile`. The Python scripts generate descriptive names and then rename/copy them to `infile` at runtime.

### Logical Control Toggles (Integers)
| Parameter | Description |
| :--- | :--- |
| `ILINE` | Toggle for single vs. multiple line calculations. |
| `IPROFL` | Profile evolution (0 = Fixed, 1 = Mobile/Erosion). |
| `ISEDAV` | Sand availability (0 = Unlimited, 1 = Hard bottom). |
| `IPERM` | Permeability of the bed (0 = Impermeable, 1 = Permeable). |
| `IOVER` | Overtopping toggle (0 = No, 1 = Yes). |
| `IWTRAN` | Wave transmission due to overtopping. |
| `IPOND` | Ponding seaward of SWL. |
| `INFILT` | Infiltration landward of dune crest. |
| `IROLL` | Roller physics (0 = No, 1 = Yes). |
| `IWIND` | Wind effects. |
| `ITIDE` | Tidal effects on currents. |
| `IVEG` | Vegetation effects (0 = No, 1 = Yes). |

### Physical Constants (Floats)
| Parameter | Unit | Description |
| :--- | :--- | :--- |
| `DXC` | m | Constant grid spacing for numerical nodes. |
| `GAMMA` | - | Wave breaking parameter (H/h ratio). |
| `D50` | mm | Median sediment grain size. |
| `WF` | m/s | Sediment fall velocity. |
| `SG` | - | Specific gravity of sediment (typically 2.65). |
| `TEMP` | C | Water temperature. |
| `SALIN` | ppt | Salinity. |

### Boundary Conditions (Time-Series)
| Parameter | Unit | Description |
| :--- | :--- | :--- |
| `Time` | s | Elapsed time since simulation start. |
| `Tp` | s | Peak spectral wave period. |
| `Hrms` | m | Root-mean-square wave height. |
| `SWL` | m | Still water level elevation (Surge + Tide). |
| `Angle` | deg | Peak wave angle (shore-normal = 0). |

---

## 2. Output Parameters

CSHORE generates several text files prefixed with `O`.

### `ODOC` (Summary Data)
*   **Runup**: `R2P` (2% exceedance) and `R_mean` (mean).
*   **Transition Points**: Node numbers for `JDRY` (shoreline), `JSWL` (water intersection), and `JR` (breaking point).

### `OBPROF` (Morphology Data)
*   **`Initial Profile`**: Bed elevation at $t=0$ (post-smoothing).
*   **`Final Profile`**: Bed elevation after the storm.
*   **`Min/Max Profile`**: The "envelope" of maximum scour and accretion.

### `OSETUP` (Hydrodynamic Data)
*   **`Setup`**: Wave-induced increase in mean water level.
*   **`Depth`**: Total water depth (Setup + Surge + Bathymetry).
*   **`Max Water Elev`**: Highest elevation reached by Surge + Setup.
*   **`Max Wave Ht`**: Largest wave height experienced at each node.

---

## 3. Python Transformation

The Python scripts transform these raw outputs into formats suitable for further analysis:
1.  **HDF5 (`.h5`)**: Used for efficient storage of thousands of Monte Carlo iterations.
2.  **DAT (`.dat`)**: The final formatted text file used for import into Beach-fx.
3.  **Unit Conversion**: Results are typically converted from Metric (CSHORE) to Imperial (Beach-fx) during this phase.
