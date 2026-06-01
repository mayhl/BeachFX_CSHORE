import numpy as np


def _smooth(z, window=7):
    """Centred moving average — suppresses CSHORE numerical noise."""
    kernel = np.ones(window) / window
    return np.convolve(z, kernel, mode="same")


def _first_local_max(arr, x, min_val=0.5, order=3):
    """
    Return (index, x, z) of the first local maximum in arr that exceeds
    min_val, using a half-window of `order` points on each side.
    Returns (None, nan, nan) if not found.
    """
    for i in range(order, len(arr) - order):
        if arr[i] > min_val and arr[i] == np.max(arr[i - order : i + order + 1]):
            return i, float(x[i]), float(arr[i])
    return None, np.nan, np.nan


def extract_beach_metrics(x, z, smooth_window=7):
    """
    Extract dune height and berm width from a cross-shore CSHORE profile.

    x : array-like, cross-shore distance in feet (landward-positive, x=0 offshore)
    z : array-like, bed elevation in feet

    Key features extracted:
      shoreline_x   : x at z=0 crossing (ft)
      dune_height   : max elevation (ft)
      dune_x        : x of dune crest (ft)
      berm_crest_z  : elevation of berm crest — first local max above shoreline (ft)
      berm_crest_x  : x of berm crest (ft)
      dune_toe_x    : x of dune toe — point of max positive curvature between
                      berm crest and dune crest, i.e. base of steep dune face (ft)
      berm_width    : x_dune_toe - x_shoreline (ft)
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zs = _smooth(z, smooth_window)

    nan_result = dict(
        shoreline_x=np.nan, dune_height=np.nan, dune_x=np.nan,
        berm_crest_z=np.nan, berm_crest_x=np.nan,
        dune_toe_x=np.nan, berm_width=np.nan,
    )

    if len(x) < 20:
        return nan_result

    # --- Dune crest: global max of smoothed profile ---
    dune_idx = int(np.argmax(zs))
    dune_height = float(z[dune_idx])
    dune_x = float(x[dune_idx])

    # --- Shoreline: last wet→dry transition seaward of dune crest ---
    # Use z>0 / z<=0 to handle z==0 cleanly
    dry = (zs[:dune_idx] > 0).astype(int)
    transitions = np.where(np.diff(dry) > 0)[0]  # wet→dry
    if len(transitions) == 0:
        return {**nan_result, "dune_height": dune_height, "dune_x": dune_x}
    ci = int(transitions[-1])
    z0, z1 = float(zs[ci]), float(zs[ci + 1])
    shoreline_x = float(x[ci] + (0 - z0) * (x[ci + 1] - x[ci]) / (z1 - z0))
    shore_idx = ci + 1

    if dune_idx <= shore_idx + 6:
        return {**nan_result, "dune_height": dune_height, "dune_x": dune_x,
                "shoreline_x": shoreline_x}

    # Beach zone arrays (shoreline → dune crest)
    beach_zs = zs[shore_idx : dune_idx + 1]
    beach_z  = z[shore_idx  : dune_idx + 1]
    beach_x  = x[shore_idx  : dune_idx + 1]

    # --- Berm crest: first local max in beach zone above min_val ---
    berm_local_idx, berm_crest_x, berm_crest_z = _first_local_max(
        beach_zs, beach_x, min_val=0.5, order=3
    )
    if berm_local_idx is None:
        # No distinct berm — use max of lower half as proxy
        half = len(beach_zs) // 2
        berm_local_idx = int(np.argmax(beach_zs[:half]))
        berm_crest_x = float(beach_x[berm_local_idx])
        berm_crest_z = float(beach_z[berm_local_idx])

    # --- Dune toe: max positive curvature between berm crest and dune crest ---
    # d²z/dx² peaks at the concave-up inflection point = base of dune face
    start = berm_local_idx + 1
    if dune_idx - shore_idx - start < 4:
        dune_toe_x = berm_crest_x
    else:
        seg_zs = zs[shore_idx + start : dune_idx]
        seg_x  = x[shore_idx + start  : dune_idx]
        dx_arr = np.diff(seg_x)
        dz1    = np.diff(seg_zs)
        # Second derivative (curvature proxy): d(dz/dx)/dx
        slope1 = np.where(dx_arr > 0, dz1 / dx_arr, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            d2z = np.diff(slope1) / dx_arr[:-1]
        if len(d2z) == 0 or np.all(np.isnan(d2z)):
            dune_toe_x = berm_crest_x
        else:
            toe_local = int(np.nanargmax(d2z)) + 1  # offset for double diff
            dune_toe_x = float(seg_x[toe_local]) if toe_local < len(seg_x) else berm_crest_x

    berm_width = float(dune_toe_x - shoreline_x)

    return dict(
        shoreline_x=shoreline_x,
        dune_height=dune_height,
        dune_x=dune_x,
        berm_crest_z=berm_crest_z,
        berm_crest_x=berm_crest_x,
        dune_toe_x=dune_toe_x,
        berm_width=berm_width,
    )


def compute_chain_metrics(storm_results):
    """
    Per-storm dune height and berm width changes, plus mean/min/max summary.
    storm_results : list of dicts with initial/final_profile_x/zb in feet.
    """
    dune_changes = []
    berm_changes = []

    for r in storm_results:
        m_init  = extract_beach_metrics(r["initial_profile_x"], r["initial_profile_zb"])
        m_final = extract_beach_metrics(r["final_profile_x"],   r["final_profile_zb"])
        dune_changes.append(m_final["dune_height"] - m_init["dune_height"])
        berm_changes.append(m_final["berm_width"]  - m_init["berm_width"])

    def stats(arr):
        a = np.array(arr)
        return {"mean": float(np.nanmean(a)), "min": float(np.nanmin(a)),
                "max": float(np.nanmax(a))}

    return {
        "dune_height_change_ft": stats(dune_changes),
        "berm_width_change_ft":  stats(berm_changes),
        "per_storm": [
            {
                "storm_id": r["storm_id"].split("-")[-1],
                "dune_height_change_ft": dc,
                "berm_width_change_ft":  bc,
            }
            for r, dc, bc in zip(storm_results, dune_changes, berm_changes)
        ],
    }
