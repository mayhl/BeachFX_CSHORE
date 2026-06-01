from __future__ import division
import numpy as np


def load_raw_profile(profile_path, d50):
    """
    Load a profile CSV (feet, Beach-FX seaward-positive convention) and
    convert to CSHORE format (meters, landward-positive).
    """
    raw = np.genfromtxt(profile_path, delimiter=",", encoding="utf-8-sig")
    valid = raw[~np.isnan(raw[:, 0])]
    x_ft, z_ft = valid[:, 0], valid[:, 1]

    x_m = x_ft * 0.3048
    z_m = z_ft * 0.3048

    # Reverse to CSHORE landward-positive (offshore end becomes x=0)
    z_m = z_m[::-1]
    diffs = np.diff(x_m)[::-1]  # positive spacings in reversed order
    x_m = np.concatenate(([0.0], np.cumsum(diffs)))

    return {"x": x_m, "z": z_m, "d50": d50}
