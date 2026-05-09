# Import Packages
import numpy as np
import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime


# Define Methods
def find_nearest_latlon(
    target_lat, target_lon, latitudes, longitudes, max_radius_km=None
):
    # Convert degrees to radians
    target_lat = np.deg2rad(target_lat)
    target_lon = np.deg2rad(target_lon)
    latitudes = np.deg2rad(latitudes)
    longitudes = np.deg2rad(longitudes)

    # Compute differences
    dlat = latitudes - target_lat
    dlon = longitudes - target_lon

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(target_lat) * np.cos(latitudes) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = 6371 * c  # Earth's radius in km

    # Find nearest point
    if max_radius_km is None:
        min_indx = np.argmin(distance_km)
        nearest_lat = np.rad2deg(latitudes[min_indx])
        nearest_lon = np.rad2deg(longitudes[min_indx])
        within_radius = None
    else:
        within_radius = distance_km <= max_radius_km
        if not np.any(within_radius):
            return None, None, within_radius, distance_km, None
        filtered_distances = distance_km[within_radius]
        min_indx_local = np.argmin(filtered_distances)
        filtered_latitudes = latitudes[within_radius]
        filtered_longitudes = longitudes[within_radius]
        nearest_lat = np.rad2deg(filtered_latitudes[min_indx_local])
        nearest_lon = np.rad2deg(filtered_longitudes[min_indx_local])
        min_indx = np.where(within_radius)[0][min_indx_local]

    return nearest_lat, nearest_lon, within_radius, distance_km, min_indx


def list_h5_files(folder_to_scan):
    files = glob.glob(os.path.join(folder_to_scan, "*.h5"))
    return [os.path.basename(f) for f in files]


def chs_wave_model_header_locator(headers):
    """
    Identify the actual header strings for Hm0, Tp/Tm, and wave direction.

    Parameters
    ----------
    headers : list of str
        List of header strings from the converted CHS file.

    Returns
    -------
    matched_headers : dict
        Dictionary with keys "Hm0", "Tp", "wDir" containing the actual matched header strings.
    Tp_special : int
        Flag value: 0 if Tp is found, 1 if only Tm is found.
    """

    # --- Hm0 ---
    hm0_candidates = [
        "Zero Moment Wave Height",
        "Significant Wave Height",
        "Significant Wave Height Total Sea",
    ]
    hm0 = next((h for h in hm0_candidates if h in headers), None)
    if hm0 is None:
        raise ValueError("Hm0 not found.")

    # --- Tp or Tm ---
    tp_candidates = [
        "Peak Spectral Wave Period Total Sea",
        "Peak Period",
        "Smoothed Peak Period",
        "Peak Wave Period",
    ]
    tp = next((h for h in tp_candidates if h in headers), None)
    if tp is None:
        tm_candidates = ["Mean Wave Period"]
        tp = next((h for h in tm_candidates if h in headers), None)
        if tp is None:
            raise ValueError("Tp/Tm not found.")
        Tp_special = 1  # Tm case
    else:
        Tp_special = 0  # Tp case

    # --- Wave direction ---
    wdir_candidates = ["Mean Wave Direction Total Sea", "Mean Wave Direction"]
    wdir = next((h for h in wdir_candidates if h in headers), None)
    if wdir is None:
        raise ValueError("Mean Wave Direction not found.")

    # --- Result ---
    matched_headers = {"Hm0": hm0, "Tp": tp, "wDir": wdir}

    return matched_headers, Tp_special


def infer_type(value):
    """
    Infer a PyArrow data type.
    """
    if isinstance(value, (list, np.ndarray)):
        # Handle list-based data explicitly
        return pa.list_(pa.float64())

    # If it's already a string, check if it can be a timestamp or keep it as string.
    # Do not try to convert to int/float if it contains non-numeric characters.
    if isinstance(value, str):
        # Try datetime
        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                datetime.strptime(value, fmt)
                return pa.timestamp("s")
            except:
                pass
        return pa.string()

    # Try integer
    try:
        int(value)
        return pa.int64()
    except:
        pass

    # Try float
    try:
        float(value)
        return pa.float64()
    except:
        pass

    # Default to string
    return pa.string()


def write_parquet(fout: str, dict_in: dict):
    # Convert To DataFrame
    aa = pd.DataFrame(dict_in)

    # Explicitly define schema to avoid inference issues with sequences
    fields = []
    for col in aa.columns:
        # Use first non-null element to infer type
        sample = aa[col].dropna().iloc[0]
        dtype = infer_type(sample)
        fields.append(pa.field(col, dtype))
    schema = pa.schema(fields)

    # Create Table
    table = pa.Table.from_pandas(
        aa,
        schema=schema,
        preserve_index=False,
    )
    # Write Out Parquet
    pq.write_table(table, fout)
