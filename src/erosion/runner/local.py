from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid

import numpy as np

from pydantic import BaseModel

from .base import CSHOREResult, CSHORERunner
from .cshore_io import cshoreIO
from ..profile import Profile


class CSHOREParams(BaseModel):
    """CSHORE physical and calibration parameters per SOW S3.1.

    SOW-required calibration parameters: d50, blp, effb, gamma.
    Remaining fields match the ``config["cshore"]`` dict consumed by
    ``cshoreIO.make_CSHORE_infile()``.  Defaults are taken from the
    project reference ``config.json``.
    """
    # SOW calibration parameters
    d50: float = 0.3        # median grain size (mm); used as reach-wide default for profile init
    blp: float = 0.001      # bedload parameter
    effb: float = 0.002     # suspension efficiency due to breaking
    gamma: float = 0.7      # shallow-water wave height to depth ratio

    # Additional CSHORE parameters (commonly adjusted during calibration)
    efff: float = 0.005     # suspension efficiency due to friction
    slp: float = 0.5        # suspended load parameter
    slpot: float = 0.1      # overtopping suspended load parameter
    tanphi: float = 0.63    # tangent of sediment friction angle
    dx: float = 1.0         # profile node spacing (m)
    rwh: float = 0.02       # numerical runup wire height (m)
    fw: float = 0.015       # bed friction

    # Physical constants (rarely changed)
    sg: float = 2.65        # specific gravity of sand
    sporo: float = 0.4      # sediment porosity
    temp: float = 20.0      # water temperature (°C)
    salin: float = 0.0      # salinity (ppt)

    # CSHORE model logic flags
    iline: int = 1          # wave transformation flag
    iprofl: float = 1.1     # profile change flag (1.1 = mobile bed)
    isedav: int = 0         # unlimited sediment
    iperm: int = 0          # no permeability
    iover: int = 1          # allow overwash
    infilt: int = 0         # no infiltration
    iwtran: int = 0         # no wave transmission
    ipond: int = 0          # no ponding
    iwcint: int = 0         # no wave-current interaction
    iroll: int = 0          # no roller model
    iwind: int = 0          # no wind
    itide: int = 0          # no tide
    ilab: int = 0           # natural conditions (0=field, 1=lab)

_EXE_NAMES = {
    "darwin": "cshore_usace_macos.out",
    "linux": "CSHORE_USACE_LINUX.out",
    "win32": "cshore_usace_win.out",
}

_EXECUTABLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "executables")
)


def _exe_path() -> str:
    name = _EXE_NAMES.get(sys.platform, "CSHORE_USACE_LINUX.out")
    return os.path.join(_EXECUTABLES_DIR, name)


def _build_cshore_config(params: CSHOREParams) -> dict:
    return {
        "cshore": {
            "dx": params.dx,
            "gamma": params.gamma,
            "effb": params.effb,
            "efff": params.efff,
            "slp": params.slp,
            "slpot": params.slpot,
            "tanphi": params.tanphi,
            "blp": params.blp,
            "rwh": params.rwh,
            "fw": params.fw,
            "sporo": params.sporo,
            "sg": params.sg,
            "temp": params.temp,
            "salin": params.salin,
        },
        "model_logic": {
            "iline": params.iline,
            "iprofl": params.iprofl,
            "isedav": params.isedav,
            "iperm": params.iperm,
            "iover": params.iover,
            "infilt": params.infilt,
            "iwtran": params.iwtran,
            "ipond": params.ipond,
            "iwcint": params.iwcint,
            "iroll": params.iroll,
            "iwind": params.iwind,
            "itide": params.itide,
            "ilab": params.ilab,
        },
        "vegetation": {
            "enabled": False,
            "Cd": 1.0,
            "n": 0.0,
            "dia": 0.0,
            "ht": 0.0,
            "rod": 0.0,
            "extent": 0.0,
        },
    }


class LocalCSHORERunner(CSHORERunner):
    """Runs CSHORE via subprocess, reusing cshoreIO for infile generation and parsing."""

    def __init__(
        self,
        params: CSHOREParams,
        work_dir: str,
        infile_dir: str | None = None,
    ) -> None:
        self.config = _build_cshore_config(params)
        self.work_dir = work_dir
        self.exe = _exe_path()
        self.infile_dir = infile_dir
        if infile_dir:
            os.makedirs(infile_dir, exist_ok=True)

    def run(self, profile: Profile, storm_forcing: dict) -> CSHOREResult:
        csio = cshoreIO()  # fresh instance per call → thread-safe (no class-level state)

        storm_uuid = str(uuid.uuid4())[:8]
        storm_dir = os.path.join(self.work_dir, f"{profile.id}_{storm_uuid}")
        os.makedirs(storm_dir, exist_ok=True)

        infile_path = os.path.join(storm_dir, "infile")

        n = len(profile.x)
        BC_dict = {
            **storm_forcing,
            "x": profile.x,
            "zb": profile.zb,
            "x_p": np.zeros(n),
            "zb_p": np.zeros(n),
            "fw": np.zeros(n),
            "d50": profile.d50,
        }
        meta_dict = {"Reach": "fw", "Profile": profile.id, "Storm": storm_uuid}

        csio.make_CSHORE_infile(infile_path, BC_dict, meta_dict, self.config)

        if self.infile_dir:
            dest = os.path.join(self.infile_dir, f"{profile.id}_{storm_uuid}.infile")
            shutil.copy2(infile_path, dest)

        subprocess.run(
            [self.exe],
            cwd=storm_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        odoc_path = os.path.join(storm_dir, "ODOC")
        if not os.path.exists(odoc_path):
            raise RuntimeError(
                f"CSHORE produced no output in {storm_dir}. "
                "Surge likely exceeded profile crest elevation."
            )

        params, bc, veg, hydro, sed, morpho = csio.load_CSHORE_results(storm_dir)

        zb_final = np.array(morpho["zb"][-1])
        x_final = np.array(morpho["x"][-1])

        # Strip trailing NaN padding introduced by grid size variation between storms
        valid = ~np.isnan(zb_final)
        zb_final = zb_final[valid]
        x_final = x_final[valid]

        num_steps = params["num_steps"]
        last = num_steps - 1
        eta = np.array(hydro["mwl"].get(last, hydro["mwl"][0]))
        Hs = np.array(hydro["Hs"].get(last, hydro["Hs"][0]))

        runup_arr = csio.ODOC_dict.get("runup_2_percent", np.array([np.nan]))
        runup_val = runup_arr[-1] if len(runup_arr) else np.nan
        runup_m = 0.0 if np.isnan(runup_val) else float(runup_val)

        # Release large CSHORE output dicts immediately — in thread-mode Dask workers
        # GC can be delayed by traceback references, so be explicit.
        del params, bc, veg, hydro, sed, morpho, csio

        return CSHOREResult(
            zb=zb_final,
            x=x_final,
            eta=eta,
            Hs=Hs,
            runup_m=runup_m,
        )
