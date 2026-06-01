from .base import AbstractModelAdapter
from .cshore_io import cshoreIO
from .cshore_infiles import MakeInfiles
from .result_aggregator import ParquetResultAggregator
from .morpho_updater import MorphologyUpdater, MockMorphoUpdater
from utils.chs_utils import write_parquet
from typing import List, Dict
import os
import sys
import subprocess
import numpy as np
import h5py
import shutil
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CshoreAdapter(AbstractModelAdapter):
    def __init__(self, config, updater: MorphologyUpdater = None):
        self.config = config
        self.updater = updater or MockMorphoUpdater()
        self.csio = cshoreIO()
        self.mkInfiles = MakeInfiles()
        _exe_names = {
            "darwin": "cshore_usace_macos.out",
            "linux": "CSHORE_USACE_LINUX.out",
            "win32": "cshore_usace_win.out",
        }
        _exe_name = _exe_names.get(sys.platform, "CSHORE_USACE_LINUX.out")
        self.exe_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "executables", _exe_name)
        )

    def prepare_infiles(self, meta_dict: dict, profiles: dict, storms: dict) -> None:
        self.mkInfiles.init(meta_dict, self.config, profiles, storms)

    def run_simulation(self, reach_dir: str, infile_name: str) -> dict:
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        reach_path = os.path.join(root_path, self.config["paths"]["infiles"], reach_dir)

        # Isolated directory per storm
        storm_dir = os.path.join(reach_path, infile_name)
        if not os.path.exists(storm_dir):
            os.makedirs(storm_dir)

        # Copy necessary files to isolated directory
        shutil.copy(
            os.path.join(reach_path, f"{infile_name}.infile"),
            os.path.join(storm_dir, "infile"),
        )

        logger.debug(f"Executing binary: {self.exe_path} in {storm_dir}")

        # Execute binary in the isolated directory
        subprocess.run(
            [self.exe_path],
            cwd=storm_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info(f"Binary execution finished for {infile_name}")

        # Check CSHORE produced output — binary silently deletes files when
        # water elevation exceeds maximum profile elevation
        if not os.path.exists(os.path.join(storm_dir, "ODOC")):
            raise RuntimeError(
                f"CSHORE produced no output for {infile_name}. "
                f"Surge likely exceeded profile crest elevation. "
                f"Check storm forcing or profile extent."
            )

        # Load results from the isolated directory
        params, bc, veg, hydro, sed, morpho = self.csio.load_CSHORE_results(storm_dir)
        logger.info(f"Results loaded for {infile_name}")

        # Return structured results in meters (SI throughout)
        return {
            "storm_id": infile_name,
            "initial_profile_x": np.array(morpho["x"][0]),
            "initial_profile_zb": np.array(morpho["zb"][0]),
            "final_profile_x": np.array(morpho["x"][-1]),
            "final_profile_zb": np.array(morpho["zb"][-1]),
        }

    def save_master_parquet(self, reach_dir: str, profile_key: str, data: List[Dict]):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        reach_out_path = os.path.join(
            root_path, self.config["paths"]["outfiles"], reach_dir
        )
        os.makedirs(reach_out_path, exist_ok=True)

        parquet_filename = os.path.join(reach_out_path, f"{profile_key}.parquet")

        # Consolidate list of dicts to DataFrame; convert m→ft for Beach-FX output
        master_df = pd.DataFrame(data)
        master_df["chain_order"] = range(len(master_df))   # preserve chain sequence
        for col in ["initial_profile_x", "initial_profile_zb", "final_profile_x", "final_profile_zb"]:
            if col in master_df.columns:
                master_df[col] = master_df[col].apply(lambda v: v / 0.3048)
        master_df["profile_id"] = profile_key.replace("_chain", "")
        write_parquet(parquet_filename, master_df.to_dict(orient="list"))
        logger.info(f"Successfully wrote master output: {parquet_filename}")

    def parse_outputs(self, reach_dir: str) -> None:
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        reach_out_path = os.path.join(
            root_path, self.config["paths"]["outfiles"], reach_dir
        )

        generator = ParquetResultAggregator(reach_out_path)
        generator.generate()
