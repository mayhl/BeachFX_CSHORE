from .base import AbstractModelAdapter
from .cshore_io import cshoreIO
from .cshore_infiles import MakeInfiles
from .result_aggregator import ParquetResultAggregator
from .morpho_updater import MorphologyUpdater, MockMorphoUpdater
from utils.chs_utils import write_parquet
from typing import List, Dict
import os
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
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.exe_path = os.path.join(
            root_path, self.config["paths"]["executables"], "CSHORE_USACE_LINUX.out"
        )

    def prepare_infiles(self, meta_dict: dict, profiles: dict, storms: dict) -> None:
        self.mkInfiles.init(meta_dict, self.config, profiles, storms)

    def run_simulation(self, reach_dir: str, infile_name: str) -> dict:
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
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

        # Load results from the isolated directory
        params, bc, veg, hydro, sed, morpho = self.csio.load_CSHORE_results(storm_dir)
        logger.info(f"Results loaded for {infile_name}")

        # Return structured results
        return {
            "storm_id": infile_name,
            "initial_profile_x": np.array(morpho["x"][0]) / 0.3048,
            "initial_profile_zb": np.array(morpho["zb"][0]) / 0.3048,
            "final_profile_x": np.array(morpho["x"][-1]) / 0.3048,
            "final_profile_zb": np.array(morpho["zb"][-1]) / 0.3048,
        }

    def save_master_parquet(self, reach_dir: str, profile_key: str, data: List[Dict]):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        reach_out_path = os.path.join(
            root_path, self.config["paths"]["outfiles"], reach_dir
        )
        os.makedirs(reach_out_path, exist_ok=True)

        parquet_filename = os.path.join(reach_out_path, f"{profile_key}.parquet")

        # Consolidate list of dicts to DataFrame
        master_df = pd.DataFrame(data)
        master_df["profile_id"] = profile_key.replace("_chain", "")
        write_parquet(parquet_filename, master_df.to_dict(orient="list"))
        logger.info(f"Successfully wrote master output: {parquet_filename}")

    def parse_outputs(self, reach_dir: str) -> None:
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        reach_out_path = os.path.join(
            root_path, self.config["paths"]["outfiles"], reach_dir
        )

        generator = ParquetResultAggregator(reach_out_path)
        generator.generate()
