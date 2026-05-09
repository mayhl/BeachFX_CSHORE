from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os
import h5py
import logging
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExecutionPolicy(ABC):
    @abstractmethod
    def run(
        self,
        adapter: "AbstractModelAdapter",
        meta_dict: Dict,
        profiles: Dict,
        storms: Dict,
    ) -> None:
        pass


class BatchPolicy(ExecutionPolicy):
    def run(
        self,
        adapter: "AbstractModelAdapter",
        meta_dict: Dict,
        profiles: Dict,
        storms: Dict,
    ) -> None:
        start_time = time.time()

        tasks = []
        for reach in profiles.keys():
            infiles_base = os.path.join(meta_dict["work_directory"], "infiles", reach)
            infiles = sorted(
                [f for f in os.listdir(infiles_base) if f.endswith(".infile")]
            )
            for infile in infiles:
                tasks.append((adapter, reach, infile))

        logger.info("Generating input files...")
        adapter.prepare_infiles(meta_dict, profiles, storms)

        logger.info(f"Running {len(tasks)} simulations in parallel...")
        with ProcessPoolExecutor() as executor:
            # Consume generator to ensure all finish
            list(
                executor.map(
                    lambda t: adapter.run_simulation(t[1], t[2].rsplit(".", 1)[0]),
                    tasks,
                )
            )

        logger.info("Aggregating results...")
        for reach in profiles.keys():
            adapter.parse_outputs(reach_dir=reach)

        print(f"Pipeline Summary | Total Time: {time.time() - start_time:.2f}s")


class ChainedPolicy(ExecutionPolicy):
    def run(
        self,
        adapter: "AbstractModelAdapter",
        meta_dict: Dict,
        profiles: Dict,
        storms: Dict,
    ) -> None:
        tasks = []
        for reach, reach_profiles in profiles.items():
            for profile_key, profile_data in reach_profiles.items():
                tasks.append(
                    (adapter, reach, profile_key, meta_dict, profile_data, storms)
                )

        logger.info(f"Submitting {len(tasks)} chains to executor...")

        # Process chains in parallel
        with ProcessPoolExecutor() as executor:
            list(executor.map(self._run_profile_chain, tasks))

        # Aggregate results
        logger.info("Aggregating reach results...")
        for reach in profiles.keys():
            adapter.parse_outputs(reach_dir=reach)

    def _run_profile_chain(self, args):
        adapter, reach, profile_key, meta_dict, profile_data, storms = args

        all_storm_results = []
        logger.debug(f"Starting profile chain: {reach} - {profile_key}")

        for storm_key in storms.keys():
            # Update BC_dict
            current_zb = profile_data["z"]
            BC_dict = {
                "timebc_wave": storms[storm_key]["time"],
                "Hs": storms[storm_key]["Hmo"],
                "Hrms": storms[storm_key]["Hrms"],
                "Tp": storms[storm_key]["tp"],
                "Wsetup": np.zeros(len(storms[storm_key]["tp"])),
                "swlbc": storms[storm_key]["surge"],
                "angle": storms[storm_key]["angle"],
                "x": profile_data["x"],
                "x_p": np.zeros(len(profile_data["x"])),
                "zb": current_zb,
                "zb_p": np.zeros(len(profile_data["x"])),
                "fw": np.zeros(len(profile_data["x"])),
                "d50": profile_data["d50"],
            }

            temp_meta = meta_dict.copy()
            temp_meta["Reach"] = reach
            temp_meta["Profile"] = profile_key
            temp_meta["Storm"] = storm_key

            # Write and Run
            infile = adapter.mkInfiles.write_single_infile(
                reach, profile_key, storm_key, BC_dict, temp_meta, adapter.config
            )
            infile_name = os.path.basename(infile).rsplit(".", 1)[0]

            storm_results = adapter.run_simulation(
                reach_dir=reach, infile_name=infile_name
            )
            all_storm_results.append(storm_results)

            # Update zb
            current_zb = adapter.updater.update(storm_results)

        # Save master parquet for this chain
        adapter.save_master_parquet(reach, f"{profile_key}_chain", all_storm_results)
