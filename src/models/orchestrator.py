from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os
import h5py
import logging
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from utils.morpho_metrics import compute_chain_metrics

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
        current_x = profile_data["x"]
        current_zb = profile_data["z"]
        logger.debug(f"Starting profile chain: {reach} - {profile_key}")

        for storm_key in storms.keys():
            BC_dict = {
                "timebc_wave": storms[storm_key]["time"],
                "Hs": storms[storm_key]["Hmo"],
                "Hrms": storms[storm_key]["Hrms"],
                "Tp": storms[storm_key]["tp"],
                "Wsetup": np.zeros(len(storms[storm_key]["tp"])),
                "swlbc": storms[storm_key]["surge"],
                "angle": storms[storm_key]["angle"],
                "x": current_x,
                "x_p": np.zeros(len(current_x)),
                "zb": current_zb,
                "zb_p": np.zeros(len(current_x)),
                "fw": np.zeros(len(current_x)),
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

            # Carry forward updated profile to next storm
            updated = adapter.updater.update(storm_results)
            current_x = updated["x"]
            current_zb = updated["z"]

        # Save master parquet for this chain
        adapter.save_master_parquet(reach, f"{profile_key}_chain", all_storm_results)

        # Compute and log morphology metrics
        metrics = compute_chain_metrics(all_storm_results)
        self._log_metrics(reach, profile_key, metrics)

        # Plot profile evolution, per-storm differences, and cumulative change
        self._plot_chain_profiles(reach, profile_key, profile_data, all_storm_results, adapter.config)
        self._plot_chain_differences(reach, profile_key, all_storm_results, adapter.config)
        self._plot_chain_cumulative(reach, profile_key, all_storm_results, adapter.config)

    def _log_metrics(self, reach, profile_key, metrics):
        dh = metrics["dune_height_change_ft"]
        bw = metrics["berm_width_change_ft"]
        logger.info(
            f"{reach}/{profile_key} — dune height change (ft): "
            f"mean={dh['mean']:+.4f}  min={dh['min']:+.4f}  max={dh['max']:+.4f}"
        )
        logger.info(
            f"{reach}/{profile_key} — berm width change (ft):  "
            f"mean={bw['mean']:+.4f}  min={bw['min']:+.4f}  max={bw['max']:+.4f}"
        )
        for s in metrics["per_storm"]:
            logger.info(
                f"  {s['storm_id']}: dune={s['dune_height_change_ft']:+.4f} ft  "
                f"berm={s['berm_width_change_ft']:+.4f} ft"
            )

    def _plot_chain_profiles(self, reach, profile_key, profile_data, storm_results, config):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        out_dir = os.path.join(root_path, config["paths"]["outfiles"], reach)
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 4))

        # Input profile (feet)
        x0 = profile_data["x"] / 0.3048
        z0 = profile_data["z"] / 0.3048
        ax.plot(x0, z0, color="black", linewidth=1.5, label="Input profile")

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(storm_results)))
        for i, result in enumerate(storm_results):
            storm_id = result["storm_id"].split("-")[-1]
            ax.plot(
                result["final_profile_x"] / 0.3048,
                result["final_profile_zb"] / 0.3048,
                color=colors[i],
                linewidth=1,
                label=f"After {storm_id}",
            )

        ax.axhline(0, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Cross-shore distance (ft)")
        ax.set_ylabel("Elevation (ft)")
        ax.set_title(f"{reach} — {profile_key} chain profile evolution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, f"{profile_key}_chain_profiles.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved profile plot: {plot_path}")

    def _plot_chain_differences(self, reach, profile_key, storm_results, config):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        out_dir = os.path.join(root_path, config["paths"]["outfiles"], reach)
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 4))

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(storm_results)))
        for i, result in enumerate(storm_results):
            storm_id = result["storm_id"].split("-")[-1]
            dz = (result["final_profile_zb"] - result["initial_profile_zb"]) / 0.3048
            ax.plot(
                result["final_profile_x"] / 0.3048,
                dz,
                color=colors[i],
                linewidth=1,
                label=storm_id,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Cross-shore distance (ft)")
        ax.set_ylabel("Bed level change (ft)")
        ax.set_title(f"{reach} — {profile_key} per-storm bed level change")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, f"{profile_key}_chain_differences.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved difference plot: {plot_path}")

    def _plot_chain_cumulative(self, reach, profile_key, storm_results, config):
        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        out_dir = os.path.join(root_path, config["paths"]["outfiles"], reach)
        os.makedirs(out_dir, exist_ok=True)

        # Reference grid and initial profile from first storm (meters)
        ref_x = storm_results[0]["initial_profile_x"]
        ref_zb = storm_results[0]["initial_profile_zb"]

        fig, ax = plt.subplots(figsize=(12, 4))

        # Sum per-storm dz values on reference grid; convert to feet for display
        dz_total = np.zeros(len(ref_x))
        for result in storm_results:
            dz_total += np.interp(ref_x, result["final_profile_x"], result["final_profile_zb"]) \
                      - np.interp(ref_x, result["initial_profile_x"], result["initial_profile_zb"])
        ref_x_ft = ref_x / 0.3048
        dz_total_ft = dz_total / 0.3048

        ax.plot(ref_x_ft, dz_total_ft, color="black", linewidth=1.2)
        ax.fill_between(ref_x_ft, 0, dz_total_ft, where=dz_total_ft >= 0, alpha=0.25, color="green", label="Accretion")
        ax.fill_between(ref_x_ft, 0, dz_total_ft, where=dz_total_ft <  0, alpha=0.25, color="red",   label="Erosion")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Cross-shore distance (ft)")
        ax.set_ylabel("Cumulative bed level change (ft)")
        ax.set_title(f"{reach} — {profile_key} cumulative bed level change from input profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, f"{profile_key}_chain_cumulative.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved cumulative plot: {plot_path}")
