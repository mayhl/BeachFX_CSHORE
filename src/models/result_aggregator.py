import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class ParquetResultAggregator:
    def __init__(self, reach_out_path):
        self.reach_out_path = reach_out_path

    def generate(self):
        # Scan reach_out_path for all individual storm parquet files
        # Exclude existing master files to prevent recursive aggregation
        parquet_files = [
            f
            for f in os.listdir(self.reach_out_path)
            if f.endswith(".parquet") and not f.endswith("_master.parquet")
        ]
        logger.debug(
            f"Found {len(parquet_files)} parquet files in {self.reach_out_path}: {parquet_files}"
        )

        # Aggregate all results into a single list of DataFrames
        all_data = []
        for f in parquet_files:
            df = pd.read_parquet(os.path.join(self.reach_out_path, f))
            all_data.append(df)

        if all_data:
            # Consolidate into one master file, sorted by chain order
            master_df = pd.concat(all_data, ignore_index=True)
            if "chain_order" in master_df.columns:
                master_df = master_df.sort_values("chain_order").reset_index(drop=True)
            # Use os.path.basename(self.reach_out_path) to get the reach name
            reach_name = os.path.basename(self.reach_out_path)
            master_filename = os.path.join(
                self.reach_out_path, reach_name + "_master.parquet"
            )
            master_df.to_parquet(master_filename, index=False)
            logger.info(f"Aggregated {len(all_data)} storms into: {master_filename}")

            # Cleanup individual files
            for f in parquet_files:
                os.remove(os.path.join(self.reach_out_path, f))
            logger.info(f"Cleaned up {len(parquet_files)} individual chain files.")
        else:
            logger.warning("No parquet data to aggregate.")
