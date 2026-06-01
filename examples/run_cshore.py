import os
import sys
import json

# Setup paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_path, "src"))

from models.cshore import CshoreAdapter
from models.orchestrator import ChainedPolicy
from utils.geometry import load_raw_profile
from utils.create_storms import CreateStorms


def main():
    config_path = os.path.join(root_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    strms = CreateStorms()
    meta_dict = {"work_directory": os.path.join(root_path, config["paths"]["data"])}

    profiles = {}
    for reach, reach_cfg in config["profile"].items():
        profile_path = os.path.join(root_path, reach_cfg["file"])
        profiles[reach] = {"raw_profile": load_raw_profile(profile_path, reach_cfg["d50"])}

    storm_path = os.path.join(root_path, config["paths"]["storms"])
    strms.read_storms_parquet(storm_path)
    storms = strms.cshore_storms

    adapter = CshoreAdapter(config)
    orchestrator = ChainedPolicy()
    orchestrator.run(adapter, meta_dict, profiles, storms)


if __name__ == "__main__":
    main()
