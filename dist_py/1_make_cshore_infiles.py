import os
import sys
import json
import numpy as np

# Setup paths
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(current_path, ".."))
pypath = os.path.join(current_path, "pyfiles")
sys.path.insert(0, pypath)

# Load configuration
config_path = os.path.join(root_path, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Import new framework
from models.cshore import CshoreAdapter
from models.orchestrator import ChainedPolicy
from utils.geometry import PopulateProfileSpace
from utils.create_storms import CreateStorms

# Prepare data
PPS = PopulateProfileSpace()
strms = CreateStorms()

meta_dict = {"work_directory": os.path.join(root_path, config["paths"]["data"])}
profiles = {}
for reach_num, reach in enumerate(config["profile"]["names"]):
    reach_profiles = PPS.init(meta_dict, config["profile"], reach_num, reach)
    profiles.update(reach_profiles)

# Use Parquet file
storm_path = os.path.join(root_path, "data", "New_EventDate_LC.parquet")
strms.read_storms_parquet(storm_path)
storms = strms.cshore_storms

# Execute using new Framework
adapter = CshoreAdapter(config)
orchestrator = ChainedPolicy()

# Run Infile Generation & Execution
orchestrator.run(adapter, meta_dict, profiles, storms)
