"""
this script is used to generate the infiles for CSHORE runs that will be used in Beach-fx modeling.
	-the infiles will be written to the following directory structure:
		../work/infiles/"reach"/infiles
	-the user should not remove the infiles from this location until the cshore runs have been completed. 
	-information about the required input values can be found throughout this script.
user input begins following "BEGIN USER INPUT"
	**Note that the units of the user input vary for Beach-fx (US) and CSHORE (SI)
	-the script performs necessary unit conversions assuming the input units are correct. 

Python scripts (except cshoreIO.py) written by 
	Dylan R. Sanderson
	Coastal and Hydraulics Laboratory
	Engineer Research and Development Center
	Vicksburg, MS
Work flow and tide creation from Brad Johnson. 
May, 2018
"""
import os
import sys
import json

# ~~~ starting script ~~~
current_path = os.path.abspath(os.path.dirname(__file__))	       #reading current path
root_path = os.path.abspath(os.path.join(current_path, ".."))
pypath = os.path.join(current_path, "pyfiles")                         #finding ../pyfiles subdirectory
sys.path.insert(0, pypath)					       #adding pyfiles subdirectory to this working path

# Load configuration
config_path = os.path.join(root_path, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

profile_dict = config['profile']
tide_dict = config['tide']

from PopulateProfileSpace import PopulateProfileSpace		       #importing and setting up "background" pyfiles
from CreateStorms import CreateStorms
from MakeInfiles import MakeInfiles
PPS = PopulateProfileSpace()
strms = CreateStorms()
mkInfiles = MakeInfiles()

meta_dict = {}
meta_dict['work_directory'] = os.path.join(root_path, config['paths']['data'])

for reach_num, reach in enumerate(profile_dict['names']):		#loop through reaches
	profiles = PPS.init(meta_dict, profile_dict, reach_num, reach)	#populating profile parameter space for reach
	if reach_num == 0:						#if first time through loop, setting up storms
		storms = strms.init(meta_dict, tide_dict)

#printing summary of the task
prof_in_reach = [(len(profiles[i].keys())) for i in profiles.keys()]	#calc number of profiles in each reach
tot_num = sum([len(storms.keys()) * i for i in prof_in_reach])		#calc number of profile/storm combinations

print("\n" + "="*50)
print("CSHORE INFILE GENERATION")
print("="*50)
print(f"Number of Reaches: {len(profile_dict['names'])}")
print(f"Reaches to process: {', '.join(profile_dict['names'])}")
print(f"Number of Storms:  {len(storms.keys())}")
print("-"*50)

mkInfiles.init(meta_dict, config, profiles, storms)		#making infiles for each profile/storm combination


print(f"Success: {tot_num} infiles written to {meta_dict['work_directory']}/infiles/")
print("="*50 + "\n")
