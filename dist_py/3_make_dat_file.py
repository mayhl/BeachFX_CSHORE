import os
import sys
import json


# ~~~ begin script ~~~
current_path = os.path.abspath(os.path.dirname(__file__))			#adding pyfiles subdirectory to this working path
root_path = os.path.abspath(os.path.join(current_path, ".."))
pypath = os.path.join(current_path, "pyfiles")
sys.path.insert(0, pypath)

# Load configuration
config_path = os.path.join(root_path, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

from output_to_DAT_background import output_to_DAT_background		#importing background script
otdb = output_to_DAT_background()

meta_dict = {}														#setting up dictionary with work and infile directories
meta_dict['work_directory'] = os.path.join(root_path, config['paths']['data'])
meta_dict['infile_directory'] = os.path.join(root_path, config['paths']['infiles'])
meta_dict['outfile_directory'] = os.path.join(root_path, config['paths']['outfiles'])

reaches_to_process = []
if not reaches_to_process:											#if there are no reaches to process
	reaches_to_process = os.listdir(meta_dict['outfile_directory'])	#read what's in the infile directory
	reaches_to_process = [f for f in reaches_to_process if not f.startswith('.')]

print("\n" + "="*50)
print("BEACH-FX DAT FILE GENERATION")
print("="*50)

for reach in reaches_to_process:									#loop through each reach
	print(f"[*] Formatting Reach: {reach:20}", end="", flush=True)
	os.chdir(os.path.join(meta_dict['outfile_directory'], reach))	#change to reach directory

	meta_dict['outfile_directory_reach'] = os.path.join(os.getcwd()) 	#adding the reach outfile path to meta_dict
	h5file = os.listdir(os.getcwd())
	h5file = [f for f in h5file if f.endswith('.h5')][0]
	otdb.init(h5file, meta_dict)
	print(" DONE")

print("-"*50)
print(f"Success: DAT files written to {meta_dict['outfile_directory']}")
print("="*50 + "\n")
