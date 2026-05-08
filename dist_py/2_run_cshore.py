import os
import sys
import time
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

from run_cshore_background import run_cshore_background				#loading background pyfiles
rcb = run_cshore_background()

meta_dict = {}														#setting up dictionaries
meta_dict['work_directory'] = os.path.join(root_path, config['paths']['data'])
meta_dict['infile_directory'] = os.path.join(root_path, config['paths']['infiles'])
meta_dict['outfile_directory'] = os.path.join(root_path, config['paths']['outfiles'])
meta_dict['exe_directory'] = os.path.join(root_path, config['paths']['executables'])

reaches = []
if not reaches:													#if the reaches list is blank, 
	reaches = os.listdir(meta_dict['infile_directory'])			# read all reaches in infile directory
	reaches = [f for f in reaches if not f.startswith('.')]		#remove hidden files that start with '.'

if not os.path.exists(meta_dict['outfile_directory']): 	#check if output path exists
	os.makedirs(meta_dict['outfile_directory'])			#if not, make directory

print("\n" + "="*50)
print("CSHORE SIMULATION RUNNER")
print("="*50)
total_start_time = time.time()

for reach in reaches:												#loop through reaches
	reach_start_time = time.time()
	print(f"[*] Processing Reach: {reach:20}", end="", flush=True)
	
	os.chdir(os.path.join(meta_dict['infile_directory'], reach))	#change to reach directory
	if not os.path.exists(os.path.join(meta_dict['outfile_directory'], reach)):
		os.makedirs(os.path.join(meta_dict['outfile_directory'], reach))
	meta_dict['outfile_directory_reach'] = os.path.join(meta_dict['outfile_directory'], reach)
	

	infiles = os.listdir(os.getcwd())
	rcb.init(meta_dict, reach)
	
	reach_end_time = time.time()
	duration = reach_end_time - reach_start_time
	print(f" DONE ({duration:6.2f}s)")

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print("-"*50)
print(f"Total processing time: {total_duration:.2f}s ({total_duration/60:.2f}m)")
print("="*50 + "\n")
