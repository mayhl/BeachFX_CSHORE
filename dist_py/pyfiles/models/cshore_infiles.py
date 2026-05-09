import os
import numpy as np
from .cshore_io import cshoreIO

csio = cshoreIO()


class MakeInfiles(object):
    def __init__(self):
        self.infiles = {}
        self.csio = cshoreIO()

    def init(self, meta_dict, config, profiles, storms):
        self.meta_dict = meta_dict  # setting up dictionaries
        self.config = config
        self.profiles = profiles
        self.storms = storms

        self.make_indirectory()  # creating (and moving to) an infile directory
        self.make_infiles()  # writing to infile directory

    def make_indirectory(self):
        for reach in self.profiles.keys():
            infile_directory = os.path.join(
                self.meta_dict["work_directory"], "infiles", reach
            )
            if not os.path.exists(infile_directory):
                os.makedirs(infile_directory)
            os.chdir(infile_directory)

    def write_single_infile(self, reach, profile, storm, BC_dict, meta_dict, config):
        """
        Atomic generation of a single .infile.
        """
        infile_directory = os.path.join(meta_dict["work_directory"], "infiles", reach)
        os.makedirs(infile_directory, exist_ok=True)

        fname = f"{reach}_{profile}-{storm}.infile"
        full_path = os.path.join(infile_directory, fname)

        self.csio.make_CSHORE_infile(full_path, BC_dict, meta_dict, config)
        return full_path

    def make_infiles(self):
        for reach in self.profiles.keys():
            for storm in self.storms.keys():
                for profile in self.profiles[reach].keys():
                    BC_dict = {}
                    BC_dict["timebc_wave"] = self.storms[storm]["time"]
                    BC_dict["Hs"] = self.storms[storm]["Hmo"]
                    BC_dict["Hrms"] = self.storms[storm]["Hrms"]
                    BC_dict["Tp"] = self.storms[storm]["tp"]
                    BC_dict["Wsetup"] = np.zeros(len(BC_dict["Tp"]))
                    BC_dict["swlbc"] = self.storms[storm]["surge"]
                    BC_dict["angle"] = np.zeros(len(BC_dict["Tp"]))
                    BC_dict["x"] = self.profiles[reach][profile]["x"]
                    BC_dict["x_p"] = np.zeros(len(BC_dict["x"]))
                    BC_dict["zb"] = self.profiles[reach][profile]["z"]
                    BC_dict["zb_p"] = np.zeros(len(BC_dict["x"]))
                    BC_dict["fw"] = np.zeros(len(BC_dict["x"]))
                    BC_dict["d50"] = self.profiles[reach][profile]["d50"]

                    temp_meta = self.meta_dict.copy()
                    temp_meta["Reach"] = reach
                    temp_meta["Profile"] = profile
                    temp_meta["Storm"] = storm

                    self.write_single_infile(
                        reach, profile, storm, BC_dict, temp_meta, self.config
                    )
