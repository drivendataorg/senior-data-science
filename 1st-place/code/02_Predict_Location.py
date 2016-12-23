# -*- coding: utf-8 -*-

import os
import time

# Start
start_time = time.time()
print ("Starting Predict Location ...")

# Generate list of files
files = ["SAS_L0_001_FeatureExtraction_targets_locations.py",
         "SAS_L0_601_PredictLocation_L1_ET_vA1.py"]
files.sort()
print ("...total {:} files".format(len(files)))
print

# execute files
def execute_file(file_txt):
    cmd="python ./"+file_txt
    print ("......Running: {:}".format(file_txt))
    os.system(cmd)
    print ("......Finished: {:}".format(file_txt))
    print

for file_txt in files:
    execute_file(file_txt)
    
print
print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))
