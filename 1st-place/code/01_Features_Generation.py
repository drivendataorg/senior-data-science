# -*- coding: utf-8 -*-

import os
import fnmatch
import multiprocessing
import time

# Start
start_time = time.time()
print ("Starting Feature Generation ...")

# Set cores
cores = 12
print ("...using {:} cores".format(cores))


# Generate list of files
files = []
for file_txt in os.listdir('.'):
    if fnmatch.fnmatch(file_txt, "SAS_L0_???_FeatureExtraction_ds*.py"):
        #print file_txt
        files.append(file_txt)
files.sort()
print ("...total {:} files".format(len(files)))
print

# execute files in parallel
def execute_file(file_txt):
    cmd="python ./"+file_txt+" >/dev/null"
    print ("......Running: {:}".format(file_txt))
    os.system(cmd)
    print ("......Finished: {:}".format(file_txt))


pool = multiprocessing.Pool(processes=cores)
pool.map(execute_file, files)

print
print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))
