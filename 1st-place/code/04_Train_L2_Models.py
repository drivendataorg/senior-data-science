# -*- coding: utf-8 -*-

import os
import fnmatch
import time

# read parameters
import sys
validate = False
if len(sys.argv) >= 2:
    if sys.argv[1] == "validate":
        validate = True
    
# Start
start_time = time.time()
print ("Start Training L2 Models ...")
if validate:
    print ("...CV validation is activated")

# Generate list of files
files = []
for file_txt in os.listdir('.'):
    if fnmatch.fnmatch(file_txt, "SAS_L2_???_model*.py"):
        #print file_txt
        files.append(file_txt)
files.sort()
print ("...total {:} files".format(len(files)))
print

# execute files
def execute_file(file_txt):
    cmd="python ./"+file_txt
    if validate:
        cmd = cmd + " validate"
    print ("......Running: {:}".format(file_txt))
    tmp_start_time = time.time()
    os.system(cmd)
    print
    print ("......File executed in: {:.1f} min".format((time.time() - tmp_start_time)/60))
    print ("......Finished: {:}".format(file_txt))
    print

for file_txt in files:
    execute_file(file_txt)

print
print ("Average L2 Models ...")

files = ["SAS_L3_100_WA_vD2.py"]
for file_txt in files:
    execute_file(file_txt)

print
print ("Saved final submission in ../final_submission/L3_WA_vD2_submission.csv")
print
print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))