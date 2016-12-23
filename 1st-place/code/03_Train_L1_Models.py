# -*- coding: utf-8 -*-

import os
import fnmatch
import time

# Start
start_time = time.time()
print ("Start Training L1 Models ...")

# Generate list of files
files = []
for file_txt in os.listdir('.'):
    if fnmatch.fnmatch(file_txt, "SAS_L1_???_model*.py"):
        #print file_txt
        files.append(file_txt)
files.sort()
print ("...total {:} files".format(len(files)))
print

# execute files
def execute_file(file_txt):
    cmd="python ./"+file_txt
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
print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))