# -*- coding: utf-8 -*-
# For number crunching
import numpy as np
import pandas as pd

# Misc
import json 
import os

# load custom libraries
import func_data as fd
import func_predictors as fp

# read parameters
import sys
validate = False
if len(sys.argv) >= 2:
    if sys.argv[1] == "validate":
        validate = True


# Load in data
activity_names = json.load(open('../public_data/annotations.json', 'r'))
class_weights = np.asarray(json.load(open('../public_data/class_weights.json', 'r')))
rows, tmp = fd.load_test(['ds_pir_v0'])

# Load L2 predictions
files_txt = ['L2_XGB_vD1_1','L2_NN_vC1_0','L2_ET_vC1_0','L2_NN_vD1_0','L2_XGB_vD1_0','L2_XGB_vB2_1',
'L2_NN_vD1_0','L2_XGB_vC2_0']

# Set weights
weights = np.ones(len(files_txt))

if validate:
    validated = np.zeros((16124,20))
    for i, file_txt in enumerate(files_txt):
        validated = validated + fd.load_L1_train([file_txt]) * weights[i]
    validated = validated / np.sum(weights)
    
    all_train_y = fd.load_train_y([1,2,3,4,5,6,7,8,9,10])
    all_train_y = all_train_y[np.isfinite(all_train_y.sum(1))]
    print ("Final Submission local-CV score: {}".format(fp.brier_score(all_train_y, validated, class_weights)))


# Predict L3 test data
predicted = np.zeros((16600,20))
for i, file_txt in enumerate(files_txt):
    predicted = predicted + fd.load_submissions([file_txt]) * weights[i]
predicted = predicted / np.sum(weights)


name_to_save = 'L3_WA_vD2'

# Save files
directory = '../final_submission/'
if not os.path.exists(directory):
    os.makedirs(directory)
# Submission
submission = pd.concat((pd.DataFrame(rows), pd.DataFrame(predicted)), axis=1)
submission.columns = ['record_id'] + ['start', 'end'] + activity_names
submission.to_csv('{}{}_submission.csv'.format(directory, name_to_save), index=False)

