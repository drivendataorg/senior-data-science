# -*- coding: utf-8 -*-
# For number crunching
import numpy as np
import pandas as pd

# Misc
import json 
import os
from functools import partial

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

files_txt = ['L1_XGB_vH1t', 'L1_ET_vC1t', 'L1_NN_vG2t', 'L1_GLM_vA1t', 'L1_GLM_vA1'
            ] + ['L1_XGB_vH1t','L1_NN_vD2t','L1_XGB_vG1t','L1_XGB_vF1t','L1_NN_vE1t','L1_RF_vD1t','L1_XGB_vE1t'
            ] + ['L1_RF_vA1t','L1_NN_vA1t','L1_NN_vC1t','L1_RF_vH1t','L1_XGB_vA1t','L1_NN_vH2t'
            ] + ['L1_RF_vE1t', 'L1_XGB_vH1t', 'L1_ET_vH1t', 'L1_NN_vD2t', 'L1_RF_vH1t'
            ] + ['L1_ET_vA2t', 'L1_XGB_vD1t', 'L1_RF_vH1t', 'L1_NN_vD2t', 'L1_RF_vE1t']
files_txt = np.unique(files_txt)
  
all_train_x = fd.load_L1_train(files_txt)
all_train_y = fd.load_train_y([1,2,3,4,5,6,7,8,9,10])
all_train_y = all_train_y[np.isfinite(all_train_y.sum(1))]
all_test_x = fd.load_submissions(files_txt)
train_seq = fd.get_clean_sequences([1,2,3,4,5,6,7,8,9,10])
rows, tmp = fd.load_test(['ds_pir_v0'])

dataset = {'train_x':all_train_x, 'train_y':all_train_y, 'train_seq':train_seq, 'test_x':all_test_x}

# Add past data
dataset = {'train_x':np.c_[all_train_x, fd.get_past_data(all_train_x, train_seq, 1, -9999,200),
                           fd.get_past_data(all_train_x, train_seq, 2, -9999,200),
                           fd.get_future_data(all_train_x, train_seq, 1, -9999,200)], 
            'train_y':all_train_y, 
            'train_seq':train_seq, 
            'test_x':np.c_[all_test_x, fd.get_past_data(all_test_x, rows[:,0], 1, -9999),
                           fd.get_past_data(all_test_x, rows[:,0], 2, -9999),
                           fd.get_future_data(all_test_x, rows[:,0], 1, -9999)]}

# Define prediction function
def predict_model(train_x, train_y, test_x, test_y=None, class_weights=None, random_state=0):
    # Learn the ProbMC model 
    param = {'n_estimators':200, 'seed':1, 'missing':np.nan, 'nthread':11,
             'max_depth':6, 'learning_rate':0.04, 'colsample_bytree':0.2, 'subsample':0.9}
    model = fp.ProbMC_XGBoost(train_y.shape[1], bags=1, param=param) 
    model.fit(train_x, train_y)
    
    # Predict on the test instances
    test_predicted = model.predict_proba(test_x)
    
    return(test_predicted)


# Set Parameters
name_to_save = 'L2_XGB_vD1_0'
random_state = 1

prepbd_params = {'imputer_strategy':None}
f_preprocess = partial(fd.batch_preprocess, params=prepbd_params)

f_predict_model = partial(predict_model, random_state=random_state, class_weights=class_weights)


# Train & Predict L2_test                   
test_predicted = fp.L1_test(dataset, f_preprocess, f_predict_model, class_weights)


# Save files
directory = '../submissions/'
if not os.path.exists(directory):
    os.makedirs(directory)
# Submission
submission = pd.concat((pd.DataFrame(rows), pd.DataFrame(test_predicted)), axis=1)
submission.columns = ['record_id'] + ['start', 'end'] + activity_names
submission.to_csv('{}{}_submission.csv'.format(directory, name_to_save), index=False)

if validate:
    
    # Train & Predict L2_train
    valid_predicted, scores = fp.L1_train(dataset, f_preprocess, f_predict_model, class_weights, verbose=1)
    
    # L2_data
    valid_predicted = pd.DataFrame(valid_predicted)
    valid_predicted.columns = activity_names
    valid_predicted.to_csv('{}{}_valid.csv'.format(directory, name_to_save), index=False)
    np.savetxt('{}{}_score.csv'.format(directory, name_to_save), scores, delimiter=",")