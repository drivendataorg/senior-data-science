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


# Load in data
activity_names = json.load(open('../public_data/annotations.json', 'r'))
class_weights = np.asarray(json.load(open('../public_data/class_weights.json', 'r')))
sequence_train = [1,2,3,4,5,6,7,8,9,10]
data_source =['ds_rssi_M32_v1C_s2', 'ds_pir_M2s_v1', 'ds_rssi_v0', 'ds_accel_M10_v1B_s2', 'ds_video_sta_M00_v1A_s2', 
              'ds_accel_v0', 'ds_accel_M00_v1B_s1']

all_train_x, all_train_y, train_seq = fd.load_sequences(sequence_train, data_source)
rows, all_test_x = fd.load_test(data_source)

# Overwrite target
train_y_has_annotation = np.isfinite(all_train_y.sum(1)) # These are the samples to use later
all_train_y = fd.load_target_location(sequence_train)
all_train_y = all_train_y[train_y_has_annotation]
all_train_x = all_train_x[train_y_has_annotation]
train_seq = train_seq[train_y_has_annotation]

# Preprocess the whole data
prepwd_params = {'remove_nan_targets':False, 'imputer_strategy':'most_frequent'}
all_train_x, all_train_y, train_seq, rows, all_test_x = fd.whole_preprocess(all_train_x, 
             all_train_y, train_seq, rows, all_test_x, params=prepwd_params)
dataset = {'train_x':all_train_x, 'train_y':all_train_y, 'train_seq':train_seq, 'test_x':all_test_x}


# Define prediction function
def predict_model(train_x, train_y, test_x, test_y=None, class_weights=None, random_state=0):
    
    # Learn the ProbMC model 
    params = {'n_jobs':-1, 'random_state':random_state, 
              'n_estimators':1000, 'max_features':0.75, 'min_samples_leaf':1}

    model = fp.PMC_MultiTaskExtraTreesRegressor(train_y.shape[1], bags=1, params=params) 
    sample_weight = None    
    model.fit(train_x, train_y, sample_weight)
    
    # Predict on the test instances
    test_predicted = model.predict_proba(test_x)
    
    # Learn the ProbMC model 
    params = {'n_jobs':12, 'random_state':random_state, 
              'n_estimators':1000, 'max_features':0.45, 'min_samples_leaf':1}

    model = fp.PMC_MultiTaskExtraTreesRegressor(train_y.shape[1], bags=1, params=params) 
    sample_weight = None    
    model.fit(train_x, train_y, sample_weight)
    
    # Predict on the test instances
    test_predicted = test_predicted + model.predict_proba(test_x)
    
    return(test_predicted / 2)


# Set Parameters
name_to_save = 'pl_L1_ET_vA1'
random_state = 1

prepbd_params = {'imputer_strategy':None}
f_preprocess = partial(fd.batch_preprocess, params=prepbd_params)

f_predict_model = partial(predict_model, random_state=random_state, class_weights=None)


# Train & Predict L1_train
valid_predicted, scores = fp.L1_train(dataset, f_preprocess, f_predict_model)
                        
    
# Train & Predict L1_test                   
test_predicted = fp.L1_test(dataset, f_preprocess, f_predict_model, class_weights)


# Save files
directory = '../predict_location/'
if not os.path.exists(directory):
    os.makedirs(directory)
# Submission
room_names = json.load(open('../public_data/rooms.json', 'r'))
submission = pd.concat((pd.DataFrame(rows), pd.DataFrame(test_predicted)), axis=1)
submission.columns = ['record_id'] + ['start', 'end'] + room_names
submission.to_csv('{}{}_submission.csv'.format(directory, name_to_save), index=False)

# L1_data
valid_predicted = pd.DataFrame(valid_predicted)
valid_predicted.columns = room_names
valid_predicted.to_csv('{}{}_valid.csv'.format(directory, name_to_save), index=False)
np.savetxt('{}{}_score.csv'.format(directory, name_to_save), scores, delimiter=",")
