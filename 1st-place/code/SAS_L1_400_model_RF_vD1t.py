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
data_source =['ds_accel_M01_v2A','ds_accel_M01_v2B','ds_accel_M01_v2C','ds_accel_M01_v2D',
              'ds_accel_M10_v2A','ds_accel_M10_v2B','ds_accel_M10_v2C','ds_accel_M10_v2D',
              'ds_pir_M0s_v1',
              'ds_rssi_M16_v1A_s1','ds_rssi_M16_v1B_s1','ds_rssi_M16_v1C_s1',
              'ds_rssi_M16_v1D_s1','ds_rssi_M16_v1E_s1','ds_rssi_M16_v1F_s1',              
              'ds_video_fig_M01_v1A_s1','ds_video_fig_M01_v1B_s1',
              'ds_video_mov_M01_v1A_s2','ds_video_mov_M01_v1B_s2',
              'ds_video_sta_M01_v1A_s2','ds_video_sta_M01_v1B_s2',
              'ds_video_fig_M06_v1A_s1','ds_video_fig_M06_v1B_s1',
              'ds_video_mov_M06_v1A_s2','ds_video_mov_M06_v1B_s2',
              'ds_video_sta_M06_v1A_s2','ds_video_sta_M06_v1B_s2']

all_train_x, all_train_y, train_seq = fd.load_sequences(sequence_train, data_source)
rows, all_test_x = fd.load_test(data_source)

# Preprocess the whole data
prepwd_params = {'remove_nan_targets':True, 'imputer_strategy':'most_frequent'}
all_train_x, all_train_y, train_seq, rows, all_test_x = fd.whole_preprocess(all_train_x, 
             all_train_y, train_seq, rows, all_test_x, params=prepwd_params)

# Add preprocessed data)
all_train_x = np.concatenate((all_train_x, pd.read_csv('../predict_location/pl_L1_ET_vA1_valid.csv').values), axis=1)
all_test_x = np.concatenate((all_test_x, 
             pd.read_csv('../predict_location/pl_L1_ET_vA1_submission.csv').values[:, 3:]), axis=1)
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
    params = {'n_jobs':-1, 'random_state':random_state, 'bootstrap':False,
              'n_estimators':500, 'max_features':0.12, 'min_samples_leaf':5}
    model = fp.PMC_MultiTaskRandomForest(train_y.shape[1], bags=1, params=params) 
    sample_weight = None    
    model.fit(train_x, train_y, sample_weight)
    
    # Predict on the test instances
    test_predicted = model.predict_proba(test_x)
    
    return(test_predicted)


# Set Parameters
name_to_save = 'L1_RF_vD1t'
random_state = 1

prepbd_params = {'imputer_strategy':None}
f_preprocess = partial(fd.batch_preprocess, params=prepbd_params)

f_predict_model = partial(predict_model, random_state=random_state, class_weights=class_weights)


# Trials
if False:
    fp.L1_trial(dataset, 1, f_preprocess, f_predict_model, class_weights, verbose=2)


# Train & Predict L1_train
valid_predicted, scores = fp.L1_train(dataset, f_preprocess, f_predict_model, class_weights, verbose=1)
                        
    
# Train & Predict L1_test                   
test_predicted = fp.L1_test(dataset, f_preprocess, f_predict_model, class_weights)


# Save files
directory = '../submissions/'
if not os.path.exists(directory):
    os.makedirs(directory)
# Submission
submission = pd.concat((pd.DataFrame(rows), pd.DataFrame(test_predicted)), axis=1)
submission.columns = ['record_id'] + ['start', 'end'] + activity_names
submission.to_csv('{}{}_submission.csv'.format(directory, name_to_save), index=False)

# L1_data
valid_predicted = pd.DataFrame(valid_predicted)
valid_predicted.columns = activity_names
valid_predicted.to_csv('{}{}_valid.csv'.format(directory, name_to_save), index=False)
np.savetxt('{}{}_score.csv'.format(directory, name_to_save), scores, delimiter=",")

'''
    Fold 1 score = 0.14476
    Fold 2 score = 0.15913
    Fold 3 score = 0.21064
    Fold 4 score = 0.13169
    Fold 5 score = 0.14213
    Fold 6 score = 0.20982
    Fold 7 score = 0.17335
    Fold 8 score = 0.18688
    Fold 9 score = 0.19088
    Fold 10 score = 0.17434

Mean score = 0.17236
Total score = 0.17205
Total execution time: 113.8 min
'''