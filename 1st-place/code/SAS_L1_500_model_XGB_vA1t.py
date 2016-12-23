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
data_source =['ds_accel_M00_v1A_s1','ds_accel_M00_v1B_s1',
              'ds_accel_M00_v2A','ds_accel_M00_v2B','ds_accel_M00_v2C','ds_accel_M00_v2D',
              'ds_rssi_v0','ds_pir_v0',
              'ds_video_fig_M00_v1A_s1','ds_video_fig_M00_v1B_s1','ds_video_fig_M00_v2A','ds_video_fig_M00_v2B',
              'ds_video_mov_M00_v1A_s1','ds_video_mov_M00_v1B_s1','ds_video_mov_M00_v2A','ds_video_mov_M00_v2B',
              'ds_video_sta_M00_v1A_s1','ds_video_sta_M00_v1B_s1','ds_video_sta_M00_v2A','ds_video_sta_M00_v2B']

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
dataset = {'train_x':np.c_[all_train_x, fd.get_past_data(all_train_x, train_seq, 1, -9999, 200),
                           fd.get_past_data(all_train_x, train_seq, 2, -9999, 200),
                           fd.get_future_data(all_train_x, train_seq, 1, -9999, 200)], 
            'train_y':all_train_y, 
            'train_seq':train_seq, 
            'test_x':np.c_[all_test_x, fd.get_past_data(all_test_x, rows[:,0], 1, -9999),
                           fd.get_past_data(all_test_x, rows[:,0], 2, -9999),
                           fd.get_future_data(all_test_x, rows[:,0], 1, -9999)]}

# Define prediction function
def predict_model(train_x, train_y, test_x, test_y=None, class_weights=None, random_state=0):
    # Learn the ProbMC model 
    param = {'n_estimators':200, 'seed':1, 'missing':np.nan, 'nthread':10,
             'max_depth':6, 'learning_rate':0.04, 'colsample_bytree':0.2, 'subsample':0.9}
    model = fp.ProbMC_XGBoost(train_y.shape[1], bags=1, param=param) 
    model.fit(train_x, train_y)
    
    # Predict on the test instances
    test_predicted = model.predict_proba(test_x)
    
    return(test_predicted)


# Set Parameters
name_to_save = 'L1_XGB_vA1t'
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