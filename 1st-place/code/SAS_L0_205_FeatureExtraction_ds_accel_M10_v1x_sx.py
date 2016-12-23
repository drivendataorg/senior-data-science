# -*- coding: utf-8 -*-

# For number crunching
import numpy as np
import pandas as pd

# Misc
import os

# Functions libraries
import func_visualise_data as fvd

import warnings
warnings.filterwarnings('ignore')

# Define functions

def extract_accel_v1A(df, n_split):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean, std, min, median, max] of [x, y, z] for each 'n_splits' group
    '''
    num_ff = (3 + 0) * n_split * 5  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        for i in range(n_split,0,-1):
            splits = splits + [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Define functions
        feature_functions = [lambda x: np.mean(x), 
                             lambda x: np.std(x), 
                             lambda x: np.min(x),
                             lambda x: np.median(x),
                             lambda x: np.max(x)]
        
        # Apply functions for each group
        result_row = np.array([])
        for split in splits:
            for f, feature_function in enumerate(feature_functions):
                result_row = np.append(result_row, df.iloc[split].apply(feature_function, axis=0).values)
        
    return result_row


def extract_accel_v1B(df, n_split):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean, std, min, median, max] of [comp, max_id, max, min_id, min] 
        for each 'n_splits' group
    '''
    num_ff = (0 + 5) * n_split * 5  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else:
        
        # feature engineering
        df['comp'] = np.power((np.power(df['x'],2) + np.power(df['y'],2) + np.power(df['z'],2)), 0.5)
        df['max_id'] = df[['x','y','z']].apply(lambda x: np.argmax(abs(x.values)),axis=1)
        df['max'] = df[['x','y','z']].apply(lambda x: np.max(abs(x.values)),axis=1)
        df['min_id'] = df[['x','y','z']].apply(lambda x: np.argmin(abs(x.values)),axis=1)
        df['min'] = df[['x','y','z']].apply(lambda x: np.min(abs(x.values)),axis=1)
        df = df.iloc[:,3:]
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        for i in range(n_split,0,-1):
            splits = splits + [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Define functions
        feature_functions = [lambda x: np.mean(x), 
                             lambda x: np.std(x), 
                             lambda x: np.min(x),
                             lambda x: np.median(x),
                             lambda x: np.max(x)]
        
        # Apply functions for each group
        result_row = np.array([])
        for split in splits:
            for f, feature_function in enumerate(feature_functions):
                result_row = np.append(result_row, df.iloc[split].apply(feature_function, axis=0).values)
        
    return result_row


# Column nomes
column_names_A_s1 = []
for i1 in range(1,(1+1)):
    for i2 in ['x','y','z']:
        for i3 in ['mean','std','min','median','max',]:
            column_names_A_s1.append('{0}_{1}of1_{2}_{3}'.format('accel_M10', i1, i2, i3)) 

column_names_B_s1 = []
for i1 in range(1,(1+1)):
    for i2 in ['comp','max_id','max','min_id','min']:
        for i3 in ['mean','std','min','median','max',]:
            column_names_B_s1.append('{0}_{1}of1_{2}_{3}'.format('accel_M10', i1, i2, i3)) 

column_names_A_s2 = []
for i1 in range(1,(1+2)):
    for i2 in ['x','y','z']:
        for i3 in ['mean','std','min','median','max',]:
            column_names_A_s2.append('{0}_{1}of2_{2}_{3}'.format('accel_M10', i1, i2, i3)) 

column_names_B_s2 = []
for i1 in range(1,(1+2)):
    for i2 in ['comp','max_id','max','min_id','min']:
        for i3 in ['mean','std','min','median','max',]:
            column_names_B_s2.append('{0}_{1}of2_{2}_{3}'.format('accel_M10', i1, i2, i3))  

"""
Iterate over all training/testing directories
"""

for train_test in ('train', 'test', ): 
    
    # Print msg 
    if train_test is 'train': 
        print ('Extracting features from training data.\n')
    else: 
        print ('\n\n\nExtracting features from testing data.\n')
    
    # iterate folders
    for fi, file_id in enumerate(sorted(os.listdir('../public_data/{}/'.format(train_test)))):
        stub_name = str(file_id).zfill(5)

        if train_test == 'train' or np.mod(fi, 50) == 0:
             print ("Starting feature extraction for {}/{}".format(train_test, stub_name))
        
        
        #### Generate ds_accel_M10_v1A/B_s1 / ds_accel_M10_v1A/B_s2 (4 files)

        # Use the sequence loader to load the data from the directory. 
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()
        data.accel_movSum(10,10) # Transform accel data to mov_sum        
        
        # iterate time-window data
        rows_A_s1 = []
        rows_A_s2 = []
        rows_B_s1 = []
        rows_B_s2 = []
        for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
            
            # Append the row to the full set of features
            rows_A_s1.append(extract_accel_v1A(accel, 1))
            rows_A_s2.append(extract_accel_v1A(accel, 2))
            rows_B_s1.append(extract_accel_v1B(accel, 1))
            rows_B_s2.append(extract_accel_v1B(accel, 2))
            
            # Report progress 
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

                if np.mod(ri + 1, 500) == 0:
                    print
        
        # Wrap data
        rows_A_s1 = np.vstack(rows_A_s1)
        rows_A_s2 = np.vstack(rows_A_s2)
        rows_B_s1 = np.vstack(rows_B_s1)
        rows_B_s2 = np.vstack(rows_B_s2)

        # save data
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_A_s1 = pd.DataFrame(rows_A_s1)
        df_A_s1.columns = column_names_A_s1
        df_A_s1.to_csv('{}ds_accel_M10_v1A_s1.csv'.format(directory), index=False)
        
        df_A_s2 = pd.DataFrame(rows_A_s2)
        df_A_s2.columns = column_names_A_s2
        df_A_s2.to_csv('{}ds_accel_M10_v1A_s2.csv'.format(directory), index=False)
        
        df_B_s1 = pd.DataFrame(rows_B_s1)
        df_B_s1.columns = column_names_B_s1
        df_B_s1.to_csv('{}ds_accel_M10_v1B_s1.csv'.format(directory), index=False)
        
        df_B_s2 = pd.DataFrame(rows_B_s2)
        df_B_s2.columns = column_names_B_s2
        df_B_s2.to_csv('{}ds_accel_M10_v1B_s2.csv'.format(directory), index=False)

        # Print progress
        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))