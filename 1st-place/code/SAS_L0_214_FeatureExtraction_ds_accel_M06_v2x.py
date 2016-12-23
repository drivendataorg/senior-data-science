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

def extract_accel_v2A(df):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [dif, slo] of [x, y, z, comp, max, min] using splits
    '''
    num_ff = (3 + 3) * 2  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        df['comp'] = np.power((np.power(df['x'],2) + np.power(df['y'],2) + np.power(df['z'],2)), 0.5)
        df['max'] = df[['x','y','z']].apply(lambda x: np.max(x.values),axis=1)
        df['min'] = df[['x','y','z']].apply(lambda x: np.min(x.values),axis=1)
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups to calculate variations
        n_split = 4
        splits = []
        tmp_row_id = row_id
        for i in range(n_split,0,-1):
            splits = splits + [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Define functions
        feature_functions = [lambda x: (np.argmax(x) - np.argmin(x)) / len(x),
                             lambda x: (np.mean(x.iloc[splits[0]]) - np.mean(x.iloc[splits[-1]])) *100 / n_split]
        
        # Apply functions for each group
        result_row = np.array([])
        for f, feature_function in enumerate(feature_functions):
            result_row = np.append(result_row, df.apply(feature_function, axis=0).values)
        
    return result_row


def extract_accel_v2B(df):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [mean, std, min, median, max, dif, slo] of [|x|, |y|, |z|, amax, amin] using splits
    '''
    num_ff = (0 + 5) * 7  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        df['ax'] = abs(df['x'])
        df['ay'] = abs(df['y'])
        df['az'] = abs(df['z'])
        df['amax'] = df[['x','y','z']].apply(lambda x: np.max(abs(x.values)),axis=1)
        df['amin'] = df[['x','y','z']].apply(lambda x: np.min(abs(x.values)),axis=1)
        df= df.iloc[:,3:]
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups to calculate variations
        n_split = 4
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
                             lambda x: np.max(x),
                             lambda x: (np.argmax(x) - np.argmin(x)) / len(x),
                             lambda x: (np.mean(x.iloc[splits[0]]) - np.mean(x.iloc[splits[-1]])) *100 / n_split]
        
        # Apply functions for each group
        result_row = np.array([])
        for f, feature_function in enumerate(feature_functions):
            result_row = np.append(result_row, df.apply(feature_function, axis=0).values)
        
    return result_row


def extract_accel_v2C(df):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [mean, std, min, median, max, dif, slo] of [x^y, x^z, y^z] using splits
    '''
    num_ff = (0 + 3) * 7  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        df['xy'] = np.arctan(df['y']/df['x']) * 180/np.pi
        df['xz'] = np.arctan(df['z']/df['x']) * 180/np.pi
        df['yz'] = np.arctan(df['z']/df['y']) * 180/np.pi
        df= df.iloc[:,3:]
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups to calculate variations
        n_split = 4
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
                             lambda x: np.max(x),
                             lambda x: (np.argmax(x) - np.argmin(x)) / len(x),
                             lambda x: (np.mean(x.iloc[splits[0]]) - np.mean(x.iloc[splits[-1]])) *100 / n_split]
        
        # Apply functions for each group
        result_row = np.array([])
        for f, feature_function in enumerate(feature_functions):
            result_row = np.append(result_row, df.apply(feature_function, axis=0).values)
        
    return result_row


def extract_accel_v2D(df):
    '''
    This function extract features from collected 'accel' data.
    df: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [mean, std, min, median, max, dif, slo] of [x^y^z, z^x^y, y^z^x] using splits
    '''
    num_ff = (0 + 3) * 7  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        df['xyz'] = np.arctan(np.power((np.power(df['y'],2) + np.power(df['z'],2)), 0.5)/df['x']) * 180/np.pi
        df['zxy'] = np.arctan(np.power((np.power(df['x'],2) + np.power(df['y'],2)), 0.5)/df['z']) * 180/np.pi
        df['yzx'] = np.arctan(np.power((np.power(df['z'],2) + np.power(df['x'],2)), 0.5)/df['y']) * 180/np.pi
        df= df.iloc[:,3:]
    
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups to calculate variations
        n_split = 4
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
                             lambda x: np.max(x),
                             lambda x: (np.argmax(x) - np.argmin(x)) / len(x),
                             lambda x: (np.mean(x.iloc[splits[0]]) - np.mean(x.iloc[splits[-1]])) *100 / n_split]
        
        # Apply functions for each group
        result_row = np.array([])
        for f, feature_function in enumerate(feature_functions):
            result_row = np.append(result_row, df.apply(feature_function, axis=0).values)
        
    return result_row


# Column nomes
series_name = 'accel_M06'

column_names_A = []
for i2 in ['x','y','z','comp','max','min']:
    for i3 in ['dif','slo']:
        column_names_A.append('{0}_{1}_{2}_{3}'.format(series_name, 'v2', i2, i3)) 

column_names_B = []
for i2 in ['ax','ay','az','amax','amin']:
    for i3 in ['mean','std','min','median','max','dif','slo']:
        column_names_B.append('{0}_{1}_{2}_{3}'.format(series_name, 'v2', i2, i3)) 

column_names_C = []
for i2 in ['xy','xz','yz']:
    for i3 in ['mean','std','min','median','max','dif','slo']:
        column_names_C.append('{0}_{1}_{2}_{3}'.format(series_name, 'v2', i2, i3)) 

column_names_D = []
for i2 in ['xyz','zxy','yzx']:
    for i3 in ['mean','std','min','median','max','dif','slo']:
        column_names_D.append('{0}_{1}_{2}_{3}'.format(series_name, 'v2', i2, i3)) 


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
        
        
        #### Generate ds_accel_M06_v2A/B/C/D (4 files)

        # Use the sequence loader to load the data from the directory. 
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()
        data.accel_movSum(6,6) # Transform accel data to mov_sum   
        
        # iterate time-window data
        rows_A = []
        rows_B = []
        rows_C = []
        rows_D = []
        for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
            
            # Append the row to the full set of features
            rows_A.append(extract_accel_v2A(accel.copy()))
            rows_B.append(extract_accel_v2B(accel.copy()))
            rows_C.append(extract_accel_v2C(accel.copy()))
            rows_D.append(extract_accel_v2D(accel.copy()))
            
            # Report progress 
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

                if np.mod(ri + 1, 500) == 0:
                    print
        
        # Wrap data
        rows_A = np.vstack(rows_A)
        rows_B = np.vstack(rows_B)
        rows_C = np.vstack(rows_C)
        rows_D = np.vstack(rows_D)

        # save data
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_A = pd.DataFrame(rows_A)
        df_A.columns = column_names_A
        df_A.to_csv('{}ds_accel_M06_v2A.csv'.format(directory), index=False)
        
        df_B = pd.DataFrame(rows_B)
        df_B.columns = column_names_B
        df_B.to_csv('{}ds_accel_M06_v2B.csv'.format(directory), index=False)
        
        df_C = pd.DataFrame(rows_C)
        df_C.columns = column_names_C
        df_C.to_csv('{}ds_accel_M06_v2C.csv'.format(directory), index=False)
        
        df_D = pd.DataFrame(rows_D)
        df_D.columns = column_names_D
        df_D.to_csv('{}ds_accel_M06_v2D.csv'.format(directory), index=False)

        # Print progress
        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))




