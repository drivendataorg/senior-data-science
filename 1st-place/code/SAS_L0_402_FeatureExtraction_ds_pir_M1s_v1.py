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

def extract_pir(df):
    '''
    This function extract features from collected 'pir' data.
    df: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [mean] of [bath,bed1,bed2,hall,kitchen,living,stairs,study,toilet]
        and 'sum' of them as ['sumON']
    '''
    num_ff = 10  # Number of final features
    
    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # Apply functions for each group
        result_row = np.array([])              
        result_row = np.append(result_row, df.apply(lambda x: np.mean(x), axis=0).values)
        result_row = np.append(result_row, np.nansum(result_row))  # Add feature 'sumON': How many 'pir' are on?

    return result_row


def extract_pir_colnames():
    '''
    This function return column names for extracted features from 'pir' data.
    '''
    # Column names
    tmp_column_names = []
    for i2 in ['bath','bed1','bed2','hall','kitchen','living','stairs','study','toilet','sumON']:
        tmp_column_names.append('{0}_{1}'.format('pir', i2)) 
    
    return tmp_column_names


def extract_video(vid_lr, vid_k, vid_h):
    '''
    This function extract features from collected 'video' data used as 'pir' device.
    vid_lr, vid_k, vid_h: pandas dataFrame of data for a given time-window (1s)
    return: numpy 2D-array of new features:
        [mean] of TRES/FALSE[vid_lr, vid_k, vid_h]
    '''
    result_row = np.array([])
    
    df = vid_lr
    if df.shape[0] == 0:
        result_row = np.append(result_row, 0)
    else:
        result_row = np.append(result_row, np.mean(df.apply(lambda x: sum(np.isnan(x)) != df.shape[1], axis=0)))
    
    df = vid_k
    if df.shape[0] == 0:
        result_row = np.append(result_row, 0)
    else:
        result_row = np.append(result_row, np.mean(df.apply(lambda x: sum(np.isnan(x)) != df.shape[1], axis=0)))
    
    df = vid_h
    if df.shape[0] == 0:
        result_row = np.append(result_row, 0)
    else:
        result_row = np.append(result_row, np.mean(df.apply(lambda x: sum(np.isnan(x)) != df.shape[1], axis=0)))
    
    return result_row


def extract_video_colnames():
    '''
    This function return column names for extracted features from 'video' data
    '''
    # Column names
    tmp_column_names = []
    for i2 in ['livingVideo','kitchenVideo','hallVideo']:
        tmp_column_names.append('{0}_{1}'.format('pir', i2)) 
    
    return tmp_column_names


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
        
        # Use the sequence loader to load the data from the directory. 
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()
        
        # iterate time-window data
        rows = []
        for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
            
            # Add features
            row = np.array([])
            pir_row = extract_pir(pir)
            row = np.append(row, pir_row)
            video_row = extract_video(vid_lr, vid_k, vid_h)
            row = np.append(row, video_row)
            
            # Append row to the full set of features
            rows.append(row)
            
            # Report progress 
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

                if np.mod(ri + 1, 500) == 0:
                    print
        
        # Wrap data
        rows = np.vstack(rows)
        column_names = extract_pir_colnames() + extract_video_colnames()
        
        # Normalize data using 1s-future & 1s-past rows
        rows = rows + np.r_[rows[1:], rows[-1:]] + np.r_[rows[:1], rows[:-1]]
        
        # save data
        df = pd.DataFrame(rows)
        df.columns = column_names
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv('{}ds_pir_M1s_v1.csv'.format(directory), index=False)
        
        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))