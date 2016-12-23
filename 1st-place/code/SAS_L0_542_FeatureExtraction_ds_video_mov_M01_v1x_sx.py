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

def extract_video_mov_v1A(df, n_split):
    '''
    This function extract features from collected 'video' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean, std, min, median, max, sum, asum] of [2D features (i)-(i-1)] for each 'n_splits' group
    '''
    num_ff = (6 + 0) * n_split * 7  # Number of final features

    if df.shape[0] < 2:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        new_df=[]
        for i_row in range(1, df.shape[0]):
            new_df.append((df.iloc[i_row,:] - df.iloc[i_row-1,:]).values)
        new_df = pd.DataFrame(np.vstack(new_df))
        new_df.columns = df.columns
        df = new_df.iloc[:,:6]
        
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
                             lambda x: np.max(x),
                             lambda x: np.sum(x),
                             lambda x: np.sum(abs(x))]
        
        # Apply functions for each group
        result_row = np.array([])
        for split in splits:
            for f, feature_function in enumerate(feature_functions):
                result_row = np.append(result_row, df.iloc[split].apply(feature_function, axis=0).values)
        
    return result_row


def extract_video_mov_v1B(df, n_split):
    '''
    This function extract features from collected 'video' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean, std, min, median, max, sum, asum] of [2D features (i)-(i-1)] for each 'n_splits' group
    '''
    num_ff = (9 + 0) * n_split * 7  # Number of final features

    if df.shape[0] < 2:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
        
    else: 
        
        # feature engineering
        new_df=[]
        for i_row in range(1, df.shape[0]):
            new_df.append((df.iloc[i_row,:] - df.iloc[i_row-1,:]).values)
        new_df = pd.DataFrame(np.vstack(new_df))
        new_df.columns = df.columns
        df = new_df.iloc[:,6:]
        
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
                             lambda x: np.max(x),
                             lambda x: np.sum(x),
                             lambda x: np.sum(abs(x))]
        
        # Apply functions for each group
        result_row = np.array([])
        for split in splits:
            for f, feature_function in enumerate(feature_functions):
                result_row = np.append(result_row, df.iloc[split].apply(feature_function, axis=0).values)
        
    return result_row


# Column nomes
column_names_A_s1 = []
for i0 in ['vid_lr','vid_k','vid_h']:
    for i1 in range(1,(1+1)):
        for i2 in ['2Df1','2Df2','2Df3','2Df4','2Df5','2Df6']:
            for i3 in ['mean','std','min','median','max','sum','asum']:
                column_names_A_s1.append('video_mov_M01_{0}_{1}of1_{2}_{3}'.format(i0, i1, i2, i3)) 

column_names_B_s1 = []
for i0 in ['vid_lr','vid_k','vid_h']:
    for i1 in range(1,(1+1)):
        for i2 in ['3Df1','3Df2','3Df3','3Df4','3Df5','3Df6','3Df7','3Df8','3Df9']:
            for i3 in ['mean','std','min','median','max','sum','asum']:
                column_names_B_s1.append('video_mov_M01_{0}_{1}of1_{2}_{3}'.format(i0, i1, i2, i3)) 

column_names_A_s2 = []
for i0 in ['vid_lr','vid_k','vid_h']:
    for i1 in range(1,(1+2)):
        for i2 in ['2Df1','2Df2','2Df3','2Df4','2Df5','2Df6']:
            for i3 in ['mean','std','min','median','max','sum','asum']:
                column_names_A_s2.append('video_mov_M01_{0}_{1}of1_{2}_{3}'.format(i0, i1, i2, i3)) 

column_names_B_s2 = []
for i0 in ['vid_lr','vid_k','vid_h']:
    for i1 in range(1,(1+2)):
        for i2 in ['3Df1','3Df2','3Df3','3Df4','3Df5','3Df6','3Df7','3Df8','3Df9']:
            for i3 in ['mean','std','min','median','max','sum','asum']:
                column_names_B_s2.append('video_fmov_M01_{0}_{1}of1_{2}_{3}'.format(i0, i1, i2, i3)) 


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
        
        
        #### Generate ds_video_mov_M01_v1A/B_s1 / ds_video_mov_M01_v1A/B_s2 (4 files)

        # Use the sequence loader to load the data from the directory. 
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()
        data.video_movSum(1,1) # Transform video data to mov_sum 
        
        # iterate time-window data
        rows_A_s1 = []
        rows_A_s2 = []
        rows_B_s1 = []
        rows_B_s2 = []

        for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
            
            # Append the row to the full set of features
            row = np.array([])
            row = np.append(row, extract_video_mov_v1A(vid_lr.copy(), 1))
            row = np.append(row, extract_video_mov_v1A(vid_k.copy(), 1))
            row = np.append(row, extract_video_mov_v1A(vid_h.copy(), 1))
            rows_A_s1.append(row)
            
            row = np.array([])
            row = np.append(row, extract_video_mov_v1A(vid_lr.copy(), 2))
            row = np.append(row, extract_video_mov_v1A(vid_k.copy(), 2))
            row = np.append(row, extract_video_mov_v1A(vid_h.copy(), 2))
            rows_A_s2.append(row)
            
            row = np.array([])
            row = np.append(row, extract_video_mov_v1B(vid_lr.copy(), 1))
            row = np.append(row, extract_video_mov_v1B(vid_k.copy(), 1))
            row = np.append(row, extract_video_mov_v1B(vid_h.copy(), 1))
            rows_B_s1.append(row)
            
            row = np.array([])
            row = np.append(row, extract_video_mov_v1B(vid_lr.copy(), 2))
            row = np.append(row, extract_video_mov_v1B(vid_k.copy(), 2))
            row = np.append(row, extract_video_mov_v1B(vid_h.copy(), 2))
            rows_B_s2.append(row)
            
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
        df_A_s1.to_csv('{}ds_video_mov_M01_v1A_s1.csv'.format(directory), index=False)
        
        df_A_s2 = pd.DataFrame(rows_A_s2)
        df_A_s2.columns = column_names_A_s2
        df_A_s2.to_csv('{}ds_video_mov_M01_v1A_s2.csv'.format(directory), index=False)
        
        df_B_s1 = pd.DataFrame(rows_B_s1)
        df_B_s1.columns = column_names_B_s1
        df_B_s1.to_csv('{}ds_video_mov_M01_v1B_s1.csv'.format(directory), index=False)
        
        df_B_s2 = pd.DataFrame(rows_B_s2)
        df_B_s2.columns = column_names_B_s2
        df_B_s2.to_csv('{}ds_video_mov_M01_v1B_s2.csv'.format(directory), index=False)

        # Print progress
        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
