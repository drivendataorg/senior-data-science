# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from math import atan, sqrt, fabs
import os


######
# The goal of this section of code is to perform the feature engineering on the train set. 
# It consists of : 
# - adding features to the train dataset
# - normalizing features by user
# - multiplying by -1 the accelerometer data for certain users
# - adding features of lags and leads to the train dataset
######

input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_before_feature_engineering.csv')
train_df = pd.read_csv(filepath_or_buffer = input_path,encoding='utf-8')


# Add features to the train dataset, based on simple formulas
train_df['video_room_centre_2d_x_length'] = train_df['video_room_bb_2d_br_x_mean']-train_df['video_room_bb_2d_tl_x_mean']
train_df['video_room_centre_2d_y_length'] = train_df['video_room_bb_2d_br_y_mean']-train_df['video_room_bb_2d_tl_y_mean']
train_df['video_room_bb_3d_brb_x_length'] = (train_df['video_room_bb_3d_brb_x_mean']-train_df['video_room_bb_3d_flt_x_mean'])
train_df['video_room_bb_3d_brb_y_length'] = (train_df['video_room_bb_3d_flt_y_mean']-train_df['video_room_bb_3d_brb_y_mean'])
train_df['room_angle_turn_x'] = ((train_df['video_room_bb_3d_brb_x_mean']-train_df['video_room_centre_3d_x_mean'])/(train_df['video_room_bb_3d_brb_z_mean']-train_df['video_room_centre_3d_z_mean'])).apply(atan)

# Normalize video_room features by user
video_cols = [col for col in train_df.columns if 'video_room_' in col]
for col in video_cols:
    train_df[col] = train_df.groupby("original_dataset")[col].apply(lambda x: (x - x.mean()) / x.std())
    
# Mulitply acceleration by -1 for users that seem to have worn their accelerometer upside down
acceleration_x_cols = [col for col in train_df.columns if 'acceleration_x' in col]
acceleration_y_cols = [col for col in train_df.columns if 'acceleration_y' in col]
for i in [2,5,6,7,8,10]:
    for col in acceleration_x_cols:
        if not('acceleration_x_std' in col):
            train_df.loc[train_df.original_dataset == i, col] = train_df.loc[train_df.original_dataset == i, col].apply(lambda x: x*-1)
for i in [2,5,6,7,8,10]:
    for col in acceleration_y_cols:
        if not('acceleration_y_std' in col):
            train_df.loc[train_df.original_dataset == i, col] = train_df.loc[train_df.original_dataset == i, col].apply(lambda x: x*-1)
# Add features indicating how many seconds have passed since the last change of value in video_room/original_dataset
# Indeed, we will add lags/leads to our train set: so the first lines will have many Nan values in their lags: it's interesting to add a feature that may explain this behaviour
output_df = pd.DataFrame(columns=['time_since_change_of_record_id','time_since_change_of_video_room'],index = train_df.index)

for index in train_df.index:

    record_id_has_chged_over_10_seconds = False
    for j in range(1,11):
        if (train_df.final_dataset[index] != train_df.final_dataset.shift(j)[index]):
            record_id_has_chged_over_10_seconds = True
            break
    if record_id_has_chged_over_10_seconds:
        output_df.loc[index]['time_since_change_of_record_id'] = j
    else: 
        output_df.loc[index]['time_since_change_of_record_id'] = 11

    video_room_has_chged_over_10_seconds = False
    if ( pd.isnull(train_df.video_room.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 0
    elif (pd.isnull(train_df.video_room.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    elif not(train_df.video_room[index] == train_df.video_room.shift(1)[index]):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    elif not((train_df.final_dataset[index] == train_df.final_dataset.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    else: 
        k=0
        while ( (train_df.video_room[index] == train_df.video_room.shift(k)[index] ) and (train_df.final_dataset[index] == train_df.final_dataset.shift(k)[index])):
            output_df.loc[index]['time_since_change_of_video_room'] = k
            k = k+1
result = pd.concat([train_df, output_df], axis=1, join='inner')
   
# We add features of lag/lead to train dataset: lags and leads up to 10 seconds
list_1_of_columns_to_lag = ['acceleration_x','acceleration_y','acceleration_z','rssi_Kitchen_AP','rssi_Lounge_AP','rssi_Upstairs_AP','rssi_Study_AP']
list_2_of_columns_to_lag = ['mean','std','min','median','max']
for name_1 in list_1_of_columns_to_lag:
    for name_2 in list_2_of_columns_to_lag:
        for lag in range(1,11):
            result[name_1+'_'+name_2+'_lag_'+str(lag)]=result[name_1+'_'+name_2].shift(periods=lag)
            mask = (result.final_dataset != result.final_dataset.shift(lag))
            result[name_1+'_'+name_2+'_lag_'+str(lag)][mask] = np.nan
        for lead in range(1,11):
            result[name_1+'_'+name_2+'_lead_'+str(lead)]=result[name_1+'_'+name_2].shift(periods=-lead)
            mask = (result.final_dataset != result.final_dataset.shift(-lead))
            result[name_1+'_'+name_2+'_lead_'+str(lead)][mask] = np.nan


list_3_of_columns_to_lag = ['video_room_centre_2d_x','video_room_centre_2d_y','video_room_bb_2d_br_x','video_room_bb_2d_br_y','video_room_bb_2d_tl_x','video_room_bb_2d_tl_y',
                    'video_room_centre_3d_x','video_room_centre_3d_y','video_room_centre_3d_z','video_room_bb_3d_brb_x',
                            'video_room_bb_3d_brb_y','video_room_bb_3d_brb_z','video_room_bb_3d_flt_x','video_room_bb_3d_flt_y', 'video_room_bb_3d_flt_z']
for name_1 in list_3_of_columns_to_lag:
    for name_2 in list_2_of_columns_to_lag:
        for lag in range(1,11):
            result[name_1+'_'+name_2+'_lag_'+str(lag)]=result[name_1+'_'+name_2].shift(periods=lag)
            mask = ((result.final_dataset != result.final_dataset.shift(lag)) |  (result.video_room != result.video_room.shift(lag)))
            result[name_1+'_'+name_2+'_lag_'+str(lag)][mask] = np.nan
        for lead in range(1,11):            
            result[name_1+'_'+name_2+'_lead_'+str(lead)]=result[name_1+'_'+name_2].shift(periods=-lead)
            mask = ((result.final_dataset != result.final_dataset.shift(-lead)) |  (result.video_room != result.video_room.shift(-lead)))
            result[name_1+'_'+name_2+'_lead_'+str(lead)][mask] = np.nan
            
list_4_of_columns_to_lag = ['video_room','pir','probability_of_value_living','probability_of_value_kitchen','probability_of_value_bed2',
                           'probability_of_value_bath','probability_of_value_toilet','probability_of_value_bed1','probability_of_value_hall_haut',
                           'probability_of_value_stairs','probability_of_value_hall_bas','probability_of_value_study']
for name_1 in list_4_of_columns_to_lag:
    for lag in range(1,11):
        result[name_1+'_lag_'+str(lag)]=result[name_1].shift(periods=lag)
        mask = (result.final_dataset != result.final_dataset.shift(lag))
        result[name_1+'_lag_'+str(lag)][mask] = np.nan
    for lead in range(1,11):
        result[name_1+'_lead_'+str(lead)]=result[name_1].shift(periods=-lead)
        mask = (result.final_dataset != result.final_dataset.shift(-lead))
        result[name_1+'_lead_'+str(lead)][mask] = np.nan


list_5_of_columns_to_lag = [ 'video_room_centre_2d_x_length','video_room_centre_2d_y_length','room_angle_turn_x',
                           'video_room_bb_3d_brb_x_length','video_room_bb_3d_brb_y_length']
for name_1 in list_5_of_columns_to_lag:
    for lag in range(1,11):
        result[name_1+'_lag_'+str(lag)]=result[name_1].shift(periods=lag)
        mask = ((result.final_dataset != result.final_dataset.shift(lag)) |  (result.video_room != result.video_room.shift(lag)))
        result[name_1+'_lag_'+str(lag)][mask] = np.nan
    for lead in range(1,11):
        result[name_1+'_lead_'+str(lead)]=result[name_1].shift(periods=-lead)
        mask = ((result.final_dataset != result.final_dataset.shift(-lead)) |  (result.video_room != result.video_room.shift(-lead)))
        result[name_1+'_lead_'+str(lead)][mask] = np.nan


# We add features of lagdiff to the train dataset. We also add features of lagdiff of lagdiff to the train dataset
list_6_of_columns_to_lag = ['acceleration_x_median','acceleration_y_median','acceleration_z_median',
                            'video_room_centre_3d_x_mean','video_room_centre_3d_y_mean','video_room_centre_3d_z_mean',
                           'video_room_centre_2d_x_length','video_room_centre_2d_y_length','room_angle_turn_x',
                           'video_room_bb_3d_brb_x_length','video_room_bb_3d_brb_y_length']
for name_1 in list_6_of_columns_to_lag:
    lag = 1
    result[name_1+'_lagdiff']= result[name_1].diff(periods=lag)
    mask = ((result.final_dataset != result.final_dataset.shift(lag)) |  (result.video_room != result.video_room.shift(lag)))
    result[name_1+'_lagdiff'][mask] = np.nan
    lead = 1
    result[name_1+'_leaddiff']= result[name_1].diff(periods=-lead)
    mask = ((result.final_dataset != result.final_dataset.shift(-lead)) |  (result.video_room != result.video_room.shift(-lead)))
    result[name_1+'_leaddiff'][mask] = np.nan

list_7_of_columns_to_lag = ['acceleration_x_median_lagdiff','acceleration_y_median_lagdiff','acceleration_z_median_lagdiff',
                            'video_room_centre_3d_x_mean_lagdiff','video_room_centre_3d_y_mean_lagdiff','video_room_centre_3d_z_mean_lagdiff']
for name_1 in list_7_of_columns_to_lag:
    lag = 1
    result[name_1+'_lagdiff']=result[name_1].diff(periods=lag)
    mask = ((result.final_dataset != result.final_dataset.shift(lag+1)) |  (result.video_room != result.video_room.shift(lag+1)))
    result[name_1+'_lagdiff'][mask] = np.nan
    lead = 1
    result[name_1+'_leaddiff']=result[name_1].diff(periods=-lead)
    mask = ((result.final_dataset != result.final_dataset.shift(-lead-1)) |  (result.video_room != result.video_room.shift(-lead-1)))
    result[name_1+'_leaddiff'][mask] = np.nan

# We use formulas of the newly created features, in order to create even more features 
result['room_angle_turn_x_lagdiff_abs'] = (result['room_angle_turn_x_lagdiff']).apply(fabs)
result['room_angle_turn_x_leaddiff_abs'] = (result['room_angle_turn_x_leaddiff']).apply(fabs)
result['speed_room_centre_3d_horizontal'] = (result['video_room_centre_3d_x_mean_lagdiff'].pow(2) + result['video_room_centre_3d_z_mean_lagdiff'].pow(2)).apply(sqrt)
result['speed_room_centre_3d_vertical'] = (result['video_room_centre_3d_z_mean_lagdiff']).apply(fabs)
result['acceleration_room_centre_3d_horizontal'] = (result['video_room_centre_3d_x_mean_lagdiff_lagdiff'].pow(2) + result['video_room_centre_3d_z_mean_lagdiff_lagdiff'].pow(2)).apply(sqrt)
result['acceleration_room_centre_3d_vertical'] = (result['video_room_centre_3d_y_mean_lagdiff_lagdiff']).apply(fabs)

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/processed/columns_train_with_feature_engineering.csv')
result.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
