# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from math import atan, sqrt, fabs
import os


######
# The goal of this section of code is to perform the feature engineering on the test set. 
# It consists of : 
# - adding features to the test dataset
# - normalizing features by user
# - multiplying by -1 the accelerometer data for certain users
# - adding features of lags and leads to the test dataset
# We  consider (although it might not be totally true) that all the test sets labeled by the same annotator, correpond to the same user: we use annotators_id as a user_id
######

input_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test_before_feature_engineering.csv')
test_df = pd.read_csv(filepath_or_buffer = input_path_1,encoding='utf-8')
input_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/annotators_id_test.csv')
annotators_id_df = pd.read_csv(filepath_or_buffer = input_path_2,encoding='utf-8')


# Add features to the train dataset, based on simple formulas
test_df['video_room_centre_2d_x_length'] = test_df['video_room_bb_2d_br_x_mean']-test_df['video_room_bb_2d_tl_x_mean']
test_df['video_room_centre_2d_y_length'] = test_df['video_room_bb_2d_br_y_mean']-test_df['video_room_bb_2d_tl_y_mean']
test_df['video_room_bb_3d_brb_x_length'] = (test_df['video_room_bb_3d_brb_x_mean']-test_df['video_room_bb_3d_flt_x_mean'])
test_df['video_room_bb_3d_brb_y_length'] = (test_df['video_room_bb_3d_flt_y_mean']-test_df['video_room_bb_3d_brb_y_mean'])
test_df['room_angle_turn_x'] = ((test_df['video_room_bb_3d_brb_x_mean']-test_df['video_room_centre_3d_x_mean'])/(test_df['video_room_bb_3d_brb_z_mean']-test_df['video_room_centre_3d_z_mean'])).apply(atan)

# We add a column 'annotators_id' to the test set: we then use it as a user id 
output_df = pd.DataFrame(columns=['annotators_id'],index = test_df.index)
for index in test_df.index:
    record_id = test_df.loc[index]['record_id']
    output_df.loc[index]['annotators_id'] = annotators_id_df.loc[annotators_id_df.record_id == record_id]['annotators'].iloc[0]
test_df = pd.concat([test_df, output_df], axis=1, join='inner')

# Normalize video_room features by user
video_cols = [col for col in test_df.columns if 'video_room_' in col]
for col in video_cols:
    test_df[col] = test_df.groupby("annotators_id")[col].apply(lambda x: (x - x.mean()) / x.std())

# Mulitply acceleration by -1 for users that seem to have worn their accelerometer upside down
acceleration_x_cols = [col for col in test_df.columns if 'acceleration_x' in col]
acceleration_y_cols = [col for col in test_df.columns if 'acceleration_y' in col]
for i in ['[3, 5]','[10, 5]','[]','[13, 5]']:
    for col in acceleration_x_cols:
        if not('acceleration_x_std' in col):
            test_df.loc[test_df.annotators_id == i, col] = test_df.loc[test_df.annotators_id == i, col].apply(lambda x: x*-1)
for i in ['[3, 5]','[10, 5]','[]','[13, 5]']:
    for col in acceleration_y_cols:
        if not('acceleration_y_std' in col):
            test_df.loc[test_df.annotators_id == i, col] = test_df.loc[test_df.annotators_id == i, col].apply(lambda x: x*-1)

# We add features of lag/lead to train dataset: lags and leads up to 10 seconds
list_1_of_columns_to_lag = ['acceleration_x','acceleration_y','acceleration_z','rssi_Kitchen_AP','rssi_Lounge_AP',
                            'rssi_Upstairs_AP','rssi_Study_AP']
list_2_of_columns_to_lag = ['mean','std','min','median','max']
for name_1 in list_1_of_columns_to_lag:
    for name_2 in list_2_of_columns_to_lag:
        for lag in range(1,11):
            test_df[name_1+'_'+name_2+'_lag_'+str(lag)]=test_df[name_1+'_'+name_2].shift(periods=lag)
            mask = (test_df.record_id != test_df.record_id.shift(lag))
            test_df[name_1+'_'+name_2+'_lag_'+str(lag)][mask] = np.nan
        for lead in range(1,11):
            test_df[name_1+'_'+name_2+'_lead_'+str(lead)]=test_df[name_1+'_'+name_2].shift(periods=-lead)
            mask = (test_df.record_id != test_df.record_id.shift(-lead))
            test_df[name_1+'_'+name_2+'_lead_'+str(lead)][mask] = np.nan

list_3_of_columns_to_lag = ['video_room_centre_2d_x','video_room_centre_2d_y',
                            'video_room_bb_2d_br_x','video_room_bb_2d_br_y','video_room_bb_2d_tl_x','video_room_bb_2d_tl_y',
                    'video_room_centre_3d_x','video_room_centre_3d_y','video_room_centre_3d_z','video_room_bb_3d_brb_x',
                            'video_room_bb_3d_brb_y','video_room_bb_3d_brb_z','video_room_bb_3d_flt_x','video_room_bb_3d_flt_y',
                            'video_room_bb_3d_flt_z']
for name_1 in list_3_of_columns_to_lag:
    for name_2 in list_2_of_columns_to_lag:
        for lag in range(1,11):
            test_df[name_1+'_'+name_2+'_lag_'+str(lag)]=test_df[name_1+'_'+name_2].shift(periods=lag)
            mask = ((test_df.record_id != test_df.record_id.shift(lag)) |  (test_df.video_room != test_df.video_room.shift(lag)))
            test_df[name_1+'_'+name_2+'_lag_'+str(lag)][mask] = np.nan
        for lead in range(1,11):
            test_df[name_1+'_'+name_2+'_lead_'+str(lead)]=test_df[name_1+'_'+name_2].shift(periods=-lead)
            mask = ((test_df.record_id != test_df.record_id.shift(-lead)) |  (test_df.video_room != test_df.video_room.shift(-lead)))
            test_df[name_1+'_'+name_2+'_lead_'+str(lead)][mask] = np.nan


list_4_of_columns_to_lag = ['video_room','pir','probability_of_value_living','probability_of_value_kitchen','probability_of_value_bed2',
                           'probability_of_value_bath','probability_of_value_toilet','probability_of_value_bed1','probability_of_value_hall_haut',
                           'probability_of_value_stairs','probability_of_value_hall_bas','probability_of_value_study']
for name_1 in list_4_of_columns_to_lag:
    for lag in range(1,11):
        test_df[name_1+'_lag_'+str(lag)]=test_df[name_1].shift(periods=lag)
        mask = (test_df.record_id != test_df.record_id.shift(lag))
        test_df[name_1+'_lag_'+str(lag)][mask] = np.nan
for name_1 in list_4_of_columns_to_lag:
    for lead in range(1,11):
        test_df[name_1+'_lead_'+str(lead)]=test_df[name_1].shift(periods=-lead)
        mask = (test_df.record_id != test_df.record_id.shift(-lead))
        test_df[name_1+'_lead_'+str(lead)][mask] = np.nan
        
list_5_of_columns_to_lag = [ 'video_room_centre_2d_x_length','video_room_centre_2d_y_length','room_angle_turn_x',
                           'video_room_bb_3d_brb_x_length','video_room_bb_3d_brb_y_length']
for name_1 in list_5_of_columns_to_lag:
    for lag in range(1,11):
        test_df[name_1+'_lag_'+str(lag)]=test_df[name_1].shift(periods=lag)
        mask = ((test_df.record_id != test_df.record_id.shift(lag)) |  (test_df.video_room != test_df.video_room.shift(lag)))
        test_df[name_1+'_lag_'+str(lag)][mask] = np.nan
    for lead in range(1,11):
        test_df[name_1+'_lead_'+str(lead)]=test_df[name_1].shift(periods=-lead)
        mask = ((test_df.record_id != test_df.record_id.shift(-lead)) |  (test_df.video_room != test_df.video_room.shift(-lead)))
        test_df[name_1+'_lead_'+str(lead)][mask] = np.nan


# We add features of lagdiff to the train dataset. We also add features of lagdiff of lagdiff to the train dataset
list_6_of_columns_to_lag = ['acceleration_x_median','acceleration_y_median','acceleration_z_median',
                            'video_room_centre_3d_x_mean','video_room_centre_3d_y_mean','video_room_centre_3d_z_mean',
                           'video_room_centre_2d_x_length','video_room_centre_2d_y_length','room_angle_turn_x',
                           'video_room_bb_3d_brb_x_length','video_room_bb_3d_brb_y_length']
for name_1 in list_6_of_columns_to_lag:
    lag = 1
    test_df[name_1+'_lagdiff']= test_df[name_1].diff(periods=lag)
    mask = ((test_df.record_id != test_df.record_id.shift(lag)) |  (test_df.video_room != test_df.video_room.shift(lag)))
    test_df[name_1+'_lagdiff'][mask] = np.nan
    lead = 1
    test_df[name_1+'_leaddiff']= test_df[name_1].diff(periods=-lead)
    mask = ((test_df.record_id != test_df.record_id.shift(-lead)) |  (test_df.video_room != test_df.video_room.shift(-lead)))
    test_df[name_1+'_leaddiff'][mask] = np.nan

list_7_of_columns_to_lag = ['acceleration_x_median_lagdiff','acceleration_y_median_lagdiff','acceleration_z_median_lagdiff',
                            'video_room_centre_3d_x_mean_lagdiff','video_room_centre_3d_y_mean_lagdiff','video_room_centre_3d_z_mean_lagdiff']
for name_1 in list_7_of_columns_to_lag:
    lag = 1
    test_df[name_1+'_lagdiff']=test_df[name_1].diff(periods=lag)
    mask = ((test_df.record_id != test_df.record_id.shift(lag+1)) |  (test_df.video_room != test_df.video_room.shift(lag+1)))
    test_df[name_1+'_lagdiff'][mask] = np.nan
    lead = 1
    test_df[name_1+'_leaddiff']=test_df[name_1].diff(periods=-lead)
    mask = ((test_df.record_id != test_df.record_id.shift(-lead-1)) |  (test_df.video_room != test_df.video_room.shift(-lead-1)))
    test_df[name_1+'_leaddiff'][mask] = np.nan

# We use formulas of the newly created features, in order to create even more features 
test_df['room_angle_turn_x_lagdiff_abs'] = (test_df['room_angle_turn_x_lagdiff']).apply(fabs)
test_df['room_angle_turn_x_leaddiff_abs'] = (test_df['room_angle_turn_x_leaddiff']).apply(fabs)
test_df['speed_room_centre_3d_horizontal'] = (test_df['video_room_centre_3d_x_mean_lagdiff'].pow(2) + test_df['video_room_centre_3d_z_mean_lagdiff'].pow(2)).apply(sqrt)
test_df['speed_room_centre_3d_vertical'] = (test_df['video_room_centre_3d_z_mean_lagdiff']).apply(fabs)
test_df['acceleration_room_centre_3d_horizontal'] = (test_df['video_room_centre_3d_x_mean_lagdiff_lagdiff'].pow(2) + test_df['video_room_centre_3d_z_mean_lagdiff_lagdiff'].pow(2)).apply(sqrt)
test_df['acceleration_room_centre_3d_vertical'] = (test_df['video_room_centre_3d_y_mean_lagdiff_lagdiff']).apply(fabs)

# Add features indicating how many seconds have passed since the last change of value in video_room/original_dataset
# Indeed, we will add lags/leads to our train set: so the first lines will have many Nan values in their lags: it's interesting to add a feature that may explain this behaviour
output_df = pd.DataFrame(columns=['time_since_change_of_record_id','time_since_change_of_video_room'],index = test_df.index)
for index in test_df.index:
    record_id_has_chged_over_10_seconds = False
    for j in range(1,11):
        if (test_df.record_id[index] != test_df.record_id.shift(j)[index]):
            record_id_has_chged_over_10_seconds = True
            break
    if record_id_has_chged_over_10_seconds:
        output_df.loc[index]['time_since_change_of_record_id'] = j
    else: 
        output_df.loc[index]['time_since_change_of_record_id'] = 11

    video_room_has_chged_over_10_seconds = False
    if ( pd.isnull(test_df.video_room.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 0
    elif (pd.isnull(test_df.video_room.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    elif not(test_df.video_room[index] == test_df.video_room.shift(1)[index]):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    elif not((test_df.record_id[index] == test_df.record_id.shift(1)[index])):
        output_df.loc[index]['time_since_change_of_video_room'] = 1
    else: 
        k=0
        while ( (test_df.video_room[index] == test_df.video_room.shift(k)[index] ) and (test_df.record_id[index] == test_df.record_id.shift(k)[index])):
            output_df.loc[index]['time_since_change_of_video_room'] = k
            k = k+1
result = pd.concat([test_df, output_df], axis=1, join='inner')
           
# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/processed/columns_test_with_feature_engineering.csv')
result.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
