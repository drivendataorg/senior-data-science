# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os


######
# The goal of this script is to simplify the "pir" columns on the test dataset, and to create a sparse structure for video_room data
######
input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test.csv')
columns_test_df = pd.read_csv(filepath_or_buffer = input_path,encoding='utf-8')


# Delete redundant pir columns
pir_cols_to_delete = [col for col in columns_test_df.columns if (('pir_' in col) and (not('_mean') in col))]
columns_test_df.drop(pir_cols_to_delete, axis=1, inplace=True)

# Rename pir columns
dict_rename = {}
pir_cols_to_rename = [col for col in columns_test_df.columns if (('pir_' in col) and (('_mean') in col))]
for i in range(len(pir_cols_to_rename)):
    dict_rename[pir_cols_to_rename[i]]=pir_cols_to_rename[i][:-5]
    dict_rename[pir_cols_to_rename[i]]=pir_cols_to_rename[i][:-5]
columns_test_df.rename(columns=dict_rename, inplace=True)

# Convert the pir variables into one categorial variable
columns_test_df['pir'] = columns_test_df[['pir_bath','pir_bed1','pir_bed2','pir_hall','pir_kitchen','pir_living',
                                                                                          'pir_stairs','pir_study','pir_toilet']
                                                                                        ].idxmax (axis=1).str.split(pat = 'pir_',expand=True)[1]    
columns_test_df.drop(['pir_bath','pir_bed1','pir_bed2','pir_hall','pir_kitchen','pir_living','pir_stairs','pir_study','pir_toilet'], axis=1, inplace=True)




# Create a sparse structure for video_room data: one column will indicate which video is activated, and the other columns will indicate its mean/median/max/min/etc
list_of_columns_1 = ['centre_2d_x','centre_2d_y','bb_2d_br_x','bb_2d_br_y','bb_2d_tl_x','bb_2d_tl_y',
                    'centre_3d_x','centre_3d_y','centre_3d_z','bb_3d_brb_x','bb_3d_brb_y','bb_3d_brb_z',
                    'bb_3d_flt_x','bb_3d_flt_y','bb_3d_flt_z']
list_of_columns_2 = ['mean','std','min','median','max']

columns_test_df['video_room'] = columns_test_df[['video_living_room_centre_2d_x_mean',
    'video_kitchen_centre_2d_x_mean','video_hallway_centre_2d_x_mean']].idxmax (axis=1).str.split(pat = '_centre',expand=True)[0]    

for name_1 in list_of_columns_1:
    for name_2 in list_of_columns_2:
        columns_test_df['video_room_'+name_1+'_'+name_2] = columns_test_df[['video_living_room_'+name_1+'_'+name_2,'video_kitchen_'+name_1+'_'+name_2,
             'video_hallway_'+name_1+'_'+name_2]].max(axis=1)
        columns_test_df.drop(['video_living_room_'+name_1+'_'+name_2,'video_kitchen_'+name_1+'_'+name_2,
            'video_hallway_'+name_1+'_'+name_2], axis=1, inplace=True)

        

# We add variables to indicate whether each rssi sensor is activated
columns_test_df['rssi_Kitchen_AP_median_bool'] = ~(columns_test_df['rssi_Kitchen_AP_median']).isnull()
columns_test_df['rssi_Lounge_AP_median_bool'] = ~(columns_test_df['rssi_Lounge_AP_median']).isnull()
columns_test_df['rssi_Upstairs_AP_median_bool'] = ~(columns_test_df['rssi_Upstairs_AP_median']).isnull()
columns_test_df['rssi_Study_AP_median_bool'] = ~(columns_test_df['rssi_Study_AP_median']).isnull()
        

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test_prepared.csv')
columns_test_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
