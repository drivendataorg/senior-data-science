# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os


######
# The goal of this section of code is to merge the train and location datasets
######


# Recipe inputs
input_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets.csv')
columns_train_with_targets_df = pd.read_csv(filepath_or_buffer = input_path_1,encoding='utf-8')
input_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/location_train.csv')
location_train_df = pd.read_csv(filepath_or_buffer = input_path_2,encoding='utf-8')

columns_train_with_targets_and_location_df = pd.merge(columns_train_with_targets_df, location_train_df, on=['original_dataset','start','end'], how='left')

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets_and_location.csv')
columns_train_with_targets_and_location_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
