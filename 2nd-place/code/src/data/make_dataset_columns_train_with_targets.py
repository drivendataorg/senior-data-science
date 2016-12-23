# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os


######
# The goal of this section of script is to merge the train and target datasets
######
input_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_prepared.csv')
columns_train_prepared_df = pd.read_csv(filepath_or_buffer = input_path_1,encoding='utf-8')
input_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/targets_train.csv')
targets_train_df = pd.read_csv(filepath_or_buffer = input_path_2,encoding='utf-8')

columns_train_with_targets_df = pd.merge(columns_train_prepared_df, targets_train_df, on=['original_dataset','start','end'], how='left')

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets.csv')
columns_train_with_targets_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
