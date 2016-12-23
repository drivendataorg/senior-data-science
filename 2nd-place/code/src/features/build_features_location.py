# -*- coding: utf-8 -*-
import sklearn as sk
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os




######
# The variable 'location' is available on train dataset, but not on the test dataset.
# The goal of this script is to add the variable 'location' on the test dataset, in a stacking-fashion
# On the train dataset, we use 90% of the data to predict the value of 'location'on the 10%. 
# We do this 10 times, to predict the value of 'location' on the whole train dataset.
# Then, we use the predicted values of 'location' on the train set, in order to predict the value of 'location' on the test set
######
input_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets_and_location_splitted.csv')
train_df = pd.read_csv(filepath_or_buffer = input_path_1,encoding='utf-8')
input_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test_prepared.csv')
test_df = pd.read_csv(filepath_or_buffer = input_path_2,encoding='utf-8')

copy_train_df = train_df.copy()
copy_test_df = test_df.copy()

######
# First, we work on the train dataset: we will use 90% of the train data to predict 'location' on 10% of the train data
######

# Keep useful variables of the train data
original_dataset_df = train_df['original_dataset']
start_df = train_df['start']
end_df = train_df['end']

# Limit the train data to features that will be pertinent to predict 'location'
columns_to_keep = [u'rssi_Kitchen_AP_median_bool', u'video_room_centre_3d_x_max', u'video_room_bb_3d_flt_z_min', u'acceleration_x_median', u'video_room_bb_3d_flt_z_max', u'video_room_bb_2d_br_x_min', u'video_room_bb_3d_flt_x_std', u'location', u'video_room_centre_2d_y_mean', u'video_room_bb_2d_br_x_std', u'video_room_bb_2d_br_x_max', u'video_room_bb_3d_flt_x_max', u'video_room_bb_2d_br_y_median', u'video_room_bb_2d_br_x_median', u'rssi_Kitchen_AP_max', u'video_room_bb_3d_flt_x_min', u'video_room_bb_2d_tl_y_median', u'rssi_Upstairs_AP_median_bool', u'video_room_centre_3d_x_std', u'acceleration_z_std', u'video_room_centre_2d_x_min', u'video_room_centre_2d_y_median', u'acceleration_z_max', u'rssi_Kitchen_AP_median', u'video_room_centre_2d_x_max', u'rssi_Study_AP_median', u'rssi_Lounge_AP_median_bool', u'rssi_Lounge_AP_max', u'acceleration_x_min', u'rssi_Kitchen_AP_mean', u'video_room_centre_2d_x_median', u'video_room_bb_3d_flt_x_median', u'video_room_bb_3d_flt_z_std', u'video_room_bb_2d_br_y_min', u'video_room_bb_3d_flt_y_std', u'acceleration_z_median', u'video_room_bb_2d_br_y_max', u'rssi_Lounge_AP_min', u'video_room_centre_3d_z_min', u'video_room_bb_2d_tl_x_max', u'video_room_bb_2d_tl_y_max', u'video_room_bb_2d_br_y_std', u'video_room_bb_2d_tl_y_mean', u'video_room_bb_2d_tl_x_min', u'video_room_bb_2d_br_y_mean', u'rssi_Study_AP_mean', u'video_room_centre_3d_z_max', u'video_room_centre_3d_z_std', u'acceleration_z_min', u'video_room_bb_3d_flt_z_mean', u'rssi_Upstairs_AP_mean', u'video_room_centre_3d_x_min', u'video_room_centre_3d_y_mean', u'rssi_Lounge_AP_std', u'video_room_bb_3d_brb_z_mean', u'acceleration_y_std', u'rssi_Study_AP_median_bool', u'video_room_bb_3d_flt_y_max', u'video_room_bb_3d_brb_z_std', u'video_room_centre_3d_z_median', u'rssi_Upstairs_AP_min', u'video_room_bb_3d_brb_x_std', u'video_room_bb_3d_brb_x_max', u'video_room_bb_3d_flt_x_mean', u'video_room_bb_3d_brb_z_median', u'video_room_bb_2d_tl_x_median', u'rssi_Study_AP_std', u'video_room_centre_3d_z_mean', u'rssi_Lounge_AP_median', u'video_room_bb_3d_brb_x_mean', u'rssi_Kitchen_AP_std', u'rssi_Upstairs_AP_median', u'acceleration_x_mean', u'video_room_bb_3d_brb_y_mean', u'video_room_centre_3d_x_median', u'acceleration_x_std', u'video_room_centre_2d_y_std', u'video_room_centre_2d_x_std', u'rssi_Study_AP_max', u'video_room_bb_3d_flt_y_median', u'rssi_Study_AP_min', u'video_room_bb_3d_flt_z_median', u'video_room_bb_3d_flt_y_mean', u'rssi_Kitchen_AP_min', u'video_room_bb_3d_brb_y_min', u'acceleration_z_mean', u'video_room_bb_2d_tl_y_std', u'video_room_bb_2d_tl_y_min', u'acceleration_y_max', u'video_room_bb_2d_br_x_mean', u'acceleration_y_median', u'pir', u'video_room_centre_3d_y_median', u'video_room_bb_3d_brb_z_max', u'video_room_bb_3d_brb_z_min', u'video_room_bb_3d_brb_y_max', u'acceleration_y_min', u'video_room_centre_3d_y_std', u'video_room_centre_2d_x_mean', u'rssi_Upstairs_AP_std', u'video_room_bb_2d_tl_x_std', u'acceleration_x_max', u'acceleration_y_mean', u'video_room_bb_2d_tl_x_mean', u'rssi_Lounge_AP_mean', u'video_room_bb_3d_brb_x_median', u'video_room_bb_3d_brb_x_min', u'video_room_centre_2d_y_max', u'rssi_Upstairs_AP_max', u'video_room_centre_2d_y_min', u'video_room_bb_3d_brb_y_median', u'video_room_bb_3d_brb_y_std', u'video_room_centre_3d_x_mean', u'video_room_centre_3d_y_min', u'video_room_centre_3d_y_max', u'video_room_bb_3d_flt_y_min', u'video_room']
train_df = train_df[columns_to_keep]

# Map the target variable: 'location'
target_map = {u'living': 0, u'toilet': 4, u'hall_bas': 8, u'bath': 3, u'bed2': 2, u'bed1': 5, u'hall_haut': 6, u'stairs': 7, u'study': 9, u'kitchen': 1}
train_df['__target__'] = train_df['location'].map(str).map(target_map)
del train_df['location']

cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(label_id), label) for (label, label_id) in target_map.iteritems()])
    ]

# Dummifiy certain variables of the train set
LIMIT_DUMMIES = 100
categorical_to_dummy_encode = [u'rssi_Kitchen_AP_median_bool', u'rssi_Upstairs_AP_median_bool', u'rssi_Lounge_AP_median_bool', u'rssi_Study_AP_median_bool', u'pir', u'video_room']

def select_dummy_values(train_df, features):
    dummy_values = {}
    for feature in categorical_to_dummy_encode:
        values = [
            value
            for (value, _) in Counter(train_df[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values
DUMMY_VALUES = select_dummy_values(train_df, categorical_to_dummy_encode)
def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
dummy_encode_dataframe(train_df)


# Split the train data between the features (X) and target variable (Y, 'location')
train_df_X = train_df.drop('__target__', axis=1)
train_df_Y = train_df['__target__']

# Fill missing values of the train
train_df_X = train_df_X.fillna(-99999.0)


# In order to predict 'location' on 100% of train set, we need 10 models. We use random forests, with reasonable (but arbitrary) parameters
from sklearn.ensemble import RandomForestClassifier

list_of_classifiers = [RandomForestClassifier(n_estimators=200,
    n_jobs=2,
    random_state=1337,
    max_depth=20,
    min_samples_leaf=1,
    verbose=2) ] * 10

output_df = pd.DataFrame(columns=['original_dataset','start','end']+range(1,11))


for i, clf in enumerate(list_of_classifiers,start = 1): 
    # for each classifier, we will learn 'location' on 90% of train set, and predict 'location' on 10% of train set
    train_df_X_train = train_df_X.loc[(~train_df['__target__'].isnull()) & (original_dataset_df != i)]
    train_df_Y_train = train_df_Y.loc[(~train_df['__target__'].isnull()) & (original_dataset_df != i)]
    train_df_X_test = train_df_X.loc[original_dataset_df == i]
    train_df_Y_test = train_df_Y.loc[original_dataset_df == i]
    # learn 'location' on 90% of train set
    clf = clf.fit(train_df_X_train, train_df_Y_train)
    print((train_df_X_train).shape)
    print((train_df_Y_train).shape)
    # predict 'location' on 10% of train set
    _probas = clf.predict_proba(train_df_X_test)
    print(train_df_X_test.shape)
    print((_probas).shape)
    probabilities = pd.DataFrame(data=_probas, index=train_df_X_test.index,columns=cols)
    # add variables to the predictions that will help identitify the prediction (variables original_dataset, start and end)
    end_df_temp = pd.DataFrame(data=end_df, index=train_df_X_test.index)
    start_df_temp = pd.DataFrame(data=start_df, index=train_df_X_test.index)
    original_dataset_df_temp = pd.DataFrame(data=original_dataset_df, index=train_df_X_test.index)

    results_test = original_dataset_df_temp.join(start_df_temp, how='left')
    results_test = results_test.join(end_df_temp, how='left')
    results_test = results_test.join(probabilities, how='left')
    if i==1:
        output_df = results_test
    else:
        output_df = output_df.append(results_test,ignore_index=False)
    
    
    
######
# Second, we work on the test dataset: we will predict 'location' on test set thanks to the 10 models that have been created and fitted right above
######   

# Keep useful variables of test data
record_id_df_test = test_df['record_id']
start_df_test = test_df['start']
end_df_test = test_df['end']

# Limit the test data to features that will be pertinent to predict 'location'
columns_to_keep.remove('location')
test_df = test_df[columns_to_keep]

# Dummifiy certain variables of the test set
dummy_encode_dataframe(test_df)

test_df = test_df.fillna(-99999.0)

# Prediction 'location' on the test set, thanks to the 10 previous models
list_of_predictions = [pd.DataFrame()] * 10

for i, clf in enumerate(list_of_classifiers,start = 1):  
    # predict 'location' on test set
    _probas_test = clf.predict_proba(test_df)
    probabilities_test = pd.DataFrame(data=_probas_test, index=test_df.index,columns=cols)
    # add variables to the predictions, in order to identitify the prediction (variables record_id, start and end)
    end_df_test_temp = pd.DataFrame(data=end_df_test, index=test_df.index)
    start_df_test_temp = pd.DataFrame(data=start_df_test, index=test_df.index)
    record_id_df_test_temp = pd.DataFrame(data=record_id_df_test, index=test_df.index)

    results_test = record_id_df_test_temp.join(start_df_test_temp, how='left')
    results_test = results_test.join(end_df_test_temp, how='left')
    results_test = results_test.join(probabilities_test, how='left')
    list_of_predictions[i-1]=results_test
    
# Each of the 10 classifiers have predicted 'location' on the test dataset: we average these 10 predictions
for i,df in enumerate(list_of_predictions,start=1):
    if i == 1:
        prediction_df = df
    else:
        prediction_df = (prediction_df + df)
prediction_df = prediction_df/10

# We add the predictions of 'location' to the train and test sets
output_train_df = pd.merge(copy_train_df,output_df,on=['original_dataset','start','end'])
output_test_df = pd.merge(copy_test_df,prediction_df,on=['record_id','start','end'])


# Recipe outputs
output_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_before_feature_engineering.csv')
output_train_df.to_csv(path_or_buf = output_path_1 , header = True, index = False, encoding='utf-8')
output_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test_before_feature_engineering.csv')
output_test_df.to_csv(path_or_buf = output_path_2 , header = True, index = False, encoding='utf-8')
