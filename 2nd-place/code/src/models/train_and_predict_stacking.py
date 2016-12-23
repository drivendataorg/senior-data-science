# -*- coding: utf-8 -*-
import numpy as np,pandas as pd, sklearn as sk, pickle as pkl
from collections import defaultdict, Counter
from sklearn import cross_validation, metrics
import xgboost as xgb
import os

#########
# Now that we have gridsearched our stacking model, we can train the stacking with these optimal parameters and make our stacking predictions.
# The goal of this script is to generate stacking predictions on the test set. 
# The stacking of 10 base learners is made thanks to an xgboost: it is the xgboost_level2. We have found its optimal parameters thanks to xgboost_level2_gridsearch
#########

# Recipe inputs
base_learners_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/Base_learners')

ml_dataset = pd.read_pickle(base_learners_path+"/dblend_train_batch1.pkl")
test = pd.read_pickle(base_learners_path+"/dblend_test_batch1.pkl")
cv_df =  pd.read_pickle(base_learners_path+"/cv_df.p")
Y_dev = pd.read_pickle(base_learners_path+"/Y_dev.pkl") 


input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/stacked_scored.csv')
stacked_scored = pd.read_csv(filepath_or_buffer = input_path,encoding='utf-8')



# keep some info                   
start_df = stacked_scored['start'] 
end_df = stacked_scored['end'] 
record_id_df = stacked_scored['record_id']  
                
columns_list=['a_ascend','a_descend','a_jump','a_loadwalk','a_walk','p_bent','p_kneel','p_lie','p_sit','p_squat','p_stand','t_bend',
              't_kneel_stand','t_lie_sit','t_sit_lie','t_sit_stand','t_stand_kneel','t_stand_sit','t_straighten','t_turn']
                                 

ml_dataset_X = ml_dataset #.drop('__target__', axis=1)
ml_dataset_Y = Y_dev #ml_dataset['__target__']
                      



X_train = ml_dataset_X
y_train = ml_dataset_Y
X_test = test


print "col test", X_test.columns
                      
# def the brier score
def brier_score(targets, predicted, weights): 
    return np.power(targets - predicted, 2.0).dot(weights).mean()

class_weights = [1.35298455691, 1.38684574053, 1.59587388404, 1.35318713948, 0.347783666015
                , 0.661081706198, 1.04723628621, 0.398865222651, 0.207586320237, 1.50578335208
                , 0.110181365961, 1.07803284435, 1.36560417316, 1.17024113802, 1.1933637414
                , 1.1803704493, 1.34414875433, 1.11683830693, 1.08083910312, 0.503152249073]                      

"""
n_estimators = 469 
lr =  0.01
md = 3
mcw = 5
cby = 0.7                      
sub = 1
"""

####### parameters found thanks to gridsearch in xgboost_level2 
n_estimators = 1092 
lr =  0.005
md = 3
mcw = 5
cby = 0.2                     
sub = 1
                    
clf = xgb.XGBClassifier( n_estimators=n_estimators
                       , objective = 'multi:softprob'
                       , learning_rate=lr
                       , max_depth=md
                       , min_child_weight  = mcw
                       , subsample=sub
                       , colsample_bytree=cby
               )
col = list(X_train.columns)
                
clf.fit(X_train[col], y_train )

                      
_probas = clf.predict_proba(X_test)
                      
probabilities = pd.DataFrame(data=_probas, index=test.index, columns=columns_list)
end_df = pd.DataFrame(data=end_df, index=test.index)
start_df = pd.DataFrame(data=start_df, index=test.index)
record_id_df = pd.DataFrame(data=record_id_df, index=test.index)

# Build scored dataset
#results_test = test_X.join(predictions, how='left')
results_test = record_id_df.join(start_df, how='left')
results_test = results_test.join(end_df, how='left')
results_test = results_test.join(probabilities, how='left')

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/prediction_drivendata_winning_solution.csv')
results_test.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
