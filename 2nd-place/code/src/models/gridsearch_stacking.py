# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn as sk
from collections import defaultdict, Counter
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import xgboost as xgb
import os

# Recipe inputs
base_learners_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/Base_learners')

ml_dataset = pd.read_pickle(base_learners_path+"/dblend_train_batch1.pkl")
cv_df =  pd.read_pickle(base_learners_path+"/cv_df.p")
Y_dev = pd.read_pickle(base_learners_path+"/Y_dev.pkl") 
answer_for_crossval = pd.read_pickle(base_learners_path+"/answer_for_crossval.p") 
print len(answer_for_crossval)


ml_dataset_X = ml_dataset
ml_dataset_Y = Y_dev
                      
valid_vals = [6,10]

test = cv_df[cv_df["original_dataset"].isin(valid_vals)].index
train = cv_df[-cv_df["original_dataset"].isin(valid_vals)].index                      

X_train = ml_dataset_X.loc[train]
y_train = ml_dataset_Y.loc[train]
X_valid = ml_dataset_X.loc[test]
y_valid = ml_dataset_Y.loc[test]

cvprobs = answer_for_crossval.loc[test]

# def the brier score
def brier_score(targets, predicted, weights): 
    return np.power(targets - predicted, 2.0).dot(weights).mean()

class_weights = [1.35298455691, 1.38684574053, 1.59587388404, 1.35318713948, 0.347783666015
                 , 0.661081706198, 1.04723628621, 0.398865222651, 0.207586320237, 1.50578335208
                 , 0.110181365961, 1.07803284435, 1.36560417316, 1.17024113802, 1.1933637414
                 , 1.1803704493, 1.34414875433, 1.11683830693, 1.08083910312, 0.503152249073]                      

score = 100000    
n_estimators = 10000000
lr =  0.005
early_stopping_rounds = 40

####### the best result for the gridsearch is: 1092 3 5 0.2 1 0.178467129723 with lr = 0.005

for md in [3] : 
    for mcw in [5]: 
        for cby in [0.2] : 
            for sub in  [1] : 
            
                print 'doing : '
                print 'max depth', md
                print 'mcw', mcw
                print 'col sample', cby
                print 'subsample', sub
            
                clf = xgb.XGBClassifier( n_estimators= n_estimators
                                        , objective = 'multi:softprob'
                                        , learning_rate=lr
                                        , max_depth=md
                                        , min_child_weight  = mcw
                                        , subsample=sub
                                        , colsample_bytree=cby
                                )
                
                eval_set  = [(X_train,y_train), (X_valid,y_valid)]
                
                clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="mlogloss", early_stopping_rounds=early_stopping_rounds)
                
                preds = clf.predict_proba(X_valid)
                
                newscore = brier_score(cvprobs, preds, class_weights)
                
                print "nbtrees",clf.best_iteration
                
                if newscore <= score : 
                    score = newscore
                    bestmodel = clf
                    print 'mybestmodel'
                    print md, mcw, cby, sub, score
                    print "nb trees : ", clf.best_iteration
                    print "loss : ", clf.best_score

## End of gridsearch: we need to read the logs to read the optimal parameters to use in stacking: look for the last apparition of 'mybestmodel'

