# Senior Data Science: Safe Aging with SPHERE - 2nd Place

# Entrant Background and Submission Overview

### Mini-bio
Graduate Student in Management Science & Engineering.

### High Level Summary of Submission
4 steps in our approach.

###### Step 1: Change the structure of the train set to make it look like the test set

The raw train and test sets were inherently different.

The train set was generated with 10 elderly individuals. The test set was also generated with 10 elderly individuals – but not the same individuals as those involved in the train set.

And here's the crucial difference between train and test sets. In the train set, each of the 10 individuals has been recorded for 5 hours. So the train set consists of 10 continuous sequences of 5 hours of monitoring. But in the test set, each of the 10 individuals has been recorded 100 times, for 10 seconds every time. So the test set consists of 1000+ continuous sequences of 10 seconds of monitoring!

The first step was to change the structure of the train set, to make it look like the test set. We randomly split the train sequences of 5 hours into 1000+ small sequences of approximately 10 seconds.

It is possible to generate several training sets this way – by using several random seeds for the random split - and follow a bagging approach: create one model per training set and average their prediction. This approach showed very good results in cross validation, but it was not part of the final model.

###### Step 2: Feature engineering
Feature engineering aims at converting raw, hard-to-interpret variables into meaningful ones. This step is key in many machine learning problems since it can determine the quality of predictions.

First, we computed some basic features: speeds, accelerations, derivative of accelerations, rotations... They should intuitively help us predict the activity performed by the individual.

Second, we added lags on most variables. How? By adding columns that inform on the value of each variable 1 second ago, 2 seconds ago... up to 10 seconds ago.

Here, we realized how crucial it was to make train and test sets look alike: if we had not done so, then the train set would consist of 5-hour continuous sequences – so only the first 10 lines would have missing lag values; whereas the test would consist of 10-second continuous sequences – almost all the lines would have many missing lag values. If we had not homogenized the structure of train and test sets, then the lag variables would have different distributions on train and test sets - we therefore avoided a covariate shift.

Since adding lags worked well... we decided to add reverse lags. Reverse lags – or leads – turned out to add a lot of information resulting in significant cross validation score improvements.

I also suspect that some people had worn their accelerometers upside down. So, for these people, I multiplied the accelerometer's raw data by -1.

###### Step 3: Enriching the data with stack transferring
A room variable indicates the room where the individual is located. Intuitively, this variable should be very useful to predict the activity of the person: for instance, when someone is in the toilets, it should be very improbable for him to be jumping or lying down!

Unfortunately, this room variable is available in the train set, but it is missing in the test set. We came up with a clever technique – that we call stack transferring – to deal with this.

* Step 1: On the test set, predict the room variable

    * You can add the room variable on the test set, by predicting it. You just need to build a model that predicts the room variable. Make it learn on the train set – where we have the exact values of the room-variable – and apply it on the test set – where the room-variable is missing. Now, you have the room variable on the test set.


* Step 2: On the train set, update the room variable: replace its exact values by its out-of-fold predictions

    * Here's the stack transferring trick.  On the train set, you should replace the exact values of the room variable by out-of-folds predictions of the room variable!  In other words, use 90% of the train set to predict the room variable on the 10% remaining train set. By doing so 10 times, you can predict the room variable on all the train set. You have just generated out-of- folds predictions of the room variable on the train set!  Now, you can update the room variable on the train set: drop the exact values of the room variable and keep its out-of-fold predictions.


* Step 3: Use the out-of-folds predictions of the room variable to predict the activity variable

    * The updated room variable should help you predict the activity performed by an individual. Simply add this updated variable to your model. It should improve your results!

    Note 1: In our case, the individuals of train and test sets were asked to perform the same list of actions in the same order. Therefore, the room variable had the same distribution on train and test sets. This is a necessary condition if you want stack transferring to perform well.

    Note 2: In order to generate out-of-fold predictions on the train set, we split our train data into 10 folds. Each fold corresponds to the data generated by one user, in order to avoid any leakage.


###### Step 4: Fine-tune individual models to enter top 5, stack to take the lead
It's in general a good idea to start with a simple model that does not need much tuning – like random forests – while you are doing feature engineering. They are easy to implement and able to handle large amounts of variables, so they give you valuable feedback on the quality of your work. Feature engineering diminished our random forest's error-rate from 22% to 16.4%, ranking us 15th of the competition.

When performance seems to reach a plateau even when you are adding new features, try other models that require more tuning. We then went on for the machine learning blockbuster – XGBoost. Once fine-tuned, it performed very well, decreasing our error rate to 14.6% and ranking us top 5!

Finally, we decided to give the stacking technique a try. This ensemble learning technique combines the predictions of diverse individual models and leverages their strengths. Stacking 10 individual models – including linear regressions, Naive Bayes classifiers, random forests, extra-trees and XGBoost models – turned out to be very performant: it reduced our error rate to 12.9% and ranked us number 1 at that point!

### Omitted Work

###### Customize XGBoost to directly optimize the brier score
A promising idea was to customize the XGBoost classifier, so that it optimizes predictions for the competition's metric – the Brier score. Doing so turned out to be trickier than expected.

Indeed, XGBoost provides a Python API to customize loss functions for regression and binary classification, but unfortunately not for multi-class classification or multi-dimensional regression. After a bit of math, we managed to implement the gradient and an approximation of the hessian functions of the Brier score directly in the C++ XGBoost source code – it compiled and worked well!

Unfortunately, our customized XGBoost took way too long to converge – because of the 10 fold replication of the lines of our data to approximate the soft classes and the diagonal approximation of the hessian – and we had to give it up.

###### Post processing
Imagine that someone is lying on a bed at a given moment. You might agree that it is rather probable for him to still be lying the next second, but rather improbable for him to be jumping the next second, right? In other words, each transition, from one activity to another, has a certain probability, which can be high or low.

This mathematical property is known as the Markov chain property. A great way to take advantage of this underlying structure is to implement Hidden Markov Models. Yet, given the deadline, we rather opted for a post- processing that smooths predictions. We averaged each prediction with that of the previous and following seconds.

We tried to implement post-processing a couple of hours before the end of the competition. We struggled with the indexes of the different datasets and eventually managed to implement the post- processing at 11:57 pm. Post-processing gave tremendous cross-validation results, with an error rate around 11%! But uploading this result took 3 minutes: it was 12:00 pm, and the competition had come to an end without taking into account our promising post-processing.

###### Change the structure of the test set to make it look like the train set
I said it was crucial for train and test sets to have the same structure. In order to do so, we changed the train set to make it look like the test set. Well, we might as well have changed the structure of the test set to make it look like the train set! In other words, instead of splitting the 5-hour train sequences into 10-second sequences, we tried to reorder the 10-second test sequences to reconstitute 5-hour sequences. If we had managed to do so, we could have detected long-term patterns in the train and test data, which would have improved our activity-recognition!

So, how can we reorder more than 1000 10-second sequences to reconstitute 10 5-hour sequences? By finding the order among the 10-second sequences that maximizes a « continuous » function. Which « continuous » function did we choose? The Euclidian distance between the last line of a sequence and the first line of another sequence (a line being seen as a vector). Unfortunately, we discovered a bit later in the competition that there were deliberate gaps between the test sequences. We could not reconstitute continuous 5-hour sequences. So we had to give up this technique.

### Model Evaluation
Train and test sets were generated with two distinct groups of individuals. We wanted our cross-validation strategy to reflect this fundamental property. So, we divided our train set into a train subset and a validation subset in the following way: data generated by users 6 and 10 formed the validation subset – the 8 remaining users formed the train test.

This cross-validation strategy gave a score that was approximately 0.1 higher than our score on public leaderboard – 50 % of the test data. Therefore, we were pretty sure that an improvement in our cross-validation score would lead to an improvement on public leaderboard.

However, this strategy may not be optimal. It might even cause overfitting if users 6 and 10 – who form the validation set and who were chosen for early stopping for XGBoost grid search parameters – turn out to be more similar to the users involved in public leaderboard, than to those involved in private leaderboard!

### Potentially Helpful Features
On the test set, would have liked to have a proper user ID variable.

Would have been interesting to have longer sequences that 10-30 sequences to find long term patterns in the data.

### Graphic Representation
The following demonstrates the flow of the project.

![Project Flow](Dataiku_DrivenData/reports/figures/Flow_of_the_Project.jpg)


During the project, the following graph of acceleration in the x direction over time was developed.

![Acceleration](Dataiku_DrivenData/reports/figures/evolution_acceleration_time.jpg)

### Future Steps
I would have tried to make the custom XGBoost work. I would have submitted postprocessing.

I would have implemented Hidden Markov Models instead of post processing.

# Replicating the Submission

Start with a brand new Max OS X.

### Install Python 2.7.5
* Install packages:
    * numpy  1.11.1
    * pandas  0.18.1
    * sklearn 0.17.1
    * xgboost 0.4
    * csv 1.0

### Run Following Scripts in Order
1. `/src/data/make_dataset_annotators_id_test.py`

2. `/src/data/make_dataset_columns_train.py`

3. `/src/data/make_dataset_columns_train_prepared.py`

4. `/src/data/make_dataset_targets_train.py`

5. `/src/data/make_dataset_columns_train_with_targets.py`

6. `/src/data/make_dataset_location_train.py`

7. `/src/data/make_dataset_columns_train_with_targets_and_location.py`

8. `/src/data/make_dataset_info_on_columns_test.py`

9. `/src/data/make_dataset_columns_train_with_targets_and_location_splitted.py`

10. `/src/data/make_dataset_columns_test.py`

11. `/src/data/make_dataset_columns_test_prepared.py`

12. `/src/features/build_features_location.py`

13. `/src/features/build_features_test_feature_engineering.py`

14. `/src/features/build_features_train_feature_engineering.py`

15.  `/src/models/gridsearch_baselearner_xgboost.py`
    * This script performs a gridsearch to find the optimal parameters of the XGBoost model. It does not output anything : you must read the logs to discover the optimal parameters. Once you have read, in the logs, the value of the optimal parameters, you should use these parameters in the script `/src/models/train_and_predict_baselearners.py`

16. `/src/models/train_and_predict_baselearners.py`
    * In this script, we create an XGBoost classifier. You should replace the values of the parameters of the XGBoost by the optimal parameters found in the script `/src/models/gridsearch_baselearner_xgboost.py`

17. `/src/models/gridsearch_stacking.py`
    * This script performs a gridsearch to find the optimal parameters of the stacking XGBoost model. It does not output anything : you must read the logs to discover the optimal parameters. Once you have read, in the logs, the value of the optimal parameters, you should use these parameters in the script `/src/models/train_and_predict_stacking.py`

18. `/src/models/train_and_predict_stacking.py`
    * In this script, we create a XGBoost classifier that will perform stacking. You should replace the values of the parameters of the XGBoost by the optimal parameters found in the script `/src/models/gridsearch_stacking.py`
