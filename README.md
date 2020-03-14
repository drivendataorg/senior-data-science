[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' style="max-width:85%;>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/P1020431_small.jpg)

# Senior Data Science: Safe Aging with SPHERE
## Goal of the Competition
This challenge is part of a large research project which centers around using sensors and algorithms to help older people live safely at home while maintaining their privacy and independence. Using passive, automated monitoring, the ultimate goal is to look out for a person's well-being without being burdensome or intrusive.

To gather data, researchers in the SPHERE Inter-disciplinary Research Collaboration (IRC) equipped volunteers with accelerometers similar to those found in cell phones or fitness wearables, and then had the subjects go about normal activities of daily living in a home-like environment that was also equipped with motion detectors. After gathering a robust set of sensor data, they had multiple annotators use camera footage to establish the ground truth, labeling chunks of sensor data as one of twenty specifically chosen activities (e.g. walk, sit, stand-to-bend, ascend stairs, descend stairs, etc).

## What's in this Repository
This repository contains code volunteered from leading competitors in the [Senior Data Science: Safe Aging with SPHERE](https://www.drivendata.org/competitions/42/) on DrivenData.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | Daniel_FG | 0.1293 | 0.1346 | I selected 7 subsets of features by random/optimization and I trained a bunch of Layer 1 models with them (ExtraTrees, RandomForest, XGBoost, Neural Networks and Linear Regression).  In Layer 2, I selected again 4 subset of meta-features (L1 models) by random/optimization and I trained 7 L2 models. The final submission is a weighted average of theses predictions.
2 | venat | 0.1343 | 0.1378 | We decided to give the stacking technique a try. This ensemble learning technique combines the predictions of diverse individual models and leverages their strengths. Stacking 10 individual models – including linear regressions, Naive Bayes classifiers, random forests, extra-trees and XGBoost models – turned out to be very performant.
3 | Dataiku | 0.1292 | 0.1381 | We then trained dozens of models using various algorithms such as logistic regression, random forest, and extreme gradient boosting tree. Our final submission is an ensemble model of these base models.


#### Winner's Interview: ["Meet the winners of the safe aging challenge"](http://blog.drivendata.org/2016/10/10/sphere-winners/)

#### Benchmark Blog Post: ["There's no place like $HOME"](http://blog.drivendata.org/2016/06/06/sphere-benchmark/)
