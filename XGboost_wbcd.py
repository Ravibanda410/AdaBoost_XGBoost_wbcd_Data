# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:38:50 2021

@author: RAVI
"""



###/\/\/\/\/\/\/\./\.\.\.\.\.\.\.\.\/\/\/\/\/\/\/\//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

## For Adaboosting model

## Sol for 2nd Q

import pandas as pd
import numpy as np

wbcd = pd.read_csv("C:/RAVI/Data science/Assignments/Module 20 AdaBoost-Extreme Gradient Boosting/Archive/wbcd.csv")

## EDA
wbcd.columns
wbcd.shape
wbcd.dtypes
wbcd.isna().sum()
wbcd.isnull().sum()

# converting B to Benign and M to Malignant 
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Benign ', wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant ', wbcd['diagnosis'])

## Removing the ID column
wbcd = wbcd.iloc[:, 1:32] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wbcd_n = norm_func(wbcd.iloc[:, 1:])
wbcd_n.describe()

# Input and Output Split
predictors = wbcd_n
type(predictors)

target = wbcd["diagnosis"]
type(target)

## Spliting the data into x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

## Buielding the model
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

## Fitting model on train data
model.fit(x_train, y_train)

## Predicting the model on test data
pred_t = model.predict(x_test)

## Crosstable for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, pred_t)


#  Testing Data accuracy
accuracy_score(y_test, pred_t)
##  0.96491228


# Evaluation on Training Data
pred_tr = model.predict(x_train)

## Crosstable for train data
from sklearn.matrics import accuracy_score, confusion_matrix
confusion_matrix(y_train, pred_tr)

## Accuracy
accuracy_score(y_train, pred_tr)
##  1.0

###/\/\/\/\/\/\/\./\.\.\.\.\.\.\.\.

## For XGBoosting model

# Input and Output Split
## Model buielding
#pip install xgboost
import xgboost as xgb
model = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

## FItting the model to train data
model.fit(x_train, y_train)

## Model evoluting on test data
pred = model.predict(x_test)

## Confusion matrics for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, pred)

## Accuracy for test data
accuracy_score(y_test, pred)
## 0.97368421

## Predicting on train data
pred_tr = model.predict(x_train)

## Confussion matrics
confusion_matrix(y_train, pred_tr)

## Accuracy
accuracy_score(y_train, pred_tr)
## 1.0
