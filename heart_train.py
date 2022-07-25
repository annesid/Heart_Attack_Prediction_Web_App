#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:35:42 2022

@author: anne
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix

#%%
def cramers_corrected_stat(conf_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum()
    phi2 = chi2/n
    r,k = conf_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%%
CSV_PATH = os.path.join(os.getcwd(),'Dataset','heart.csv')

MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')

#%% Step 1) Load Data

df = pd.read_csv(CSV_PATH)

# Step 2) Data Inspection

df.head()
df.info()
df.describe().T
df.isna().sum()

# CATEGORISED Continuous & Categorical Variable
cont=['age','trtbps','chol','thalachh','oldpeak']
cat =df.drop(labels=cont,axis=1).columns

# eda = EDA()
# eda.displot_graph(cont,df)
# eda.countplot_graph(cat,df)

# thall has null mask as 0
#caa has null mask 4

df.boxplot()

(df['thall']==0).sum()
(df['caa']==4).sum()
#we replace mask value to NaNs
df['thall'] = df['thall'].replace(0,np.nan)
df['caa'] = df['caa'].replace(0,np.nan)
df.isna().sum() 

#%% Step 3) Clean Data
# Outliers - within range. OK.
# clean data using KNN imputer

columns_name = df.columns

knn_imp = KNNImputer()
df = knn_imp.fit_transform(df) 
df =  pd.DataFrame(df) 
df.columns = columns_name

df.describe().T

df.duplicated().sum()
df = df.drop_duplicates()
df.isna().sum() # No NaNs found
#%% Step 4) Features Selection

# X = df.drop(labels='output',axis=1)
y = df['output']
selected_features=[]

# #Cont vs Cat
for i in cont:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),y)
    print(i)
    print(lr.score(np.expand_dims(df[i],axis=-1),y))
    if lr.score(np.expand_dims(df[i],axis=-1),df['output'])>0.6:
        selected_features.append(i)

# #Cat vs Cat

for i in cat:
    print(i)
    matrix = pd.crosstab(df[i],y).to_numpy()
    print(cramers_corrected_stat(matrix))
    if cramers_corrected_stat(matrix) > 0.4:
        selected_features.append(i)
print(selected_features)

# selected features everything 

#%% Step 5) Data Preprocessing

df=df.loc[:,selected_features]
X=df.drop(labels='output',axis=1)
y=df['output']

#Train test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=12)

#%% Model development
#%% 1) Machine learning --> pipeline

# Decision Tree
# KNN
# Random Forest
# Logistic Regression
# SVC
# GradientBoosting

num = 2
#Decision Tree
pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('DecisionTree',DecisionTreeClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_dt = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('DecisionTree',DecisionTreeClassifier())
                            ]) #Pipeline([STEPS])

#KNN
pipeline_mms_knn = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('KNN',KNeighborsClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('KNN',KNeighborsClassifier())
                            ]) #Pipeline([STEPS])

#Random Forest
pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('RandomForest',RandomForestClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_rf = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('RandomForest',RandomForestClassifier())
                            ]) #Pipeline([STEPS])

#Logistic regression
pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Logistic',LogisticRegression())
                            ]) #Pipeline([STEPS])

pipeline_ss_lr = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Logistic',LogisticRegression())
                            ]) #Pipeline([STEPS])
#SVC
pipeline_mms_svc = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('SVC',SVC())
                            ]) #Pipeline([STEPS])

pipeline_ss_svc = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('SVC',SVC())
                            ]) #Pipeline([STEPS])

#Gradient Boosting
pipeline_mms_gb = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('GB',GradientBoostingClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_gb = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('GB',GradientBoostingClassifier())
                            ]) #Pipeline([STEPS])

#create a list to store all the pipelines
pipelines = [pipeline_mms_dt,pipeline_ss_dt,pipeline_mms_knn,pipeline_ss_knn,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_lr,pipeline_ss_lr,
             pipeline_mms_svc,pipeline_ss_svc,pipeline_mms_gb, pipeline_ss_gb]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
scores = []
for i,pipe in enumerate(pipelines):
    scores.append(pipe.score(X_test,y_test))

print(pipelines[np.argmax(scores)])
print(scores[np.argmax(scores)])

best_pipe = pipelines[np.argmax(scores)]

#%% Model Evaluation

y_pred = best_pipe.predict(X_test)
y_true = y_test

cr = classification_report(y_true,y_pred)
print(cr)

#%% GridSearch CV

#follow result pipeline(above) to choose which Gridsearch - Logistic MinMaxScaler

# grid_param = [{'Logistic__penalty':['l1','l2'],
#                 'Logistic__solver':['liblinear']}]

grid_param=[{'Logistic__penalty':['l1','l2'],
             'Logistic__C':np.arange(1,5,.1),
             'Logistic__intercept_scaling':np.arange(1,10,1),
             'Logistic__solver':['newton-cg','lbfgs','liblinear','sag','saga']}]

# grid_param = [{'KNN__n_neighbors':[5,3,11], #must use odd number
#                'KNN__weights':['uniform','distance'],
#                'KNN__p':[1,2]}]

grid_search = GridSearchCV(estimator=pipeline_mms_lr, param_grid=grid_param,
                    cv=5,verbose=1,n_jobs=-1)

grid = grid_search.fit(X_train,y_train)

print(f"The accuracy for this estimator is {grid.best_score_:.00%}")
print(grid.best_params_)

best_model =grid.best_estimator_

#%% Model saving

with open(MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)












