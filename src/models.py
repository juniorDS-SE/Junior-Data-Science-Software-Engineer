#!/usr/bin/env python
# coding: utf-8

# # TASK5 - models

# Propose other prediction models than the one proposed by us.

# ##### Roche team used Random Forest Classifier from Ensemble Methods

# In a nutshell Random Forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting

# ----------------------------------------------------------------------------------------------------------------------

# ### LOAD PACKAGES

# In[152]:


import sys  
sys.path.insert(0, '../src/')


# In[208]:


import random
import time
import datacompy
import sys
import IPython
from IPython import display
import matplotlib 
import pandas as pd
import numpy as np 
import scipy as sp 
import sklearn
import pickle
from collections import Counter
import pickle as pkl
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix
mpl.style.use('ggplot')
sns.set_style('dark')
pylab.rcParams['figure.figsize'] = 14,5
seed =40
plt.style.use('fivethirtyeight')


# In[154]:


import functions
from functions import missing_ratio


# In[155]:


all_df = pd.read_csv("../data/all_df.csv")


# In[156]:


all_df.shape


# ----------------------------------------------------------------------------------------------------------------------

# #### Model that i have choosen is based on my own knowladge and many publications of which one is more worth mentioning than others: https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf 

# ### Split test and train data

# I will use sklearn function to split the training data in two datasets and it will be classic 75/25 split.

# In[157]:


seed = 40


# In[158]:


train = all_df.iloc[:881, :]


# In[159]:


train.tail()


# In[160]:


def nans(df): return df[df.isnull().any(axis=1)]


# In[161]:


nans(train)


# In[162]:


missing_ratio(train)


# In[163]:


test  = all_df.iloc[881:, :]


# In[164]:


test.head()


# In[165]:


missing_ratio(test)


# In[166]:


x_full = pd.concat([train.drop('survived', axis = 1), test.drop('survived', axis = 1)], axis = 0)


# In[167]:


x_full.shape


# In[168]:


x_full.drop('passengerid', axis = 1, inplace=True)


# In[169]:


x_full.isnull().sum()


# In[170]:


x_full.shape


# In[171]:


x_dummies = pd.get_dummies(x_full, drop_first= True)


# In[172]:


x_dummies.shape


# In[173]:


x_dummies.dtypes


# In[174]:


X = x_dummies[:len(train)]; new_x = x_dummies[len(train):]


# In[175]:


y = train.survived


# In[176]:


from sklearn.model_selection import train_test_split


# In[177]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = .3,
                                                    random_state = 5,
                                                   stratify = y)


# In[178]:


from xgboost import XGBClassifier


# In[179]:


xgb = XGBClassifier()


# In[180]:


xgb.fit(X_train, y_train)


# In[181]:


xgb.score(X_test, y_test)


# In[182]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


# In[183]:


# the parameter grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}


# In[184]:


# instantiate the regressor
gbm = XGBClassifier(n_estimators=10)


# In[185]:


# perform random search
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)


# In[186]:


# fit randomized_mse
xgb_random.fit(X, y)


# In[214]:


filename = 'model_klimarczyk.pkl'
model_pickle = open(f"../data/{filename}", 'wb')
pickle.dump(gbm, model_pickle)
model_pickle.close()


# In[215]:


# best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)


# In[188]:


xgb_pred = xgb_random.predict(new_x)


# In[221]:


# load the model
loaded_model = pickle.load(open(f"../data/{filename}", 'rb'))
loaded_model.fit(X, y)
loaded_model.score(X, y)


# -------------

# In[189]:


tpid = test.passengerid


# In[190]:


type(tpid)


# In[191]:


tpid = tpid.to_frame() 


# In[193]:


tpid.columns = ['PassengerId']


# In[194]:


type(tpid)


# In[195]:


pdfxgb = pd.DataFrame(xgb_pred)


# In[196]:


pdfxgb.columns = ["Survived"]


# In[197]:


pdfxgb.shape


# In[198]:


type(pdfxgb)


# In[199]:


submission = pd.concat([tpid.reset_index(drop=True), pdfxgb.reset_index(drop=True)], axis = 1).reset_index(drop=True)


# In[200]:


submission


# In[201]:


submission.to_csv('../data/titanic_submission.csv', header = True, index = False)


# In[202]:


test_org = pd.read_csv('../data/test.csv')


# In[203]:


validation = pd.merge(test_org.reset_index(drop=True), submission.reset_index(drop=True), on='PassengerId')
validation


# In[204]:


validation.to_csv('../data/klimarczyk_validation.csv', header = True, index = False)

