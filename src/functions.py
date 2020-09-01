#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/dmkravtsov/4-1-titanic

# Docker - https://medium.com/@kaggleteam/how-to-get-started-with-data-science-in-containers-6ed48cb08266

# Docker v2 - https://www.docker.com/blog/containerized-python-development-part-1/

# API - https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

# API v2 - https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
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

def detect_outliers(df,n,features):
    """ Takes a dataframe df of features and returns a list of the indices corresponding to the observations 
        containing more than n outliers according to the Tukey method.
        
    Paramaeters:
        df (DataFrame): DF in which we wan to seek for outliers
        n (int): smallest number of outliers in the column
        features (list): a list of collumn names in which we are seeking the outliers
        
    Return:
       multiple_outliers (list): function return a list of multiple outliers across given df parameter
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers 


def missing_ratio(df):
    b = pd.DataFrame()
    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    return b


def completing(in_df):
    """Completing or deleting missing values in train and validation dataset
    
    Parameters:
        in_df (DataFrame): data frame in which we want to 
    """

    # complete missing age with median
    in_df.age.fillna(in_df.age.median(), inplace = True)

    # complete embarked with mode
    in_df.embarked.fillna(in_df.embarked.mode()[0], inplace = True)

    # complete missing fare with median
    in_df.fare.fillna(in_df.fare.median(), inplace = True)

    # delete the cabin
    drop_column = ['cabin']
    in_df.drop(drop_column, axis=1, inplace = True)
    

### FEATURE ENGINEERING FUNCTIONS

def simple_feature_eng(df):
    """creating new features for specified dataset

    Parameters:
        df (DataFrame): dataframe in which feature engineering will take place

    Return:
        df (DataFrame): dataframe in which feature engineering was applied
    """  

    df['familysize'] = df.sibsp + df.parch + 1

    # init 1 as alone
    df['isalone'] = 1 
    # if family is greater than one than isalone = 0 (ain't alone) 
    df['isalone'].loc[df['familysize'] > 1] = 0 

    df['no_fare'] = df['fare'].map(lambda x: 1 if x == 0 else (0))

    # splitting name for creating title column
    df['title'] = df['name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    tickets = []
    for i in list(df.ticket):
        if not i.isdigit():
            tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
        else:
            tickets.append("x")
    df.ticket = tickets

    df = pd.get_dummies(df, columns= ["ticket"], prefix = "t")

    return df

    
def fare_category(fare):
    """Function prepared for fare column to extract 4 groups of fare
    
    """
    if fare <= 7.9:
        return 1
    elif fare <= 14.5 and fare > 7.9:
        return 2
    elif fare <= 31 and fare > 14.5:
        return 3
    return 4


def label_encoding(df):
    """convert objects to category using Label Encoder for given dataset
    
    Parameters:
        in_df (DataFrame):  dataframe in which label encoding will take place
    
    Return:
        in_df (DataFrame): dataframe in which label encoding was applied
        
    """
    
    # code the categorical features
    label = LabelEncoder()
    
    df['sex'] = label.fit_transform(df['sex'])
    df['embarked'] = label.fit_transform(df['embarked'])
    df['title'] = label.fit_transform(df['title'])
    df['name'] = label.fit_transform(df['name'])

    return df