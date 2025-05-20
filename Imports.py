# Import Nb as Python file
import os
import timeit
import time
import math
import gc
import glob
import pickle
import statistics
import pandas as pd
import numpy as np
from scipy import stats
import shap
from scipy.special import softmax
import statistics
from statistics import mean
import optuna

from interpret.glassbox import ExplainableBoostingClassifier

from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

from dask import dataframe as dd
from dask.distributed import Client

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import collections
from collections import OrderedDict
from collections import Counter

from plotly.offline import iplot
import chart_studio.plotly as py
import plotly.graph_objs as go

import csv
from itertools import zip_longest
import dataframe_image as dfi

# from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# from factor_analyzer.factor_analyzer import calculate_kmo

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, ParameterGrid, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, silhouette_score

from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.utils import resample

import feather
import pyreadr

# from skopt import BayesSearchCV
from pprint import pprint
import xgboost as xgb

from copy import deepcopy
import plotnine

from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")
# from IPython.display import Markdown, display

# "magic" command to make plots show up in the notebook
# %matplotlib inline

pd.options.display.max_rows = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option("display.precision",2)
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True)

# sns.set_style('darkgrid')
sns.set_theme(style="whitegrid", font_scale=1.5)

date_format = "%Y-%m-%d"
filepath = 'input/'

target = 'arrears'

sens_char = {'gender': 'clientdata.demo.gender',    # binary
             'age': 'clientdata.demo.age_year',     # integer
             'num_children': 'clientdata.demo.children',    # integer (0-10)
             'single_parent': 'clientdata.demo.children_singleparent',  # binary
             'ms_single': 'clientdata.demo.maritalstatus_expand_SINGLE',    # binary
             'ms_married': 'clientdata.demo.maritalstatus_expand_MARRIED',  # binary
             'ms_divorced': 'clientdata.demo.maritalstatus_expand_DIVORCED',    # binary
             'ms_widowed': 'clientdata.demo.maritalstatus_expand_WIDOWED'   # binary
            }


def print_value_counts(df, column, name):
    print(f"\n{name} - {column} Distribution")
    print(df[column].value_counts())
    print(df[column].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')


def null_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'n_missing', 1: 'perc'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
    'perc', ascending=False).round(1)
    print("Dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    mis_val_df = pd.DataFrame(mis_val_table_ren_columns)
#        mis_val_df.to_csv('inputs/mis_values.csv')
    return mis_val_df


def read_RDa(filename, df_name):
    result = pyreadr.read_r(filename)  # also works for Rds, rda
    print(result.keys())    # let's check what objects we got
    df = result[df_name]    # extract the pandas data frame for object df1
    return df


def check_intersect(info_df, features_df):
    # Checking both datasets fully intersect
    uuid_info = list(info_df['uuid'])
    uuid_features = list(features_df['uuid'])
    print(len(uuid_info))
    print(len(uuid_features))
    return set(uuid_info) ^ set(uuid_features)  # difference


def age_mapping(rating):
    '''
    Description:
        {'18 - 20': 0 | '21 - 25': 1 | '26 - 30': 2 | '31 - 35': 3 | '36 - 40': 4 | '41 - 45': 5 | '46 - 50': 6 |
        '51 - 55': 7 | '56 - 60': 8 | '61 - 65': 9 | '66+':10}
    '''
    if rating <= 20:
        return 0
    if rating <= 25:
        return 1
    if rating <= 30:
        return 2
    if rating <= 35:
        return 3
    if rating <= 40:
        return 4
    if rating <= 45:
        return 5
    if rating <= 50:
        return 6
    if rating <= 55:
        return 7
    if rating <= 60:
        return 8
    if rating <= 65:
        return 9
    return 10


def age_mapping_binary(rating):
    '''
    Description:
        {'<25': 0 (young adult) | '>=25': 1 }
    '''
    if rating < 25:
        return 0
    if rating >= 25:
        return 1


# Map number of children into 4 groups
def children_mapping_3(rating):
    '''
    Description:
        {'0 children': 0 | '1-2 children': 1 | 3+ children: 2}
    '''
    if rating < 1:
        return 0
    if rating < 3:
        return 1
    return 2


# Output
# def output_SC_summary(df, name):
#     # Create summary excel
#     appended_data = []
#     for k, v in sens_char_ext.items():
#         summary = pd.crosstab(df[v], df['arrears']).reset_index()
#         summary.drop(columns=summary.columns[0], axis=1, inplace=True)
#         summary.insert (0, 'feature_value', summary.index)
#         summary.insert(0, 'feature', [k]*len(summary))
#         summary.insert(4, 'total count', list(df[v].value_counts().sort_index(ascending=True).values))
#         # summary.insert(4, 'total count', [df[v].value_counts().sort_index(ascending=True)[i] for i in range(len(summary))])
#         appended_data.append(summary)
#     # see pd.concat documentation for more info
#     appended_data = pd.concat(appended_data)
#     # write DataFrame to an excel sheet
#     appended_data.to_excel('./outputs/1_SC/' + name + 'SC_summary.xlsx')

#     return appended_data


# Step 4: Prediction Model functions
def split_data(df):
    """
    Splits the input dataframe into training and testing sets, returning two versions:
      1. One with demographic features removed.
      2. One including all features (the 'demo' features are retained).

    Assumes the following global variables exist:
      - target: The name of the target variable.
      - key_featsubgroups: A DataFrame with a 'subgroup' column and a 'list_features' column.
        The row where subgroup == 'demo' contains the list of demographic features.

    Parameters:
      df : DataFrame
          The input dataset containing features and the target variable.

    Returns:
      X_train : DataFrame
          Training features with demographic features removed.
      X_test : DataFrame
          Testing features with demographic features removed.
      y_train : Series
          Training target values.
      y_test : Series
          Testing target values.
      X_train_demo : DataFrame
          Training features with demographic features retained.
      X_test_demo : DataFrame
          Testing features with demographic features retained.
    """
    # Get the list of demographic features from key_featsubgroups.
    demo_feat = list(key_featsubgroups.loc[key_featsubgroups.subgroup == 'demo', 'list_features'])[0]
    # Separate features and target.
    X = df.drop([target], axis=1)
    y = df[target]

    # Split the data into training and testing sets (keeping demo features).
    X_train_demo, X_test_demo, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

    # Create versions with demo features removed.
    X_train = X_train_demo.drop(demo_feat, axis=1)
    X_test = X_test_demo.drop(demo_feat, axis=1)

    # Print shapes for verification.
    print('Training Shape (incl. demo features):', X_train_demo.shape)
    print('Training Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)

    print('Testing Shape (incl. demo features):', X_test_demo.shape)
    print('Testing Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    return X_train, X_test, y_train, y_test, X_train_demo, X_test_demo

def model_pred(X_train, X_test, y_train, y_test, model, name, params, results_dict):
    """
    Trains the given model, evaluates its performance, and stores results.

    Parameters:
    X_train (DataFrame): Training feature set.
    X_test (DataFrame): Testing feature set.
    y_train (Series): Training labels.
    y_test (Series): Testing labels.
    model (sklearn classifier): The machine learning model to be trained.
    name (str): The name of the model for result storage.
    params (str): Specifies whether the model is using hyperparameter search ('search') or optimized parameters ('opt').
    results_dict (dict): Dictionary to store model performance metrics.

    Returns:
    dict: Updated results dictionary containing model evaluation metrics.
    """

    start_time = time.time()
    print("\nStarting model training and evaluation...")

    # Train the model based on parameter selection method
    if params == "search":
        model.fit(X_train, y_train.values.ravel())  # Ensure correct shape for sklearn
    elif params == "opt":
        model.fit(X_train, y_train)

    # Make predictions on train and test sets
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Compute confusion matrices
    cfm_train = confusion_matrix(y_train, train_predictions)
    cfm_test = confusion_matrix(y_test, test_predictions)

    # Compute accuracy scores
    accs_train = accuracy_score(y_train, train_predictions)
    accs_test = accuracy_score(y_test, test_predictions)

    # Compute F1-scores for both classes (0 and 1)
    f1s_train_p1 = f1_score(y_train, train_predictions, pos_label=1)
    f1s_train_p0 = f1_score(y_train, train_predictions, pos_label=0)
    f1s_test_p1 = f1_score(y_test, test_predictions, pos_label=1)
    f1s_test_p0 = f1_score(y_test, test_predictions, pos_label=0)

    # Compute ROC-AUC score for test data
    test_ras = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Extract best hyperparameters
    if params == "search":
        bp = model.best_params_
    elif params == "opt":
        bp = model.get_params()

    # Calculate total runtime in minutes
    total_time = (time.time() - start_time) / 60
    print(f"Total execution time: {total_time:.2f} minutes")

    # Store computed values in results dictionary (keeping original key names)
    results_dict[name] = {
        "classifier": deepcopy(model),
        "cfm_train": cfm_train,
        "cfm_test": cfm_test,
        "train_accuracy": accs_train,
        "test_accuracy": accs_test,
        "train F1-score label 1": f1s_train_p1,
        "train F1-score label 0": f1s_train_p0,
        "test F1-score label 1": f1s_test_p1,
        "test F1-score label 0": f1s_test_p0,
        "test roc auc score": test_ras,
        "best_params": bp,
        "time_m": total_time
    }

    return results_dict


# Step 5: Model Evaluation
def plot_roc_curve(X_test, y_test, model):
    # predict probabilities
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    # plot no skill roc curve
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(y_test, preds)
    # calculate roc auc
    roc_auc = roc_auc_score(y_test, preds)
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label='Logistic AUC = %0.2f' % roc_auc)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
#     plt.savefig(output_folder + model_name + '_roc_curve.png', bbox_inches='tight')

    plt.show()
#     return preds

# plot no skill and model precision-recall curves
def plot_pr_curve(y_test, preds):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_test[y_test==1]) / len(y_test)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, preds)
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label='Logistic PR AUC: %.3f' % auc_score)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
#     plt.savefig(output_folder + model_name + '_pr_curve.png', bbox_inches='tight')
    # show the plot
    plt.show()
