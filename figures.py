import pandas as pd
import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
#from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.classifier import classification_report, confusion_matrix
from collections import defaultdict
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import forest


df = pd.read_csv('https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_renamed.csv')

def df_info(df=df):
    buffer = io.StringIO()

    df.info(buf=buffer)

    s = buffer.getvalue()

    x = s.split('\n')

    list_1 = []

    for i in x[3:-3]:
        str_list = []
        for c in i.split(' '):
            if c != '':
                str_list.append(c)
        list_1.append(str_list)

    df_info = pd.DataFrame(list_1)

    df_info.drop(2, axis=1, inplace=True)

    df_info.columns = ['Variable', '# non-null values', 'Data type']

    df_info['# non-null values'] = df_info['# non-null values'].astype(int)

    nunique_list = []

    for i in df_info['Variable']:
        nunique_list.append(df[i].nunique())

    df_info['# unique values'] = nunique_list

    df_info = df_info.sort_values(by='# unique values', ascending=False)

    return df_info


def highlight_cols(s):
    color = 'lightgreen'
    return 'background-color: %s' % color


def sunburst_fig():
    df = pd.read_csv('https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_renamed.csv')

    ## Get categorical column names
    cat_list = []

    for col in df.columns:
        if df[col].dtype == object:
            cat_list.append(col)

    ## Get all possible levels of every categorical variable and number of data points in each level
    cat_levels = {}

    for col in cat_list:
        levels = df[col].value_counts().to_dict()
        cat_levels[col] = levels

    ## Convert nested dictionary to dataframe
    nestdict = pd.DataFrame(cat_levels).stack().reset_index()

    nestdict.columns = ['Level', 'Category', 'Population']

    nestdict['Category'] = [s + ": " for s in nestdict['Category']]

    cat_list = nestdict['Category'].unique()

    empty_list = [None] * len(cat_list)

    pop_list = ['0'] * len(cat_list)

    df1 = pd.DataFrame()

    df1['Level'] = cat_list

    df1['Category'] = empty_list

    df1['Population'] = pop_list

    df = pd.concat([df1, nestdict])

    df['Population'] = df['Population'].astype(int)

    fig = go.Figure(go.Sunburst(
        labels=df['Level'],
        parents=df['Category'],
        values=df['Population'],
        leaf={"opacity": 0.4},
        hoverinfo='skip'
    ))

    fig.update_layout(width=700, height=900, margin = dict(t=0, l=0, r=0, b=0))

    return fig

def param_search():
    # datasets = ['./telco_split_sets.pickle', './random_split_sets.pickle',
    #             './smote_split_sets.pickle', './cluster_split_sets.pickle']
    #
    # params_list = []
    #
    # for dataset in datasets:
    #     infile = open(dataset, "rb")
    #     X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)
    #
    #     params = {
    #         'min_samples_leaf': [1, 3, 5, 10, 25, 100],
    #         'max_features': ['sqrt', 'log2', 0.5, 1]
    #     }
    #
    #     rf = RandomForestClassifier(n_estimators=50, oob_score=True)
    #     grid = GridSearchCV(rf, param_grid=params, scoring='f1', n_jobs=-1)
    #     grid.fit(X_train, y_train)
    #
    #     params_list.append(grid.best_params_)
    #
    # with open('./all_params.pickle', 'wb') as f:
    #     pickle.dump(params_list, f)
    #
    # x = pd.DataFrame(params_list)
    #
    # x['Dataset'] = ['Original', 'Random oversampling', 'SMOTE-NC', 'Clusters']
    #
    # x = x[['Dataset', 'max_features', 'min_samples_leaf']]
    #
    # x.to_csv('./param_search_res.csv', index=False)

    df = pd.read_csv('./param_search_res.csv')

    df = df.set_index('Dataset')

    return df


def set_rf_samples(n):
    """ Changes Scikit-learn's random forests to give each tree a random sample of n random rows."""
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(0, n_samples, n))



def train_models():
    datasets = ['./telco_split_sets.pickle', './random_split_sets.pickle',
                './smote_split_sets.pickle', './cluster_split_sets.pickle']

    infile = open('./all_params.pickle', 'rb')
    params_list = pickle.load(infile)

    def score(m):
        res = {"Score on training set" : m.score(X_train, y_train),
               "Score on validation set" : m.score(X_val, y_val),
               "Out of bag score" : m.oob_score_}
        return res

    set_rf_samples(1500)

    score_list = []
    conf_matrix = []
    class_rep = []

    for dataset, params in zip(datasets, params_list):

        infile = open(dataset, "rb")

        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

        m = RandomForestClassifier(n_estimators=500,
                                   min_samples_leaf=params['min_samples_leaf'],
                                   max_features=params['max_features'],
                                   n_jobs=-1, oob_score=True)

        m.fit(X_train, y_train)

        score_list.append(score(m))

        y_pred = m.predict(X_test)

        cmatrix = confusion_matrix(RandomForestClassifier(n_estimators=500,
                                                              min_samples_leaf=params['min_samples_leaf'],
                                                              max_features=params['max_features'],
                                                              n_jobs=-1, oob_score=True),
                                       X_test, y_test,
                                       classes=['No churn', 'Churn'],
                                       cmap="Greens")

        conf_matrix.append(cmatrix)

        crep = classification_report(RandomForestClassifier(n_estimators=500,
                                                                   min_samples_leaf=params['min_samples_leaf'],
                                                                   max_features=params['max_features'],
                                                                   n_jobs=-1, oob_score=True),
                                            X_test, y_test,
                                            cmap="Greens",
                                            classes=['No churn', 'Churn']
                                        )

        class_rep.append(crep)

        #report = classification_report(y_test, y_pred, target_names=['No churn', 'Churn'], output_dict=True)
        #df = pd.DataFrame(report).transpose()

    scores = pd.DataFrame(score_list).set_index(pd.Index(['Original', 'Random oversample', 'SMOTE-NC', 'Clusters']))

    return scores, class_rep, conf_matrix