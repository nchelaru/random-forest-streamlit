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
import seaborn as sns


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
    infile = open('./para_df.pickle','rb')

    para_df = pickle.load(infile)

    return para_df





def train_models():
    infile = open('./scores_df.pickle', 'rb')

    scores_df = pickle.load(infile)

    return scores_df


