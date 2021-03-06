import pandas as pd
from pandas.api.types import is_numeric_dtype
import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from figures import *
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from yellowbrick.classifier import classification_report
#from yellowbrick.classifier import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import forest
from rfpimp import *
from functools import reduce
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *



# Set outline
pages = ["1. Introduction",
         "2. Data cleaning",
         "3. Explore the dataset",
         "4. Variable encoding",
         "5. Create training, validation and test sets",
         "6. Hyperparameter tuning and model fitting",
         "7. Feature importance",
         "8. Where to go from here?"]

page = st.sidebar.selectbox('Navigate', options=pages)



# 1. Introduction
if page == pages[0]:

    st.sidebar.markdown('''
    
    ---
    
    More on random forest models:
    
    - Bagging and Random Forest Ensemble Algorithms for Machine Learning [[Machine Learning Mastery]](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)
    - Algorithm Selection - Decision Tree Algos [[EliteDataScience]](https://elitedatascience.com/algorithm-selection)
    - Random forests [[University of Cincinnati]](https://uc-r.github.io/random_forests)
    - `RandomForestClassifier` class [[`scikit-learn` documentation]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    
    ''')

    '''
    Of the myriad of supervised machine learning algorithms available, the **random forest model** is well-favoured for its robust 
    out-of-box performance on datasets containing diverse data types. As it has very few statistical assumptions and require little feature
     engineering, it is a great option as the first model to fit onto a new dataset to explore how much "learnable" signal can found before attempting
      more sophisticated methods.
      
    ###
    '''

    st.image('./forest.png', caption='Image credit: Icons 8', use_column_width=True)

    '''
    ###
    
      As an ensemble method, the random forest model builds multiple decision trees and aggregate their results to create a more accurate 
    prediction than any of the individual models is capable of. Two features of the random forest model help to reduce the correlation between 
     the decision trees that hinders performance of conventional bagged decision tree models:
    '''

    st.success(''' 
    1. Each decision tree trains on a different bootstrap sample (*subset*) of the total dataset 
    2. At each split point, the decision tree can search through only a *random sample* of all available features
    
    ''')

    st.image('./random_forest.png', use_column_width=True, caption="Image credit: https://www.youtube.com/watch?v=-bYrLRMT3vY")

    '''
    
    Together, these two features help create an ensemble of decision trees that are more able to
     capture different "facets" of the patterns in the dataset, thereby aggregating to more robust (less biased) predictions. To 
     learn more about the concepts and applications of random forest models, the sidebar provides several resources that 
     I have found particularly helpful when first learning about this topic.

    As supervised machine learning is now widely used in guiding business decision making, in this microlearning series we will 
    use the [IBM Telco dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) as an example how the random forest model 
    can be used in predicting customer churn.
    
    Here, we will walk through the steps of a full machine learning workflow that incorporates both unsupervised and supervised 
    modeling approaches, from data preprocessing to calculating variable importance. Head over to the dropdown menu in the sidebar to get started!
    '''





# 2. Data cleaning
if page == pages[1]:

    st.sidebar.markdown('''
    ---
    
    
    ''')

    section = st.sidebar.radio("Steps:",
                     ('1. Preview data',
                      '2. Remove customer ID column',
                      '3. Re-encode variable',
                      '4. Rename column headers',
                      '5. Correct column data type',
                      '6. Combine variable levels'))


    st.sidebar.markdown('''
    ---
    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    ''')


    if section == '1. Preview data':
        st.title('2.1. Preview data')

        '''
        The IBM Telco customer churn dataset, which details the personal characteristics and 
        purchasing behaviours of ~7,000 previous or current customers, is a great example for exploring a typical 
        use case of supervised machine learning in the business world.
        
        We will be building a random forest model to predict whether a customer will churn based on his or her demographics and purchasing 
        history with the company, which can be used by the marketing and sales departments to inform advertising
         and/or retention campaigns.
        '''

        '''
        ```Python
        import pandas as pd
        
        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        ```
        '''

        if st.checkbox("Preview data"):
            import pandas as pd

            df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

            st.dataframe(df.head(10).T)


            st.header(" ")


            '''
            To see a summary of this dataset:
            
            ```Python
            df.info()
            ``` 
            '''

            if st.checkbox("TL;DR?"):
                st.dataframe(df_info(df))

                st.markdown('''
                There are some quick clean-ups that we need to go on this dataset before it is ready for modeling. We will walk through 
                each step in the next few pages.
                ''')




    if section == '2. Remove customer ID column':
        st.title("2.2 Remove customer ID column")

        '''
        To start off, the `customerID` column contains the uniquely generated ID associated with each customer. As we would like to train 
        a model that captures *general* patterns in customer characteristics and preferences in relation to the likelihood of churn, it may 
        be preferable to remove this feature from the model.
        '''

        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        with st.echo():
            df.drop(['customerID'], axis = 1, inplace=True)

        if st.checkbox('Drop customer ID column'):
            '''
            And... The `customerID` column is indeed gone...
            '''

            st.dataframe(df_info(df))




    if section == '3. Re-encode variable':
        st.title("2.3 Re-encode variable levels")

        ## Data processing up to now
        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)


        '''
        Next, unlike other categorical variables in this dataset, `SeniorCitizen` is encoded by 0s and 1s, presumably corresponding to
        "No"s and "Yes"s. 
        '''

        st.dataframe(df.tail(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['SeniorCitizen']]))

        '''
        Let's see the distribution of values in this variable:

        ```Python
        df['SeniorCitizen'].value_counts()
        ```
        '''

        st.warning(df['SeniorCitizen'].value_counts().to_dict())

        '''
        While we will dummy encode the categorical variables in 0s and 1s later anyways, for the sake of consistency during the 
        data exploration phase, we will re-encode this variable in the same manner as the others: 
        '''

        with st.echo():
            df['SeniorCitizen'] = np.where(df['SeniorCitizen'] == 1, 'Yes', 'No')

        if st.checkbox("Re-encode"):
            st.dataframe(df.tail(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['SeniorCitizen']]))

            '''
            ```Python
            df['SeniorCitizen'].value_counts()
            ```

            As a sanity check, we see that the proportion of "Yes"/"No" is the same as "0"/"1" after re-encoding.
            '''

            st.warning(df['SeniorCitizen'].value_counts().to_dict())




    if section == '4. Rename column headers':
        st.title("2.4 Rename column headers")

        '''
        This is a minor point, but for the sake of consistency, we will capitalize the headers for
         `Gender` and `Tenure`: 
        '''

        with st.echo():
            df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        if st.checkbox("Rename columns"):
            '''
            For a quick check:
            '''

            st.write(list(df.columns))


    if section == '5. Correct column data type':
        st.title("2.5 Correct column data type")

        '''
        Setting the correct data type for each column in a `pandas` dataframe is pretty important for the data to be treated in the 
        "correct manner" in the preprocessing and model fitting process.  
        '''

        st.header("2.5.1 Set `TotalCharges` as numeric type")

        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)

        df['SeniorCitizen'] = np.where(df['SeniorCitizen'] == 1, 'Yes', 'No')

        df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        v = df_info(df)

        '''
        First up, we need convert the variable `TotalCharges` to the right data type (`float`), as it is currently 
        set as a categorical variable (`object`), likely because the numbers are encoded as strings:
        '''

        st.dataframe(v.style.apply(lambda x: ['background: lightgreen' if x.name in [18]
                                          else '' for i in x],
                               axis=1))

        '''
        ```Python
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        ```
        '''

        if st.checkbox("Convert"):
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            x = pd.DataFrame(df_info(df))

            x['# unique values'] = x['# unique values'].astype(str)

            st.dataframe(x.style.highlight_min(axis=0))

            st.warning('''
            Interestingly, we now see that there are 11 values missing in the `TotalCharges` column. 
            ''')

            st.markdown('''
            Let's see what's up with these 11 records:
            ''')

            st.markdown('''
            ```Python
            df[df['TotalCharges'].isnull()]
            ```
            ''')

            if st.checkbox("Check data"):
                v = df[df['TotalCharges'].isnull()]

                st.dataframe(v.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Tenure']]))

                '''
                It makes sense that for customers who just started with the company to not have total charges calculated. 
                
                To check that only customers with `Tenure` = 0 have no total charges recorded:

                ```Python
                df[(df['Tenure'] == 0) | (df['TotalCharges'].isnull())]
                ```
                '''

                if st.checkbox("Let's see..."):
                    df = df[['Gender', 'SeniorCitizen', 'Tenure', 'TotalCharges',  'Partner', 'Dependents',
                             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                             'MonthlyCharges',  'Churn']]

                    p = df[(df['Tenure'] == 0) | (df['TotalCharges'].isnull())]

                    st.dataframe(p.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Tenure', 'TotalCharges']]))


                    st.success('''
                    Indeed, the two sets of problematic data points are the same. 
                    ''')

                    '''
                    So, we can replace the null `TotalCharges` values with 0 for these 11 customers.
                    '''

                    with st.echo():
                        df.fillna(0, inplace=True)

                    st.dataframe(df[df['Tenure'] == 0].style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Tenure', 'TotalCharges']]))

                    st.header('2.5.2 Set categorical variables as "category" type')

                    '''
                    This step is not strictly necessary, but setting the column data types of categorical variables
                    to "category", instead of leaving them as "object", has some advantages:
                    
                    - Reduces memory usage by the dataset
                    - Where needed, can set order of the variable levels: *i.e.* "Small" < "Medium" < "Large"
                    - Help some statistical and plotting packages to recognize categorical data types more easily
                    '''

                    with st.echo():
                        for i in df:
                            if df[i].dtype == 'object':
                                df[i] = df[i].astype('category')


                    if st.checkbox("Just to be thorough..."):
                        st.dataframe(df_info(df))





    if section == '6. Combine variable levels':
        st.title("2.6 Combine categorical variable levels")

        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)

        df['SeniorCitizen'] = np.where(df['SeniorCitizen']==1, 'Yes', 'No')

        df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        df.fillna(0, inplace=True)

        '''
        Finally, some categorical variables in this dataset have levels that can be grouped together, so as to reduce cardinality.
        
        We can look at the levels of each categorical variable by plotting:
    
        ```Python
        ## Set up subplot grid
        fig, axes = plt.subplots(nrows = 9, ncols = 2,
                                 sharex = False, sharey = False,
                                 figsize=(8, 15))

        cat_list = []

        for i in df.columns:
            if df[i].dtype == 'object':
                cat_list.append(i)

        for cat, ax in zip(cat_list, axes.flatten()):
            if df[cat].dtype == 'object':
                df[cat].value_counts().plot.barh(ax=ax)
                ax.set_title(cat, fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(False)

        fig.subplots_adjust(top=0.92, wspace=0.2, hspace=0.3)
        ```
        '''



        if st.checkbox("Abracadabra!"):
            with st.spinner('Working on it...'):
                ## Set style
                plt.style.use('seaborn-ticks')

                plt.rcParams.update(
                    {'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 8, 'ytick.labelsize': 14,
                     'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 18})

                ## Set up subplot grid
                fig, axes = plt.subplots(nrows = 9, ncols = 2,
                                         sharex = False, sharey = False,
                                         figsize=(8, 15))

                cat_list = []

                for i in df.columns:
                    if df[i].dtype == 'object':
                        cat_list.append(i)

                for cat, ax in zip(cat_list, axes.flatten()):
                    if df[cat].dtype == 'object':
                        df[cat].value_counts().plot.barh(ax=ax)
                        ax.set_title(cat, fontsize=14)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.yaxis.set_ticks_position('left')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.set_xticks([0, 2500, 5000])
                        ax.grid(False)

                fig.subplots_adjust(top=0.92, wspace=0.2, hspace=0.2)

                plt.tight_layout()


                fig.delaxes(axes[8][1])

                st.pyplot()

                plt.clf()

            st.info('''
            For many of the variables related to online services and phone service, "No internet service" or "No phone service" 
            can be combined into the "No" category for that variable, as that information is already encoded in "No" for 
            `InternetService` and `PhoneService`.
             
             Combining these levels into their respective "No" groups will help to reduce noise in the dataset and improve interpretability 
             of the resulting model.
             ''')

            st.markdown('''
            So, let's rename those levels:
            ''')


            with st.echo():
                for col in ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                    df[col] = df[col].replace({'No internet service':'No'})

                df['MultipleLines'] = df['MultipleLines'].replace({'No phone service':'No'})

            if st.checkbox("Look at the variables again"):
                with st.spinner('Working on it...'):
                    ## Set style
                    plt.style.use('seaborn-ticks')

                    plt.rcParams.update(
                        {'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 8, 'ytick.labelsize': 14,
                         'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 18})

                    ## Set up subplot grid
                    fig, axes = plt.subplots(nrows = 9, ncols = 2,
                                             sharex = False, sharey = False,
                                             figsize=(8, 15))

                    cat_list = []

                    for i in df.columns:
                        if df[i].dtype == 'object':
                            cat_list.append(i)

                    for cat, ax in zip(cat_list, axes.flatten()):
                        if df[cat].dtype == 'object':
                            df[cat].value_counts().plot.barh(ax=ax)
                            ax.set_title(cat, fontsize=14)
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.yaxis.set_ticks_position('left')
                            ax.xaxis.set_ticks_position('bottom')
                            ax.set_xticks([0, 2500, 5000])
                            ax.grid(False)

                    fig.subplots_adjust(top=0.92, wspace=0.2, hspace=0.3)

                    plt.tight_layout()

                    fig.delaxes(axes[8][1])

                    st.pyplot()

                    plt.clf()

                st.success("Now, we are ready to do some exploratory visualizations with this dataset!")





# 3. Data exploration
if page == pages[2]:
    # Import data
    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    st.sidebar.markdown('''
    
    ---
    
    Before diving into model building, it is important to get familiar with the cleaned dataset. 
    
    Select any one or two variables in the dropdown menus below
    to create a exploratory visualization.
    
    
    ''')

    cat_list = sorted([' ', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'Churn', 'Tenure'])

    var1 = st.sidebar.selectbox("Select variable 1", cat_list)

    var2 = st.sidebar.selectbox("Select variable 2", cat_list)

    plt.style.use('seaborn-ticks')

    plt.rcParams.update({'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 14,
                         'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 'medium',
                         'figure.figsize': (10, 8)})

    if var1 == ' ' and var2 == ' ':
        '''
        Click on any of the categorical variable names to expand the
         sunburt chart and see distribution of the levels in more detail, where the size of each leaf is proportional 
         to the number of customers in that level.
        '''

        with st.spinner('Working on it...'):

            fig = sunburst_fig()

            st.plotly_chart(fig, width=700, height=700)
    elif var1 != ' ' and (var2 == ' ' or var2 == var1) and df[var1].dtype == 'object':
        '''
        There are 7,043 customers in the dataset. Each symbol represents ~100 customers.
        '''

        with st.spinner('Working on it...'):
            data = df[var1].value_counts().to_dict()

            fig = plt.figure(
                FigureClass=Waffle,
                rows=5,
                columns=14,
                values=data,
                legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.2), "fontsize":20, 'ncol':2},
                icons='user',
                font_size=38,
                icon_legend=True,
                figsize=(12, 8)
            )

            # plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif var1 != ' ' and var2 == ' ' and df[var1].dtype != 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, nbins=n_bins, width=1200)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              )
                              )

            st.plotly_chart(fig)
    elif var1 != var2 and df[var1].dtype == 'object' and df[var2].dtype == 'object':
        with st.spinner('Working on it...'):

            sns.set_context("talk", rc={'axes.titlesize': 20, 'legend.fontsize': 14.0, 'legend.title_fontsize': 24.0})

            fig = sns.countplot(x=var1, hue=var2, data=df, palette="Set3")

            sns.despine(right=True, top=True)

            plt.ylabel('Count')

            if var1 == 'PaymentMethod':
                plt.xticks(rotation=30, ha="right")
            else:
                pass

            plt.grid(False)

            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype == 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=var2, opacity=0.4, barmode = 'overlay', nbins=n_bins, width=1000)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              legend_orientation="h",
                              legend=dict(x=0, y=1.1),
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              ))

            st.plotly_chart(fig)
    elif df[var1].dtype == 'object' and df[var2].dtype != 'object':
        with st.spinner('Working on it...'):
            sns.set_context("talk", rc={'axes.titlesize': 20, 'legend.fontsize': 14.0, 'legend.title_fontsize': 24.0})

            fig = sns.barplot(x=var1, y=var2, data=df, palette="Set3")

            sns.despine(right=True, top=True)

            if var1 == 'PaymentMethod':
                plt.xticks(rotation=20, ha="right")
            else:
                pass


            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype != 'object' and var1 == var2:
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=None, nbins=n_bins, width=1000)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              legend_orientation="h",
                              legend=dict(x=0, y=1.1),
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              ))

            st.plotly_chart(fig)
    elif var1 != var2 and df[var1].dtype != 'object' and df[var2].dtype != 'object':
        with st.spinner('Working on it...'):
            sns.set_context("talk", rc={'axes.titlesize': 18, 'xtick.labelsize': 12,
										 'ytick.labelsize': 12})

            sns.jointplot(df[var1], df[var2], kind="hex", color="#4CB391")

            st.pyplot()

            plt.clf()

    else:
        pass



# 3. Variable encoding
if page == pages[3]:
    st.sidebar.markdown('''
    ---
    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    More about variable encoding:
    
    - Why One-Hot Encode Data in Machine Learning? [[Machine Learning Mastery]](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
    - List of categorical variable encoding techniques implemented in `scikit-learn` style 
    [[`CategoryEncoders` documentation]](http://contrib.scikit-learn.org/categorical-encoding/index.html)
    
    ''')

    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    '''
    Most machine learning models cannot handle label data, instead requiring categorical variable levels to be mapped to numeric values.
    There is a wide range of techniques available for this task, each appropriate for different types of data and machine learning models. 
    If you are interested, check out the list of such techniques implemented in `scikit-learn` style by the `CategoryEncoders` package (see sidebar).
    
    In this dataset, for majority of the variables that have "Yes"/"No" possible values, we can use integer encoding where they are replaced by 1/0, respectively.  However, for variables that have 
    more than two levels or the two levels do not have an inherent ordinal relationship, like "Female"/"Male", 
    we need to use one-hot encoding where each level is assigned its own column, so that no natural ordering can be assumed amongst the 
    levels. Finally, as mentioned before, the random forest model does not require continuous variables (like `MonthlyCharges`) to be scaled, 
    so we can leave as is.
    
    We will start with compiling lists of names for variables that need to be 1) integer encoded, 2) one-hot encoded, or 3) left alone
    by sort them according to data type and number of possible levels:
    '''

    with st.echo():
        binary_list = []
        onehot_list = []
        numeric_list = []

        for i in df.columns:
            if df[i].dtype == 'object' and set(df[i].unique()) == set(['No', 'Yes']):
                binary_list.append(i)
            elif df[i].dtype != 'object':
                numeric_list.append(i)
            else:
                onehot_list.append(i)

    '''
    Categorical variables to be integer encoded:
    '''

    st.write(binary_list)

    '''
    Categorical variables to be one-hot encoded:
    '''

    st.write(onehot_list)

    '''
    Continuous variables:
    '''

    st.write(numeric_list)


    '''
    Next, we will apply the appropriate encoding method to variables in each of the lists:
    '''

    with st.echo():
        df[binary_list] = np.where(df[binary_list] == 'Yes', 1, 0 )

        df = pd.get_dummies(df, columns=onehot_list)


    if st.checkbox("Encode"):
        st.dataframe(df.head(3).T)

        st.write(list(df.columns))

        '''
        Now we see that the categorical variables whose possible values are not just "Yes"/"No" are split into multiple integer 
        encoded columns. This is also helpful for when we are calculating feature importance after fitting the model, to see exactly 
        how each possible value of these variables are associated with predicting customer churn.
        
        Finally, the cleaned and encoded dataset is saved in feather format for easy loading in the future:
        '''

        df = df[["Tenure",  "MonthlyCharges",   "TotalCharges",   "Churn",  "SeniorCitizen", "Partner",
                 "Dependents",  "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies",  "PaperlessBilling", "Gender_Female",  "Gender_Male",
                 "InternetService_DSL",   "InternetService_Fiber optic",  "InternetService_No", "Contract_Month-to-month",
                 "Contract_One year",  "Contract_Two year",  "PaymentMethod_Bank transfer (automatic)",
                 "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",  "PaymentMethod_Mailed check"]]

        with st.echo():
            import feather

            df.to_feather('./clean_data')



if page == pages[4]:
    st.sidebar.markdown('''
    ---
    ''')

    topic = st.sidebar.radio("Outline:",
                     ("1. Original dataset",
                      "2. Random upsampling",
                      "3. SMOTE-NC upsampling",
                      "4. Create new classes by clustering"))

    st.sidebar.markdown('''
    
    ---

    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    Want to learn more?
    
    - Eight approaches to handle imbalanced classes [[Machine Learning Mastery]](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
    - A practical guide to oversampling algorithms [[`imbalance-learn` documentation]](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html)
    
    
    
    
    ''')


    if topic == "1. Original dataset":

        '''
        # 5.1 Split the original dataset
        
        We need to split the whole dataset into three chunks for training, testing and validating the model.
        '''

        with st.echo():
            ## Import libraries
            from sklearn.model_selection import train_test_split
            import pickle

            ## Import data
            df = pd.read_feather('./clean_data')

            ## Shuffle the dataframe in place by taking a random sample with the same size as the original dataset
            df = df.sample(frac=1, replace=False, random_state=1)

            ## 80/20 split into training and testing sets, stratified by the target variable
            X_train, X_val, y_train, y_val = train_test_split(df.drop(['Churn'], axis=1),
                                                                      df['Churn'],
                                                                      test_size = 0.2,
                                                                      stratify=df['Churn'])


            ## Pickle the datasets for later use
            with open('./telco_split_sets.pickle', 'wb') as f:
                pickle.dump([X_train, y_train, X_val, y_val], f)


        data = {'Training (negative class)': y_train.value_counts()[0],
                'Training (positive class)': y_train.value_counts()[1],
                'Validation':X_val.shape[0]}

        fig = plt.figure(
            FigureClass=Waffle,
            rows=7,
            columns=10,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.03), "fontsize":13, 'ncol':3},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(8, 6)
        )

        if st.checkbox("Visualize the datasets"):
            '''
            There are 7,043 customers in the dataset. Each symbol represents ~100 customers.
            '''

            st.pyplot(width=900, height=800)

            plt.clf()

            '''
            Immediately, we see that there are way fewer instances belonging to positive class ("Churn) than in the 
            negative class ("No churn"). This is because only about a third of the ~7,000 customers have churned. This class imbalance 
            is a common problem that would negatively impact the performance of the model, as it simply do not have enough instances of 
            "Churn" to learn from. 
            
            Coming up next, we will try out two upsampling techniques that try to create more instances of the positive class. Then, we will 
            use unsupervised clustering of the entire dataset to see if we can create new *classes* as some sort of 
             proxy for `Churn` to predict, instead of predicting `Churn` itself.
            '''

    if topic == "2. Random upsampling":

        '''
        # 5.2 Address class imbalance: random upsampling of minority class
        
        The simplest approach to upsampling is to randomly draw instances from the minority class with replacement, which is implemented by
        `RandomOverSampler()` from the `imbalanced-learn` package.
        
        As an important note, you **must only upsample the training set**. As naive upsampling duplicates certain data points 
        and SMOTE generates new data points that are similar to existing ones, by splitting *after* upsampling, you are introducing redundancy 
        between the training and test sets. This way, the model will have already "seen" some of the data in the test set, leading to overly optimistic measures of its performance. 

        For a demonstration of what happens if the data set is upsampled before the train-test split, please see a post 
        by Nick Becker [here](https://beckernick.github.io/oversampling-modeling/).
        '''

        with st.echo():
            ## Import libraries
            from imblearn.over_sampling import RandomOverSampler

            ## Import data
            infile = open('./telco_split_sets.pickle','rb')

            X_train, y_train, X_val, y_val = pickle.load(infile)

            ## Naive upsample
            ros = RandomOverSampler(random_state=0)
            X_ros, y_ros = ros.fit_resample(X_train, y_train)

            ## Rename upsampled dataset with original column names
            X_ros = pd.DataFrame(X_ros)

            X_ros.columns = X_train.columns

            ## Reset numerical column data types
            num_list = ['Tenure', 'MonthlyCharges', 'TotalCharges']

            for col in num_list:
                X_ros[col] = X_ros[col].astype('float64')

        with open('./random_split_sets.pickle', 'wb') as f:
            pickle.dump([X_ros, y_ros, X_val, y_val], f)


        x = pd.DataFrame(y_ros)[0].value_counts()

        data = {'Training (positive class)' : x[0],
                'Training (negative class)' : x[1],
                'Validation' : X_val.shape[0]}


        fig = plt.figure(
            FigureClass=Waffle,
            rows=6,
            columns=13,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.15), "fontsize":13, 'ncol':3},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(9, 8)
        )

        if st.checkbox("Upsample minority class"):
            st.markdown('''
            Now we have equal-sized positive and negative classes in the training set, each containing 4,130 instances.
            
            Each symbol represents ~86 customers.
            ''')

            plt.tight_layout()

            st.pyplot(width=900, height=500)

            plt.clf()



    if topic == "3. SMOTE-NC upsampling":

        '''
        # 5.3 Address class imbalance: create new data points in minority class
        
        SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic data points by creating and random choosing k-nearest-neighbours 
        of instances in the minority class. SMOTE-NC is an extension of the method for 
        use with datasets that contain both categorical and continuous variables, like the Telco customer churn dataset.
        
        '''

        '''
        ```Python
        ## Import library
        from imblearn.over_sampling import SMOTENC

        ## Import data
        infile = open('./telco_split_sets.pickle','rb')

        X_train, y_train, X_val, y_val = pickle.load(infile)

        ## I have reordered the columns so that the three continuous
        ## variables are in the first three positions
        cat_range = range(3, 27)

        ## Upsampling using SMOTE-NC
        smote_nc = SMOTENC(categorical_features=cat_range, random_state=0)

        X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
        ```
        '''

        ## Get data from pickled file
        infile = open('./smote_split_sets.pickle','rb')

        X_resampled, y_resampled, X_val, y_val = pickle.load(infile)


        x = pd.DataFrame(y_resampled)[0].value_counts()

        data = {'Training (positive class)' : x[0],
                'Training (negative class)' : x[1],
                'Validation' : X_val.shape[0]}

        fig = plt.figure(
            FigureClass=Waffle,
            rows=6,
            columns=13,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.15), "fontsize":13, 'ncol':3},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(9, 8)
        )

        if st.checkbox("Let there be more data points!"):
            st.markdown('''
            Again, we now have equal-sized positive and negative classes in the training set, each containing 4,130 instances.
               
            Each symbol represents ~86 customers.
            ''')

            plt.tight_layout()

            st.pyplot(width=900, height=800)

            plt.clf()


    if topic == "4. Create new classes by clustering":
        '''
        # 5.4 Create new classes using unsupervised clustering
        
        As an interesting alternative approach, we might try redefining the problem at hand.
        
        From [principal dimensions analysis](http://rpubs.com/nchelaru/famd) done on this data set, I found that the "Churn" and "No Churn" populations 
        of customers are largely overlapping when projected onto the new principal dimension feature space, suggesting that they are not linearly 
        separable. 
        '''

        famd_res = pd.read_csv('./famd_res.csv')

        famd_res.rename(columns={'coord.Dim.1':'Principal dimension 1',
                                'coord.Dim.2':'Principal dimension 2',
                                'coord.Dim.3':'Principal dimension 3'}, inplace=True)

        fig = px.scatter_3d(famd_res, x='Principal dimension 1', y='Principal dimension 2', z='Principal dimension 3',
                            color='Churn')

        fig.for_each_trace(lambda t: t.update(name=t.name.replace("Churn=","")))

        fig.update_traces(marker=dict(size=3, opacity=0.5))

        st.plotly_chart(fig, width=800, height=600)

        '''
        This makes sense for a real world dataset, as customers may leave or stay at various times due interplays amongst a myriad of reasons. 
        In other words, whether a customer churns given his/her personal and buying characteristics is much less deterministic than, for example,
        the species of an iris specimem given its physical dimensions.
        
        Instead, we may try to use **unsupervised** clustering to identify "natural" groupings within this dataset. If this grouping has some correspondence with customer
        churn behaviour, it may be interesting (and potentially more fruitful) to predict this group membership instead. Here we will do so by adapting the workflow for clustering mixed-type
        data by Gower distance (in R) shown [here](https://towardsdatascience.com/clustering-on-mixed-type-data-8bbd0a2569c3).
        '''

        df = pd.read_csv('./gower_res.csv')

        if st.checkbox("Cluster the data"):
            '''
            ```R
            ## Import libraries
            library(cluster)
            library(dendextend)
            library(ggplot2)
            
            raw_df <- read.csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")
            df <- data.frame(raw_df)
            
            ## Scale numeric features
            ind <- sapply(df, is.numeric)
            df[ind] <- lapply(df[ind], scale)
            df[ind] <- lapply(df[ind], as.numeric)
            
            ## Calculate Gower distance
            gower_dist <- daisy(df, metric = "gower")
            gower_mat <- as.matrix(gower_dist)
            
            ## Optimal number of clusters
            sil_width <- c(NA)
    
            for(i in 2:5){  
              pam_fit <- pam(gower_dist, diss = TRUE, k = i)  
              sil_width[i] <- pam_fit$silinfo$avg.width  
            }
            ```
            '''

            opt_clust = pd.read_csv('./opt_clusters.csv')

            v = pd.DataFrame(opt_clust)

            v.columns = ['No. clusters', 'Silhouette width']

            plt.style.use('seaborn-white')
            plt.figure(figsize = (4, 4))

            plt.rcParams.update(
                {'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 14,
                 'legend.title_fontsize': 24, 'legend.loc': 'best'})

            v.plot.line(x='No. clusters', y='Silhouette width')

            st.pyplot()

            plt.clf()

            '''
            This shows that the data can be grouped most optimally in two clusters. Let's look at these two groups as hierarchical clusters:
             
            ```R
            ## Hierarchically cluster based on calculated distance
            fit <- hclust(d=gower_dist, method="complete")  
            
            dend <- fit %>% as.dendrogram %>% hang.dendrogram
            
            dend %>% color_branches(k=2) %>% set("labels", "") %>% plot(horiz=FALSE)
            ```
            '''

            st.image('./dendro.png', use_column_width=True)

            st.markdown('''
            We see that these two clusters are roughly equal in size, which avoids the class imbalance issue.
            
            Now let's see how they differ in terms of customer characteristics and, particularly, **churn**.
            
            ```R
            groups <- cutree(fit, k=2)   # "k=" defines the number of clusters you are using 
            
            new_df <- cbind(raw_df, groups)
            ```
            ''')

            if st.checkbox("Compare these two groups"):
                '''
                First, we can see how these two groups separate in the principal dimension feature space identified by 
                factor analysis:
                '''

                famd_clust = pd.read_csv('./famd_clust.csv')

                famd_clust['groups'] = famd_clust['groups'].astype(str)

                famd_clust.rename(columns={'coord.Dim.1':'Principal dimension 1',
                                         'coord.Dim.2':'Principal dimension 2',
                                         'coord.Dim.3':'Principal dimension 3'}, inplace=True)

                fig = px.scatter_3d(famd_clust, x='Principal dimension 1', y='Principal dimension 2', z='Principal dimension 3',
                                    color='groups').for_each_trace(lambda t: t.update(name=t.name.replace("groups=","Group=")))

                fig.for_each_trace(lambda t: t.update(name=t.name.replace("Churn=","")))

                fig.update_traces(marker=dict(size=3, opacity=0.5))

                st.plotly_chart(fig, width=800, height=600)

                '''
                In comparison to "Churn"/"No churn", customers appear to be much more linearly separable by membership in these two groups, suggesting 
                this alternative approach to be a potentially viable one.
                
                Next, let's see how these two groups of customers differ by their characteristics:
                '''

                ## Reformat columns to contain column name
                col_list = ['SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'DeviceProtection', 'MultipleLines', 'OnlineSecurity',
                            'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

                for col in col_list:
                    df[col] = np.where(df[col]=='Yes', col, 'No'+' '+col)

                ## Plots
                x = df[df['groups'] == 1]

                y = df[df['groups'] == 2]

                def get_values(df, label):
                    o =[]

                    for i in df.columns:
                        if df[i].dtype == object:
                            o.append(df[i].value_counts().to_dict())

                    result = {}

                    for k in o:
                        result.update((k))

                    j = pd.DataFrame()

                    j['Category'] = [key for key, value in result.items() if not 'No' in key]

                    j[label] = [value/df.shape[0]*100 for key, value in result.items() if not 'No' in key]

                    return j

                elec_res = get_values(x, "Group 1")

                non_res = get_values(y, "Group 2")

                final = pd.merge(elec_res, non_res, on='Category', how='inner')

                final['Diff'] = final['Group 1'] - final['Group 2']

                # Reorder it following the values of the first value:
                ordered_df = final.sort_values(by='Diff')
                my_range=range(1,len(final.index)+1)

                plt.style.use('seaborn-white')
                plt.figure(figsize=(8, 10))

                plt.hlines(y=my_range, xmin=ordered_df['Group 1'], xmax=ordered_df['Group 2'], color='grey', alpha=0.4)
                plt.scatter(ordered_df['Group 1'], my_range, color='red', alpha=1, label='Group 1')
                plt.scatter(ordered_df['Group 2'], my_range, color='green', alpha=1, label='Group 2')
                plt.legend(loc='upper right', prop={'size': 14})

                # Add title and axis names
                plt.yticks(my_range, ordered_df['Category'], fontsize=14)
                plt.xlabel('% of customers in group', fontsize=14)

                plt.tight_layout()

                st.pyplot()

                plt.clf()

                '''
                
                So these two groups do differ in terms of churn, with ~15% of customers in "Group 1" and ~40% in "Group 2" leaving the company. Given the 
                differences in some of their personal characteristics and purchasing behaviours that have been associated with different tendencies to churn in 
                previous analyses, like paying by "Electronic check" [sic] and purchasing "Fiber optic" internet, this group membership may be a useful one to predict, 
                in an attempt to separate the "loyal customers" from those who are more likely to be on the fence. While this is not as clear cut a distinction as "Churn"/"No churn", 
                better classification results may actually be better for the company to target the right populations of customers for marketing/retention campaigns.
                
                We should also compare the two groups by the three continuous variables:
                ###
                '''

                n_bins = st.slider("Number of bins",
                                   min_value=10, max_value=50, value=10, step=2)

                for i in ['MonthlyCharges', 'Tenure', 'TotalCharges']:
                    fig = px.histogram(df, x=i, color="groups", opacity=0.4,
                                       color_discrete_sequence = ['red', 'green'], barmode = 'overlay', nbins=n_bins).for_each_trace(lambda t: t.update(name=t.name.replace("groups=","Group=")))

                    fig.update_layout(legend_orientation="h",
                                      legend=dict(x=0, y=1.1),
                                      yaxis=go.layout.YAxis(
                                          title=go.layout.yaxis.Title(
                                              text="Count"
                                          )
                                      ))

                    st.plotly_chart(fig)


                df['groups'] = df['groups'].astype('str')

                df.drop('Churn', axis=1, inplace=True)

                '''
                Looks like the two groups differ the most in terms of `MonthlyCharges`, where the less "loyal" Group 2 paying
                much more on average per month than Group 1.
                
                Finally, we will do the same train-test split to get this dataset ready for modelling:
                '''

                with st.echo():
                    binary_list = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                                   "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                                   "StreamingMovies", "PaperlessBilling"]

                    onehot_list = ["Gender", "InternetService", "Contract", "PaymentMethod"]

                    numeric_list = ["Tenure", "MonthlyCharges", "TotalCharges"]

                    df['groups'] = np.where(df['groups'] == '1', 0, 1)

                    df[binary_list] = np.where(df[binary_list] == 'Yes', 1, 0 )

                    df = pd.get_dummies(df, columns=onehot_list)


                if st.checkbox("Encode and split the data"):
                    st.dataframe(df.head().T)

                    with st.echo():
                        from sklearn.model_selection import train_test_split

                        ## Shuffle the dataframe in place by taking a random sample with the same size as the original dataset
                        df = df.sample(frac=1, replace=False, random_state=1)

                        ## 80/20 split into training and testing sets, stratified by the target variable
                        X_train, X_val, y_train, y_val = train_test_split(df.drop(['groups'], axis=1),
                                                                                  df['groups'],
                                                                                  test_size = 0.2,
                                                                                  stratify=df['groups'])

                    with open('./cluster_split_sets.pickle', 'wb') as f:
                        pickle.dump([X_train, y_train, X_val, y_val], f)

                    data = {'Training (negative class)': y_train.value_counts()[0],
                            'Training (positive class)': y_train.value_counts()[1],
                            'Validation':X_val.shape[0]}

                    fig = plt.figure(
                        FigureClass=Waffle,
                        rows=7,
                        columns=10,
                        values=data,
                        legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.03), "fontsize":13, 'ncol':3},
                        icons='user',
                        font_size=22,
                        icon_legend=True,
                        figsize=(8, 6)
                    )

                    '''
                    ###
                    
                    There are 7,043 customers in the dataset. Each symbol represents ~100 customers.
                    '''

                    st.pyplot(width=900, height=800)

                    plt.clf()



if page == pages[5]:
    st.sidebar.markdown(''' --- ''')

    section = st.sidebar.radio('Navigate',
                     ('1. Hyperparameter tuning',
                      '2. Fit models',
                      '3. Evaluate model performance')
                     )

    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    
    Want to learn more?
    
    - Evaluating machine learning modes - hyperparameter tuning [[Alice Zheng]](https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/5/hyperparameter-tuning)
    - Confusion matrix [[Machine Learning Mastery]](https://machinelearningmastery.com/confusion-matrix-machine-learning/)
    - Classification report   [[`yellowbrick` documentation]](https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html)
    - Precision-Recall   [[`scikit-learn` documentation]](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
    ''')

    if section == '1. Hyperparameter tuning':
        '''
        # 6.1 Hyperparameter tuning
        
        Finding the optimal (to a degree) set of model hyperparameter settings for a particular dataset is key to getting
         the best prediction accuracy out of it. This is done by searching through a predefined hyperparameter space ("grid") for the 
         combination that achieves the best performance. Two most commonly used approaches are grid search and random search, where the former 
         exhaustively searches through every possible combination in the grid, and the latter evaluates only a random sample of points on this 
         grid. [Benchmarking](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)
         shows that random search returns similar parameters as grid search but with much lower run time, so we will be using 
         this method here:
        
        ```Python
        ## Import libraries
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier
    
        # Utility function to report best scores
        def report(results, n_top=1):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    return results['mean_test_score'][candidate], results['std_test_score'][candidate], results['params'][candidate]
                
        ## Specify parameter space
        param_dist = {"n_estimators": [600, 800, 1000],
                      "max_depth": sp_randint(1, 7),
                      'max_features': ['auto', 'sqrt'],
                      "min_samples_split": sp_randint(2, 11),
                      'min_samples_leaf': [1, 2, 4],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        
        # Instantiate a random forest classifier
        clf = RandomForestClassifier()
    
        # Randomized search
        random_search = RandomizedSearchCV(clf, 
                                           param_distributions=param_dist, 
                                           n_iter=20, 
                                           cv=5, 
                                           iid=False)
                                           
        random_search.fit(X_train, y_train)
    
        ## Get parameters
        mean_test_score, std_test_score, params_list = report(random_search.cv_results_, n_top=1)
        
        bootstrap, class_weight, criterion, max_depth, max_features, min_samples_split, n_estimators = params_list.values()
    
        ```
        '''


        if st.checkbox("Tune parameters"):
            '''
            We performed random search in the parameter grid specified for all four of the datasets that we have, getting a 
            set of parameters for each that we will be using to train the model below.
      
            '''

            infile = open('./para_df.pickle','rb')

            para_df = pickle.load(infile)

            st.dataframe(para_df.T)

    if section == '2. Fit models':
        '''
        # 6.2 Fit model to training set
        
        For each of the four datasets, we will first fit the `RandomForestClassifier` object on the training set, with the corresponding 
        hyperparameters found above (`params_list`) to the training set. Then, to evaluate performance of the fitted model, it will be 
        used to make predictions on the validation set. The `score` calculated here is the classification accuracy of the model
         on a given dataset averaged across the two classes:
        
        ```Python
        ## Define function for calculating scores
        def score(m):
            res = {"Score on training set" : m.score(X_train, y_train),
                   "Score on validation set" : m.score(X_val, y_val)}
            return res
        
        ## Create model object with parameters from random search
        m = RandomForestClassifier(**params_list)

        ## Fit model to training set
        m.fit(X_train, y_train)
    
        ## Calculate score    
        score(m)
        ```
        '''

        if st.checkbox("Fit and evaluate the model:"):
            with st.spinner("Working on it..."):
                infile = open('./scores_df.pickle', 'rb')

                scores = pickle.load(infile)

                plt.style.use('seaborn-white')

                plt.rcParams.update({'axes.labelpad': 15, 'axes.labelsize': 18,
                                     'xtick.labelsize': 14, 'ytick.labelsize': 18,
                                     'legend.loc': 'best', 'legend.fontsize': 18})

                scores.plot.barh(figsize=(8, 6), fontsize=14)

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=14)

                plt.xlabel('Classification accuracy (%)')

                plt.tight_layout()

                st.pyplot()

                plt.clf()

                '''
                Judging by the scores, the two models trained on upsampled training sets show some degrees of overfitting, as 
                seen by lower classification accuracy on the validation set than on the training set. The model trained to 
                predict grouping identified by the unsupervised clustering show the best performance (>90% accuracy) and
                 least overfitting.
                '''

    if section == '3. Evaluate model performance':
        '''
        # 6.3 Evaluate model performance
        
        ## 6.3.1 Confusion matrix
        
        What classification accuracy does not tell us is how well the model makes predictions for each of the two classes. This is particularly
         problematic since the "Churn"/"No churn" classes are very imbalanced in our validation set. This is where the confusion matrix comes to the rescue, 
         providing actual numbers of correct and incorrect predictions made for each class:
        '''

        if st.checkbox("Plot confusion matrices"):
            infile = open('./cmatrix_dict.pickle', 'rb')

            cmatrix_dict = pickle.load(infile)

            names = ['Original', 'Random', 'SMOTE', 'Clusters']

            fig = plt.figure(figsize=(10, 8))
            fig.subplots_adjust(hspace=0.4, wspace=0.3)

            sns.set_context("talk",
                            rc={'xtick.labelsize': 14,
                                'ytick.labelsize': 11,
                                'xtick.major.width': 0,
                                'ytick.major.width': 0})

            for i, c in zip(range(1, 5), names):
                ax = fig.add_subplot(2, 2, i)
                sns.heatmap(cmatrix_dict[c], annot=True, ax=ax, annot_kws={"size": 14}, fmt='d', cbar=False,
                            vmin=0, vmax=950)
                ax.set_title(c, fontsize=16)

                if i !=4 :
                    ax.set_xticklabels(['No Churn (predicted)', 'Churn (predicted)'])
                    ax.set_yticklabels(['No Churn (actual)', 'Churn (actual)'], va="center")
                else:
                    ax.set_xticklabels(['Group 1 (predicted)', 'Group 2 (predicted)'])
                    ax.set_yticklabels(['Group 1 (actual)', 'Group 2 (actual)'], va="center")


            plt.tight_layout()

            st.pyplot(width=800, height=1000)

            plt.clf()

            '''
            A model that makes more correct predictions than not in both classes will have higher numbers of instances 
            in the diagonal going from the upper left to the bottom right. The model predicting membership of the two groups
            identified using unsupervised clustering appears to be the highest performing model out of the four, with the other three
            tending to produce false positives for "Churn" (upper right quadrant), particularly with the two models trained on upsampled 
            training sets.
            '''

            '''
            ## 6.3.2 Classification report
            
            The classification report calculates several metrics from the raw data in the confusion matrix, 
            making it easier to compare performance between models:
            
            - Precision
            > What percentage of positive predictions made by the model were actually correct?
            
            - Recall
            > What percentage of positive instances were correctly predicted by the model?
            
            - f1 score
            > A weighted harmonic mean of precision and recall, with 0 being the worst, 1 being the best
            
            '''

            if st.checkbox("Check classification reports"):
                infile = open('./crep_dict.pickle', 'rb')

                crep_dict = pickle.load(infile)

                names = ['Original', 'Random', 'SMOTE', 'Clusters']

                sns.set_context("talk",
                                rc={'xtick.labelsize':14,
                                    'ytick.labelsize':14,
                                    'xtick.major.width':0,
                                    'ytick.major.width':0})

                fig = plt.figure(figsize=(10, 8))
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                for i, c in zip(range(1, 5), names):
                    df = pd.DataFrame(crep_dict[c]).T
                    ax = fig.add_subplot(2, 2, i)

                    sns.heatmap(df.drop('support', axis=1), annot=True, ax=ax, annot_kws={"size": 14},
                                cmap="YlGnBu", cbar=False, vmin=0, vmax=1)

                    ax.set_title(c, fontsize=16)


                    if i == 4:
                        ax.set_yticklabels(['Group 1', 'Group2', 'Accuracy', 'Macro avg', 'Weighted avg'], va="center")
                    else:
                        ax.set_yticklabels(['No churn', 'Churn', 'Accuracy', 'Macro avg', 'Weighted avg'], va="center")


                plt.tight_layout()

                st.pyplot(width=800, height=1000)

                plt.clf()

                '''
                Rule of thumb for interpreting classification reports:
                - A model with high precision but low recall tends to produce **false negatives**
                - A model with low precision but hight recall tends to produce **false positives**
                - The weighted average of f1 score should be used to compare classifier models
                
                As we are more interested in predicting churn, let's look at the performance of each model on that class (second row of each heatmap). 
                We can see that the model trained on the imbalanced training set tends to produce false negatives, namely failing to identify customers who churn. 
                The two models trained on the upsampled datasets show the opposite, prone to false positives, potentially due to the artificial inflation of the 
                "Churn" class. This can be a trade-off when used in real-world applications, depending on whether failing to prevent customer churn or enacting unnecessary 
                customer retention measures is more costly for the company. Judging by the f1 scores, which take into account both the precision and recall, 
                the model trained on the original imbalanced dataset performs just as well as these two models.
                
                Finally, the model trained to predict membership in the two groups identified by unsupervised clustering appear to perform the best. Recall ~15% of
                 customers in "Group 1" have churned as compared to ~40% of customers in "Group 2". Therefore, when it comes to making predictions on new customers, 
                 it may be more fruitful to try to predict whether they are **more or less likely** to churn, trading the certainty of a definite label 
                 for more robust predictions.
                '''



if page == pages[6]:
    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    Want to learn more about permutation feature importance?
    
    - Interpretable Machine Learning [[Christopher Molnar]](https://christophm.github.io/interpretable-ml-book/)
    - Permutation feature importance for random forest models [[explained.ai]](https://explained.ai/rf-importance/)
    
    ''')

    '''
    Arguably the most exciting part of machine learning is exploring
    the "decision making process" of the model. Understanding the relative importance that the model attributes to various variables not
    only provides insights into novel relationships in the data, it also allows real-world domain knowledge to enter the picture and 
    check the validity of the predictions made. The many perils of treating machine learning models as black boxes are garnering growing attention 
    as we grow to rely on them in all aspects of our lives. For a great introduction to the topic, check out the ebook by Christopher Molnar as linked in the sidebar. 
    
    ###
    '''

    st.image('./feature_importance.png', caption='Image credit: Icons 8', use_column_width=True)

    '''
    ###
    
    One reasonably efficient and reliable technique for calculating feature importance for all models is **permutation importance**, which directly measures
    variable importance by observing the effect of scrambling the values of each predictor variable on model accuracy. Note that variables with *negative*
     feature importance as calculated here means that removing them will improve model performance. 
    
     
    ```Python
    ## Import libraries
    from sklearn.ensemble import RandomForestClassifier
    from rfpimp import *
    
    ## Instantiate model
    m = RandomForestClassifier(**params_list)

    ## Fit model to training set
    m.fit(X_train, y_train)

    ## Calculate permutation feature importance on test set
    imp = importances(m, X_test, y_test, n_samples=-1)
    ```
    '''


    infile = open('./imp_dict.pickle', 'rb')
    imp_dict = pickle.load(infile)

    if st.checkbox("Calculate permutation feature importance"):
        with st.spinner('Hang on tight...'):
            '''
            First, we compare the permutated feature importance calculated for each of the three models trained to predict "Churn"/"No churn" from the 
            original ('Original'), randomly upsampled ('Random') and SMOTE-NC upsampled ('SMOTE-NC') datasets. Interestingly, the three models are fairly
             consistent in treating `InternetService_Fiber optic` and `InternetService_No` as 
            the most important features for prediction, and `Contract_Two year` and `TotalCharges` as the least. 
            
            Taking into account my earlier analyses of this dataset using approaches like [factor analysis of mixed data (FAMD)](http://rpubs.com/nchelaru/famd), 
            [association rule mining](https://nancy-chelaru-centea.shinyapps.io/assn_rules_mining/) and [survival analysis](https://survival-analysis.herokuapp.com/), 
            purchasing fiber optic internet service indeed is associated with leaving the company. It is also understandable that `TotalCharges` is not needed as a predictor variable 
            when building the classifier, as it is the product of `MonthlyCharges` and `Tenure`, so its information is redundant. What is surprising is `Contract_Two year` being deemed
            a variable that should be dropped, as it is shown to be a characteristic of "loyal" customers in the other analyses. It is possible that in dummy encoding the `Contract` variable, 
            a customer can be interpreted as having a two-year contract if he/she is negative for `Contract_Month-to-month` and `Contract_One year`. It appears to be debatable whether a "reference"
            column should be dropped, like `Contract_Two year`, when dummy encoding, so for the sake of demonstration I have opted to leave it in for this example.
            '''

            imp_list = []

            for i in ['Original', 'Random', 'SMOTE']:
                df = imp_dict[i]
                imp_list.append(df)

            imp_df = pd.concat(imp_list, axis=1, sort=True)

            imp_df.columns = ['Original', 'Random', 'SMOTE-NC']

            imp_df = imp_df.sort_values(by='Original')

            imp_df['Mean'] = imp_df.mean(axis=1)

            imp_df = imp_df.sort_values(by='Mean')

            plt.style.use('seaborn-white')

            plt.rcParams.update(
                {'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 18,
                 'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 18,
                 'figure.figsize': (10, 10)})

            imp_df.drop('Mean', axis=1).plot.barh()

            plt.xlabel('Permutated feature importance')
            plt.ylabel('')

            plt.tight_layout()

            st.pyplot()

            plt.clf()

            '''
            As for the model trained to predict membership in the two groups ("more likely to churn" and "less likely to churn") identified by 
            unsupervised clustering, it also ranked `InternetService_Fiber optic` as the most important feature for prediction. This is consistent 
            with our earlier comparison of customer characteristics in these two groups, showing that the two groups differ *the most* in the percentage 
            of customers who have purchased fiber optic internet.  
            '''

            plt.style.use('seaborn-white')

            plt.rcParams.update(
                {'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 18,
                 'legend.title_fontsize': 24, 'legend.loc': 'best', 'legend.fontsize': 18,
                 'figure.figsize': (10, 10)})

            imp_dict['Clusters'].sort_values(by='Clusters').plot.barh()

            plt.xlabel('Permutated feature importance')
            plt.ylabel('')

            plt.tight_layout()

            st.pyplot()

            plt.clf()

            '''
            We can see that these four models provide both corroborating and differing results when it comes to feature importance. 
            It is important to examine them together, as approach a data science problem from multiple angles and aggregating 
            the results are key to gaining nuanced and robust insights to the data.  
            '''



if page == pages[7]:

    st.balloons()

    '''
    ### Congratulations! That was a *long* journey and you have made it to the end!
    
    ###
    '''

    st.image('./celebrate.png', use_column_width=True, caption='Image credit: Icons 8')

    st.sidebar.markdown(
    '''
    ---
    
    [Source code](https://github.com/nchelaru/random-forest-streamlit) for this app.
    
    For other data science/web development projects that I've cooked up, please head over to my portfolio, [The Perennial Beginner](http://nancychelaru.rbind.io/).
    ''')

    '''
    ###
    
    Hopefully at this point, you are a bit more familiar with a typical workflow for applying machine learning to a business problem. 
    The steps shown here are by no means exhaustive, and are fairly simplified for demonstration purposes. 
    
    The material presented here are a summary of my own learning so far, much of which have been done using these fantastic resources:
    
    - The very popular [fast.ai](https://www.fast.ai/) courses on machine learning and deep learning
    - [Machine Learning Mastery](https://machinelearningmastery.com/), a wealth of gentle yet detailed machine learning tutorials and resources created by Jason Brownlee, PhD
    - [Real-World Machine Learning](https://www.manning.com/books/real-world-machine-learning) by Henrik Brink, Joseph W. Richards, and Mark Fetherolf
    
    I will update (and correct) the contents on this site as I continue to explore this ever expanding field, so please check back once in a while!
    
    Hope you have enjoyed your stay! :)
    
    '''











