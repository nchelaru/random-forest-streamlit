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


# Set outline
pages = ["1. Introduction",
         "2. Data cleaning",
         "3. Explore the dataset",
         "4. Variable encoding",
         "5. Create training, validation and test sets",
         "6. Parameter tuning and model fitting",
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

    st.title('1. Introduction')

    '''
    Of the myriad of supervised machine learning algorithms available, the random forest model is well-favoured for its robust 
    out-of-box performance on datasets containing diverse data types. As it has very few statistical assumptions and require little feature
     engineering, it is a great option as the first model to fit onto a new dataset to explore how much "learnable" signal can found before attempting
      more sophisticated methods.
      
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
    
    Here, we will walk through the steps of a full machine learning workflow, from data preprocessing to calculating 
    variable importance. Head over to the dropdown menu in the sidebar to get started!
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
                      '6. Combine sparse levels'))


    st.sidebar.markdown('''
    ---
    
    Feeling overwhelmed by all the new info coming your way?

    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    ''')


    if section == '1. Preview data':
        st.title('1. Preview data')

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
        st.title("2. Remove customer ID column")

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
        st.title("3. Re-encode variable levels")

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
        st.title("4. Rename column headers")

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
        st.title("5. Correct column data type")

        '''
        Setting the correct data type for each column in a `pandas` dataframe is pretty important for the data to be treated in the 
        "correct manner" in the preprocessing and model fitting process.  
        '''

        st.header("5.1 Set `TotalCharges` as numeric type")

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

                    st.header('5.2 Set categorical variables as "category" type')

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





    if section == '6. Combine sparse levels':
        st.title("6. Combine sparse levels")

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

    if var1 == ' ' and var2 == ' ':
        '''
        Click on any of the categorical variable names to expand the
         sunburt chart and see distribution of the levels in more detail, where the size of each leaf is proportional 
         to the number of customers in that level.
        '''

        with st.spinner('Working on it...'):

            fig = sunburst_fig()

            st.plotly_chart(fig, width=700, height=700)
    elif var1 != ' ' and var2 == ' ' and df[var1].dtype == 'object':
        '''
        There are 7,032 customers in the dataset. Each symbol represents ~100 customers.
        '''

        with st.spinner('Working on it...'):
            data = df[var1].value_counts().to_dict()

            fig = plt.figure(
                FigureClass=Waffle,
                rows=5,
                columns=14,
                values=data,
                legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.2), "fontsize":16, 'ncol':2},
                icons='user',
                font_size=38,
                icon_legend=True,
                figsize=(12, 8)
            )

            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif var1 != ' ' and var2 == ' ' and df[var1].dtype != 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, nbins=n_bins)

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
            fig = sns.countplot(x=var1, hue=var2, data=df, palette="Set3")

            plt.ylabel('Count')

            sns.set(style="ticks", font_scale=2.2, rc={'figure.figsize':(18, 11)})

            plt.grid(False)

            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype == 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=var2, opacity=0.4, barmode = 'overlay', nbins=n_bins)

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
            fig = sns.barplot(x=var1, y=var2, data=df, palette="Set3")

            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype != 'object' and var1 == var2:
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=None, nbins=n_bins)

            fig.update_layout(legend_orientation="h",
                              legend=dict(x=0, y=1.1),
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              ))

            st.plotly_chart(fig)
    elif var1 != var2 and df[var1].dtype != 'object' and df[var2].dtype != 'object':
        with st.spinner('Working on it...'):
            sns.jointplot(df[var1], df[var2], kind="hex", color="#4CB391")

            sns.set(style="ticks", font_scale=1.1, rc={'figure.figsize':(12, 6)})

            plt.tight_layout()

            st.pyplot()

            plt.clf()

    else:
        pass



# 3. Variable encoding
if page == pages[3]:
    st.title("4. Variable encoding")

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

        st.header("5.1 Split the original dataset")

        '''
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
            X_train, X_testing, y_train, y_testing = train_test_split(df.drop(['Churn'], axis=1),
                                                                      df['Churn'],
                                                                      test_size = 0.2,
                                                                      stratify=df['Churn'])

            ## Just in case it's needed later, also split testing set into test and validation sets
            X_test, X_val, y_test, y_val = train_test_split(X_testing,
                                                            y_testing,
                                                            test_size = 0.4,
                                                            stratify= y_testing)

            ## Pickle the datasets for later use
            with open('./telco_split_sets.pickle', 'wb') as f:
                pickle.dump([X_train, y_train, X_test, y_test, X_val, y_val], f)



        data = {'Training (negative class)': y_train.value_counts()[0],
                'Training (positive class)': y_train.value_counts()[1],
                'Test': X_test.shape[0],
                'Validation':X_val.shape[0]}

        fig = plt.figure(
            FigureClass=Waffle,
            rows=7,
            columns=10,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.03), "fontsize":16, 'ncol':2},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(8, 6)
        )

        if st.checkbox("Visualize the datasets"):
            st.pyplot(width=900, height=800)

            plt.clf()

            '''
            Immediately, we see that there are way fewer instances belonging to positive class ("Churn) than in the 
            negative class ("No churn"). This is because only about a third of the ~7,000c customers have churned. This class imbalance 
            is a common problem that would negatively impact the performance of the model, as it simply do not have enough instances of 
            "Churn" to learn from. 
            
            Coming up next, we will try out two upsampling techniques that try to create more instances of the positive class. Then, we will 
            use unsupervised clustering of the entire dataset to see if we can create new *classes* as some sort of 
             proxy for `Churn` to predict, instead of predicting `Churn` itself.
            '''

    if topic == "2. Random upsampling":

        '''
        ## 5.2 Random upsampling
        
        The simplest approach to upsampling is to randomly draw instances from the minority class with replacement, which is implemented by
        `RandomOverSampler()` from the `imbalanced-learn` package.
        
        As an important note, you **must only upsample the training set**. As naive upsampling duplicates certain data points 
        and SMOTE generates new data points that are similar to existing ones, by splitting *after* upsampling, you are introducing redundancy 
        between the training and test sets. This way, the model will have already "seen" some of the data in the test set, leading to overly optimistic measures of its performance. 

        For a demonstration of what happens if the data set is upsampled before the train-test split, please see a post 
        by Nick Becker [here](https://beckernick.github.io/oversampling-modeling/).
 
        ```Python
        ## Import libraries
        from imblearn.over_sampling import RandomOverSampler

        ## Import data
        infile = open('./telco_split_sets.pickle','rb')

        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

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
            pickle.dump([X_ros, y_ros, X_test, y_test, X_val, y_val], f)
        ```
        '''

        infile = open('./random_split_sets.pickle','rb')

        X_ros, y_ros, X_test, y_test, X_val, y_val = pickle.load(infile)

        x = pd.DataFrame(y_ros)[0].value_counts()

        data = {'Training (positive class)' : x[0],
                'Training (negative class)' : x[1],
                'Test' : X_test.shape[0],
                'Validation' : X_val.shape[0]}

        fig = plt.figure(
            FigureClass=Waffle,
            rows=6,
            columns=13,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.15), "fontsize":16, 'ncol':2},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(8, 6)
        )

        '''
        ##
        '''

        if st.checkbox("Upsample minority class"):
            st.markdown('''
            Now we have equal-sized positive and negative classes in the training set, each containing 4,130 instances.
            
            Each symbol represents ~86 customers.
            ''')

            plt.tight_layout()

            st.pyplot(width=900, height=700)

            plt.clf()



    if topic == "3. SMOTE-NC upsampling":
        st.header("5.3 Create synthetic data")

        '''
        SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic data points by creating and random choosing k-nearest-neighbours 
        of instances in the minority class. SMOTE-NC is an extension of the method for 
        use with datasets that contain both categorical and continuous variables, like the Telco customer churn dataset.
        
        ```Python
        ## Import library
        from imblearn.over_sampling import SMOTENC

        ## Import data
        infile = open('./telco_split_sets.pickle','rb')

        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

        ## I have reordered the columns so that the three continuous
        ## variables are in the first three positions
        cat_range = range(3, 27)

        ## Upsampling using SMOTE-NC
        smote_nc = SMOTENC(categorical_features=cat_range, random_state=0)

        X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

        ## Save for future use
        with open('./smote_split_sets.pickle', 'wb') as f:
            pickle.dump([X_resampled, y_resampled, X_test, y_test, X_val, y_val], f)
        ```
        '''

        infile = open('./smote_split_sets.pickle','rb')

        X_resampled, y_resampled, X_test, y_test, X_val, y_val = pickle.load(infile)

        x = pd.DataFrame(y_resampled)[0].value_counts()

        data = {'Training (positive class)' : x[0],
                'Training (negative class)' : x[1],
                'Test' : X_test.shape[0],
                'Validation' : X_val.shape[0]}

        fig = plt.figure(
            FigureClass=Waffle,
            rows=6,
            columns=13,
            values=data,
            legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.15), "fontsize":16, 'ncol':2},
            icons='user',
            font_size=22,
            icon_legend=True,
            figsize=(8, 6)
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
        st.header('5.4 Unsupervised clustering')

        '''
        As an interesting alternative approach, we might try redefining the problem at hand.
        
        From [principal dimensions analysis](http://rpubs.com/nchelaru/famd) done on this data set, I found that the "Churn" and "No Churn" populations 
        of customers are largely overlapping when projected onto the new principal dimension feature space, suggesting that they are not linearly 
        separable. 
        '''

        famd_res = pd.read_csv('./famd_res.csv')

        fig = px.scatter_3d(famd_res, x='coord.Dim.1', y='coord.Dim.2', z='coord.Dim.3',
                            color='Churn')

        fig.for_each_trace(lambda t: t.update(name=t.name.replace("Churn=","")))

        fig.update_traces(marker=dict(size=3, opacity=0.5))

        st.plotly_chart(fig, width=800, height=600)

        '''
        This makes sense for a real world dataset, as customers may leave or stay at various times due interplays amongst a myriad of reasons. 
        In other words, whether a customer churns given his/her personal and buying characteristics is much less deterministic than the species of an iris specimen
        given its physical dimensions.
        
        Instead, we may try to use unsupervised clustering to identify "natural" groupings within this dataset. If this grouping has some correspondence with customer
        churn behaviour, it may be interesting (and potentially more fruitful) to predict this group membership instead. Here we will do so by adapting the workflow for clustering mixed-type
        data (in R) shown [here](https://towardsdatascience.com/clustering-on-mixed-type-data-8bbd0a2569c3).
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

            v.columns = ['No. clusters', 'Value']

            v.plot.line(x='No. clusters', y='Value')

            st.pyplot()

            plt.clf()

            '''
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

                plt.style.use('default')
                plt.rcParams.update({'figure.figsize':[10, 8], 'font.size':18})

                plt.hlines(y=my_range, xmin=ordered_df['Group 1'], xmax=ordered_df['Group 2'], color='grey', alpha=0.4)
                plt.scatter(ordered_df['Group 1'], my_range, color='red', alpha=1, label='Group 1')
                plt.scatter(ordered_df['Group 2'], my_range, color='green', alpha=1, label='Group 2')
                plt.legend(loc='upper right', prop={'size': 16})

                # Add title and axis names
                plt.yticks(my_range, ordered_df['Category'])
                plt.xlabel('% of customers in group ')

                plt.tight_layout()

                st.pyplot()

                plt.clf()

                n_bins = st.slider("Number of bins",
                                   min_value=10, max_value=50, value=10, step=2)

                for i in ['MonthlyCharges', 'Tenure', 'TotalCharges']:
                    fig = px.histogram(df, x=i, color="groups", opacity=0.4,
                                       color_discrete_sequence = ['red', 'green'], barmode = 'overlay', nbins=n_bins)

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
                        X_train, X_testing, y_train, y_testing = train_test_split(df.drop(['groups'], axis=1),
                                                                                  df['groups'],
                                                                                  test_size = 0.2,
                                                                                  stratify=df['groups'])

                        ## Just in case it's needed later, also split testing set into test and validation sets
                        X_test, X_val, y_test, y_val = train_test_split(X_testing,
                                                                        y_testing,
                                                                        test_size = 0.4,
                                                                        stratify= y_testing)



                    with open('./cluster_split_sets.pickle', 'wb') as f:
                        pickle.dump([X_train, y_train, X_test, y_test, X_val, y_val], f)

                    data = {'Training (negative class)': y_train.value_counts()[0], 'Training (positive class)': y_train.value_counts()[1],
                            'Test': X_test.shape[0], 'Validation':X_val.shape[0]}

                    fig = plt.figure(
                        FigureClass=Waffle,
                        rows=7,
                        columns=10,
                        values=data,
                        legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.03), "fontsize":14, 'ncol':2},
                        icons='user',
                        font_size=22,
                        icon_legend=True,
                        figsize=(8, 6)
                    )

                    st.pyplot(width=900, height=800)

                    plt.clf()



if page == pages[5]:
    st.title("6. Parameter tuning and model fitting")

    st.sidebar.markdown('''
    
    ---
    
    Want to learn more?
    
    - Evaluating machine learning modes - hyperparameter tuning [[Alice Zheng]](https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/5/hyperparameter-tuning)
    
    ''')

    st.header("6.1 Hyperparameter tuning")

    '''
    Finding the optimal (to a degree) set of model hyperparameter settings for a particular dataset is key to getting
     the best prediction accuracy out of it. This is done by searching through a predefined hyperparameter space ("grid") for the 
     combination that achieves the best performance. Two most commonly used approaches are grid search and random search, where the former 
     exhaustively searches through every possible combination in the grid, and the latter evaluates only a random sample of points on this 
     grid. [Benchmarking](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)
     shows that random search returns similar parameters as grid search but with much lower run time, so we will be using 
     this method here:
    
    ```Python
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.datasets import load_digits
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
    
    # build a classifier
    clf = RandomForestClassifier()

    # run randomized search
    random_search = RandomizedSearchCV(clf, 
                                       param_distributions=param_dist, 
                                       n_iter=20, 
                                       cv=5, 
                                       iid=False)
                                       
    random_search.fit(X_train, y_train)

    ## Get values
    mean_test_score, std_test_score, params_list = report(random_search.cv_results_, n_top=1)
    
    bootstrap, class_weight, criterion, max_depth, max_features, min_samples_split, n_estimators = params_list.values()

    ```
    '''


    if st.checkbox("Tune parameters"):
        infile = open('./para_df.pickle','rb')

        para_df = pickle.load(infile)

        st.dataframe(para_df.T)

        st.header("6.2 Train models")

        '''
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

        if st.checkbox("Check scores:"):
            with st.spinner("Hang on tight, this takes a bit..."):
                infile = open('./scores_df.pickle', 'rb')

                scores = pickle.load(infile)

                scores.plot.barh(figsize=(8, 6), fontsize=14)

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=12)

                plt.tight_layout()

                st.pyplot()

                plt.clf()


                st.header("6.3 Evaluate model performance")

                '''
                Confusion matrix
                '''

                if st.checkbox("Plot confusion matrix"):
                    infile = open('./cmatrix_dict.pickle', 'rb')

                    cmatrix_dict = pickle.load(infile)

                    names = ['Original', 'Random', 'SMOTE', 'Clusters']

                    fig = plt.figure()
                    fig.subplots_adjust(hspace=0.3, wspace=0.3)

                    sns.set(font_scale = 1.5)

                    for i, c in zip(range(1, 5), names):
                        ax = fig.add_subplot(2, 2, i)
                        sns.heatmap(cmatrix_dict[c], annot=True, ax=ax, annot_kws={"size": 18}, fmt='d', cbar=False,
                                    vmin=0, vmax=600)
                        ax.set_title(c, fontsize=22)
                        if i !=4 :
                            ax.set_xticklabels(['No Churn', 'Churn'])
                            ax.set_yticklabels(['No Churn', 'Churn'], va="center")
                        else:
                            ax.set_xticklabels(['Group 1', 'Group 2'])
                            ax.set_yticklabels(['Group 1', 'Group 2'], va="center")

                    plt.tight_layout()

                    st.pyplot(width=800, height=1000)

                    plt.clf()


                    '''
                    Classification report
                    '''

                    if st.checkbox("Check classification report"):
                        infile = open('./crep_dict.pickle', 'rb')

                        crep_dict = pickle.load(infile)

                        names = ['Original', 'Random', 'SMOTE', 'Clusters']

                        fig = plt.figure()
                        fig.subplots_adjust(hspace=0.4, wspace=0.4)

                        sns.set(font_scale = 1.5)

                        for i, c in zip(range(1, 5), names):
                            df = pd.DataFrame(crep_dict[c]).T.iloc[:2]
                            ax = fig.add_subplot(2, 2, i)

                            sns.heatmap(df.drop('support', axis=1), annot=True, ax=ax, annot_kws={"size": 22},
                                        cmap="YlGnBu", cbar=False, vmin=0, vmax=1)

                            ax.set_title(c, fontsize=26)

                            if i == 4:
                                ax.set_yticklabels(['Group 1', 'Group2'], va="center")
                            else:
                                ax.set_yticklabels(['No churn', 'Churn'], va="center")


                        plt.tight_layout()

                        st.pyplot(width=800, height=1000)

                        plt.clf()






if page == pages[6]:
    st.sidebar.markdown('''
    
    ---

    
    Want to learn more about model interpretability?
    
    - Interpretable Machine Learning - A Guide for Making Black Box Models Explainable 
    [[Christoph Molnar]](https://christophm.github.io/interpretable-ml-book/)
    - Permutation feature importance for random forest models [[fast.ai]](https://explained.ai/rf-importance/)
    
    ''')

    '''
    ```Python
    m = RandomForestClassifier(**params_list)

    m.fit(X_train, y_train)

    imp = importances(m, X_test, y_test, n_samples=-1)
    ```
    '''


    infile = open('./imp_dict.pickle', 'rb')
    imp_dict = pickle.load(infile)

    imp_list = []

    for i in ['Original', 'Random', 'SMOTE', 'Clusters']:
        df = imp_dict[i]
        imp_list.append(df)

    imp_df = pd.concat(imp_list, axis=1, sort=True)

    imp_df.columns = ['Original', 'Oversampled', 'SMOTE-NC', 'Clustered']

    sns.set(style="ticks", font_scale=1.2, rc={'figure.figsize':(18, 11)})

    g = sns.clustermap(imp_df,  standard_scale=1, figsize=(8, 9))

    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

    plt.subplots_adjust(left=0.05, right=0.54, top=0.98, bottom=0.2)

    st.pyplot(width=900, height=900)

    plt.clf()



if page == pages[7]:
    '''
    Check out these resources
    
    '''











