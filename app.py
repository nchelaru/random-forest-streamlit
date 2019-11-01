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
from yellowbrick.classifier import classification_report
from yellowbrick.classifier import confusion_matrix
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

        st.markdown('''
        Finally, some categorical variables in this dataset have levels that can be grouped together, so as to reduce cardinality.
        
        We can look at the levels of each categorical variable by plotting:
        ''')

        with st.echo():
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


        if st.checkbox("Abracadabra!"):
            with st.spinner('Working on it...'):

                plt.tight_layout()

                fig.delaxes(axes[8][1])

                st.pyplot()

                plt.clf()

            st.info('''
            For many of the variables related to online services and phone service, "No internet service" or "No phone service" 
            can be combined into the "No" category for that variable, as that information is already encoded in "No" for 
            `InternetService` and `PhoneService`.
             
             This helps to reduce noise in the dataset.
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

                    plt.cla()

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

    else:
        pass



# 3. Variable encoding
if page == pages[3]:
    st.title("4. Variable encoding")

    st.sidebar.markdown('''
    ---
    
    More on the topic:
    ''')

    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    '''
    ### One-hot encoding
    
    
    ### Binary encoding
    
    ### Numeric variables
    Random forest does not need numeric variables to be scaled
    
    ###
    
    We will start with compiling lists of names for variables that need to be one-hot or binary encoded, or not not at all.
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
    Categorical variables to be binary encoded:
    '''

    st.write(binary_list)

    '''
    Categorical variables to be one-hot encoded:
    '''

    st.write(onehot_list)

    '''
    Numerical variables:
    '''

    st.write(numeric_list)


    '''
    Next, we will apply the appropriate encoding method to variables in each of the lists:
    '''

    with st.echo():
        df[binary_list] = np.where(df[binary_list] == 'Yes', 1, 0 )

        df = pd.get_dummies(df, columns=onehot_list)


    if st.checkbox("Encode"):
        st.dataframe(df.head(5).T)

        '''
        Finally, the cleaned and encoded dataset is saved in feather format for easy loading for the next steps:
        '''

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


    if topic == "1. Original dataset":
        df = pd.read_feather('./clean_data')

        df = df[[
            "Tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PaperlessBilling",
            "Gender_Female",
            "Gender_Male",
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No",
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check"]]

        st.dataframe(df)

        with st.echo():
            from sklearn.model_selection import train_test_split

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



        with open('./telco_split_sets.pickle', 'wb') as f:
            pickle.dump([X_train, y_train, X_test, y_test, X_val, y_val], f)

        data = {'X train (negative class)': y_train.value_counts()[0], 'X train (positive class)': y_train.value_counts()[1],
                 'X test': X_test.shape[0], 'X val':X_val.shape[0]}

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

        if st.checkbox("Waffle"):
            st.pyplot(width=900, height=800)

    if topic == "2. Random upsampling":
        infile = open('./telco_split_sets.pickle','rb')

        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

        with st.echo():
            ## Import libraries
            from imblearn.over_sampling import RandomOverSampler
            from collections import Counter

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

        x = pd.DataFrame(y_ros)[0].value_counts()

        data = {'Training set (positive class)' : x[0],
                'Training set  (negative class)' : x[1],
                'Test set' : X_test.shape[0],
                'Validation set' : X_val.shape[0]}

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

        if st.checkbox("Make plot"):
            st.markdown('''
                Each symbol represents ~86 customers.
                ''')

            st.pyplot(width=900, height=800)


    if topic == "3. SMOTE-NC upsampling":
        infile = open('./telco_split_sets.pickle','rb')

        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)


        with st.echo():
            ## Import library
            from imblearn.over_sampling import SMOTENC
            from collections import Counter

            ## Get indices of columns containing categorical features
            cat_range = range(3, 27)

            ## Upsampling using SMOTE-NC
            smote_nc = SMOTENC(categorical_features=cat_range, random_state=0)

            X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

        with open('./smote_split_sets.pickle', 'wb') as f:
            pickle.dump([X_resampled, y_resampled, X_test, y_test, X_val, y_val], f)

        x = pd.DataFrame(y_resampled)[0].value_counts()

        data = {'Training set (positive class)' : x[0],
                'Training set  (negative class)' : x[1],
                'Test set' : X_test.shape[0],
                'Validation set' : X_val.shape[0]}

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

        if st.checkbox("Make plot"):
            st.markdown('''
                Each symbol represents ~86 customers.
                ''')

            st.pyplot(width=900, height=800)


    if topic == "4. Create new classes by clustering":
        df = pd.read_csv('./gower_res.csv')

        st.markdown(
        '''
        ```R
        ## Import libraries
        library(cluster)
        library(Rtsne)
        
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
        
        ## Visualize clusters by tSNE
        k <- 2     # No. clusters

        pam_fit <- pam(gower_dist, diss = TRUE, k)
        
        pam_results <- df %>%
          mutate(cluster = pam_fit$clustering) %>%
          group_by(cluster) %>%
          do(the_summary = summary(.))

        tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)

        tsne_data <- tsne_obj$Y %>%
          data.frame() %>%
          setNames(c("X", "Y")) %>%
          mutate(cluster = factor(pam_fit$clustering)) 
        ```
        ''')

        st.image('./tsne.png', use_column_width=True)

        st.markdown('''
        ```R
        library(dendextend)
        library(ggplot2)
        
        fit <- hclust(d=gower_dist, method="complete")  
        
        dend <- fit %>% as.dendrogram %>% hang.dendrogram
        
        dend %>% color_branches(k=2) %>% set("labels", "") %>% plot(horiz=FALSE)
        ```
        ''')

        st.image('./dendro.png', use_column_width=True)

        st.markdown('''
        ```R
        groups <- cutree(fit, k=2)   # "k=" defines the number of clusters you are using 
        
        new_df <- cbind(raw_df, groups)
        ```
        ''')

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

        plt.hlines(y=my_range, xmin=ordered_df['Group 1'], xmax=ordered_df['Group 2'], color='grey', alpha=0.4)
        plt.scatter(ordered_df['Group 1'], my_range, color='red', alpha=1, label='Group 1')
        plt.scatter(ordered_df['Group 2'], my_range, color='green', alpha=1, label='Group 2')
        plt.legend(loc='upper right', prop={'size': 12})

        # Add title and axis names
        plt.yticks(my_range, ordered_df['Category'])
        plt.xlabel('% of customers in group ')

        plt.style.use('default')

        plt.rcParams.update({'figure.figsize':[10, 8], 'font.size':16})

        plt.tight_layout()

        st.pyplot()

        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        fig = px.histogram(df, x="MonthlyCharges", color="groups", opacity=0.4,
                           color_discrete_sequence = ['red', 'green'], barmode = 'overlay', nbins=n_bins)

        fig.update_layout(legend_orientation="h",
                          legend=dict(x=0, y=1.1),
                          yaxis=go.layout.YAxis(
                              title=go.layout.yaxis.Title(
                                  text="Count"
                              )
                          ))

        fig2 = px.histogram(df, x="Tenure", color="groups", opacity=0.4,
                            color_discrete_sequence = ['red', 'green'], barmode = 'overlay', nbins=n_bins)

        fig2.update_layout(legend_orientation="h",
                          legend=dict(x=0, y=1.1),
                          yaxis=go.layout.YAxis(
                              title=go.layout.yaxis.Title(
                                  text="Count"
                              )
                          ))

        st.plotly_chart(fig)

        st.plotly_chart(fig2)

        df['groups'] = df['groups'].astype('str')

        df.drop('Churn', axis=1, inplace=True)

        with st.echo():
            binary_list = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                           "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                           "StreamingMovies", "PaperlessBilling"]

            onehot_list = ["Gender", "InternetService", "Contract", "PaymentMethod"]

            numeric_list = ["Tenure", "MonthlyCharges", "TotalCharges"]

            df['groups'] = np.where(df['groups'] == '2', 0, 1)

            df[binary_list] = np.where(df[binary_list] == 'Yes', 1, 0 )

            df = pd.get_dummies(df, columns=onehot_list)

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

        data = {'X train (negative class)': y_train.value_counts()[0], 'X train (positive class)': y_train.value_counts()[1],
                'X test': X_test.shape[0], 'X val':X_val.shape[0]}

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

        if st.checkbox("Waffle"):
            st.pyplot(width=900, height=800)



if page == pages[5]:
    st.sidebar.markdown('''
    ---
    ''')

    st.markdown('''
    ```Python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    params = {
        'min_samples_leaf': [1, 3, 5, 10, 25, 100],
        'max_features': ['sqrt', 'log2', 0.5, 1]
    }

    rf = RandomForestClassifier(n_estimators=50, oob_score=True)
    grid = GridSearchCV(rf, param_grid=params, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    grid.best_params_
    ```
    ''')


    if st.checkbox("Tune parameters"):

        st.dataframe(param_search())

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


        if st.checkbox("Check scores:"):
            with st.spinner("Hang on right, this takes a bit..."):
                score_list = []

                for dataset, params in zip(datasets, params_list):

                    infile = open(dataset, "rb")

                    X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

                    m = RandomForestClassifier(n_estimators=500,
                                               min_samples_leaf=params['min_samples_leaf'],
                                               max_features=params['max_features'],
                                               n_jobs=-1, oob_score=True)

                    m.fit(X_train, y_train)

                    score_list.append(score(m))

                scores = pd.DataFrame(score_list).set_index(pd.Index(['Original', 'Random oversample', 'SMOTE-NC', 'Clusters']))

                scores.plot.barh(figsize=(8, 6))

                plt.tight_layout()

                st.pyplot()

                plt.cla()


                if st.checkbox("Plot confusion matrix"):
                    for dataset, params in zip(datasets, params_list):

                        infile = open(dataset, "rb")

                        X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

                        m = RandomForestClassifier(n_estimators=500,
                                                   min_samples_leaf=params['min_samples_leaf'],
                                                   max_features=params['max_features'],
                                                   n_jobs=-1, oob_score=True)

                        cmatrix = confusion_matrix(RandomForestClassifier(n_estimators=500,
                                                                          min_samples_leaf=params['min_samples_leaf'],
                                                                          max_features=params['max_features'],
                                                                          n_jobs=-1, oob_score=True),
                                                   X_test, y_test,
                                                   classes=['No churn', 'Churn'],
                                                   cmap="Greens")

                        cmatrix.show()

                        st.pyplot()

                        plt.clf()

                    if st.checkbox("Check classification report"):
                        for dataset, params in zip(datasets, params_list):

                            infile = open(dataset, "rb")

                            X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)

                            m = RandomForestClassifier(n_estimators=500,
                                                       min_samples_leaf=params['min_samples_leaf'],
                                                       max_features=params['max_features'],
                                                       n_jobs=-1, oob_score=True)

                            crep = classification_report(RandomForestClassifier(n_estimators=500,
                                                                                min_samples_leaf=params['min_samples_leaf'],
                                                                                max_features=params['max_features'],
                                                                                n_jobs=-1, oob_score=True),
                                                         X_test, y_test,
                                                         cmap="Greens",
                                                         classes=['No churn', 'Churn']
                                                         )


                            crep.show()

                            st.pyplot()

                            plt.clf()


if page == pages[6]:
    # names = ['Original', 'Random oversampled', 'SMOTE-NC', 'Clustered']
    #
    # datasets = ['./telco_split_sets.pickle', './random_split_sets.pickle',
    #             './smote_split_sets.pickle', './cluster_split_sets.pickle']
    #
    # infile = open('./all_params.pickle', 'rb')
    # params_list = pickle.load(infile)
    #
    #
    # set_rf_samples(1500)
    #
    # imp_list = []
    #
    # for dataset, params, name in zip(datasets, params_list, names):
    #
    #     infile = open(dataset, "rb")
    #
    #     X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(infile)
    #
    #     m = RandomForestClassifier(n_estimators=500,
    #                                min_samples_leaf=params['min_samples_leaf'],
    #                                max_features=params['max_features'],
    #                                n_jobs=-1, oob_score=True)
    #
    #     m.fit(X_train, y_train)
    #
    #     imp = importances(m, X_test, y_test, n_samples=-1)
    #
    #     imp.columns = [name]
    #
    #     imp_list.append(imp)
    #
    #
    # imp_df = pd.concat(imp_list, axis=1, sort=True)
    #
    # with open('./imp_df.pickle', 'wb') as f:
    #     pickle.dump(imp_df, f)

    '''
    ```Python
    m = RandomForestClassifier(n_estimators=500,
                                min_samples_leaf=params['min_samples_leaf'],
                                max_features=params['max_features'],
                                n_jobs=-1, oob_score=True)
    
    m.fit(X_train, y_train)
    
    imp = importances(m, X_test, y_test, n_samples=-1)
    ```
    
    '''


    infile = open('./imp_df.pickle', 'rb')
    imp_df = pickle.load(infile)

    imp_df.columns = ['Original', 'Oversampled', 'SMOTE-NC', 'Clustered']

    g = sns.clustermap(imp_df,  standard_scale=1, figsize=(8, 9))

    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

    plt.subplots_adjust(left=0.05, right=0.54, top=0.98, bottom=0.2)

    st.pyplot(width=900, height=900)
















