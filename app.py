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
from figures import *

# Set outline
pages = ["1. Introduction",
         "2. Data cleaning",
         "3. Explore the dataset",
         "4. Variable encoding",
         "5. Create training, validation and test sets",
         "6. Model fitting",
         "7. Feature importance",
         "8. Where to go from here?"]

page = st.sidebar.selectbox('Navigate', options=pages)



# 1. Introduction
if page == pages[0]:
    st.sidebar.markdown('''
    
    ---
    
    Random forest
    ''')

    st.title('Introduction to survival analysis')

    st.markdown('''
        Placeholder
    ''')




# 2. Data cleaning
if page == pages[1]:
    st.title('Data preparation')

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
        st.header('1. Preview data')

        st.markdown('''
        ```Python
        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        ```
        
        
        ''')

        if st.checkbox("Preview data"):
            df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")


            df.head(10).T

            st.markdown('''
            Some summary info:
            
            ```Python
            df.info()
            ``` ''')

            if st.checkbox("Summarize the data"):
                st.dataframe(df_info(df))

                st.markdown('''
                There are some quick clean-ups that we need to go on this dataset before it is ready for modeling. We will walk through 
                each step in the next few pages.
                ''')




    if section == '2. Remove customer ID column':

        st.markdown('''
        Customer IDs need to be removed, not informative and high cardinality
        ''')

        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        with st.echo():
            df.drop(['customerID'], axis = 1, inplace=True)

        if st.checkbox('Drop customer ID column'):

            st.markdown('''
            Just to check:
            
            ```Python
            df.info()
            ```
            
            ''')

            st.dataframe(df_info(df))


    if section == '3. Re-encode variable':
        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)

        st.error('''Unlike other categorical variables in this dataset, `SeniorCitizen` is encoded by 0s and 1s, presumably corresponding to
                 "No"s and "Yes"s. ''')

        st.dataframe(df.head(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['SeniorCitizen']]))

        st.markdown('''
        ```Python
        df['SeniorCitizen'].value_counts()
        ```
        ''')

        st.warning(df['SeniorCitizen'].value_counts().to_dict())

        st.markdown('''
        While we will dummy encode the categorical variables in 0s and 1s later anyways, for the sake of consistency during the 
        data exploration phase, we will re-encode this variable in the same manner as the others: 
        ''')

        with st.echo():
            df['SeniorCitizen'] = np.where(df['SeniorCitizen'] == 1, 'Yes', 'No')

        if st.checkbox("Re-encode"):
            st.dataframe(df.head(10).style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['SeniorCitizen']]))

            st.markdown('''
            ```Python
            df['SeniorCitizen'].value_counts()
            ```
            ''')

            st.warning(df['SeniorCitizen'].value_counts().to_dict())

            st.markdown('''
            As a sanity check, we see that the proportion of "Yes"/"No" is the same as "0"/"1" after re-encoding.
            ''')


    if section == '4. Rename column headers':
        with st.echo():
            df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        if st.checkbox("Rename columns"):
            st.markdown('''
            
            Just a quick check:
            
            ```Python
            df.columns
            ```
            ''')

            st.write(list(df.columns))


    if section == '5. Correct column data type':
        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)

        df['SeniorCitizen'] = np.where(df['SeniorCitizen'] == 1, 'Yes', 'No')

        df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        v = df_info(df)

        st.markdown('''
        Next, we need convert the variable `TotalCharges` to the right data type (`float`), as it is currently 
        set as a categorical variable (`object`):
        ''')

        st.dataframe(v.style.apply(lambda x: ['background: lightgreen' if x.name in [18]
                                          else '' for i in x],
                               axis=1))

        st.markdown('''
        ```Python
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        ```
        ''')

        if st.checkbox("Set column data type to 'float'"):
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

                st.markdown('''
                It makes sense that for customers who just started with the company to not have total charges calculated. 
                Just to make sure that only customers with `Tenure` = 0 have no total charges recorded:
                ''')

                df = df[['Gender', 'SeniorCitizen', 'Tenure', 'TotalCharges',  'Partner', 'Dependents',
                         'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                         'MonthlyCharges',  'Churn']]


                st.markdown(
                    '''
                    ```Python
                    df[(df['Tenure'] == 0) | (df['TotalCharges'].isnull())]
                    ```
                    ''')

                if st.checkbox("Let's see"):
                    p = df[(df['Tenure'] == 0) | (df['TotalCharges'].isnull())]

                    st.dataframe(p.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Tenure', 'TotalCharges']]))


                    st.success('''
                    Indeed, the two sets of problematic data points are the same. As there are only 11 such data points out of >7,000 in total, 
                    we can remove them.
                    ''')

                    with st.echo():
                        df = df[df['Tenure'] > 0]






    if section == '6. Combine sparse levels':

        df = pd.read_csv("https://github.com/treselle-systems/customer_churn_analysis/raw/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df.drop(['customerID'], axis = 1, inplace=True)

        df['SeniorCitizen'] = np.where(df['SeniorCitizen']==1, 'Yes', 'No')

        df.rename(columns={'gender': 'Gender', 'tenure':'Tenure'}, inplace=True)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        df = df[df['Tenure'] > 0]

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

            fig.subplots_adjust(top=0.92, wspace=0.2, hspace=0.3)


        if st.checkbox("Abracadabra!"):
            with st.spinner('Working on it...'):

                plt.tight_layout()

                fig.delaxes(axes[8][1])

                st.pyplot()

                plt.cla()

            st.info('''
            So, for many of the variables related to online services and phone service, "No internet service" or "No phone service" 
            can be combined into the "No" category for that variable, as that information is already encoded in `InternetService` and `PhoneService`.
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
        st.markdown('''
            Click on any of the categorical variable names to expand the
             sunburt chart and see distribution of the levels in more detail.
            ''')

        with st.spinner('Working on it...'):

            fig = sunburst_fig()

            st.plotly_chart(fig, width=700, height=700)
    elif var1 != ' ' and var2 == ' ' and df[var1].dtype == 'object':
        st.markdown('''
            There are 7,032 customers in the dataset. Each symbol represents ~10 customers.
            ''')

        with st.spinner('Working on it...'):
            data = df[var1].value_counts().to_dict()

            fig = plt.figure(
                FigureClass=Waffle,
                rows=5,
                columns=14,
                values=data,
                legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.2), "fontsize":16},
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

            fig.update_layout(
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

            sns.set(style="ticks", font_scale=1.8, rc={'figure.figsize':(16, 11)})

            plt.tight_layout()

            st.pyplot()
    elif df[var1].dtype != 'object' and df[var2].dtype == 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=var2, nbins=n_bins)

            fig.update_layout(legend_orientation="h",
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



