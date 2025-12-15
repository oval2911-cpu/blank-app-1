# import libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.axes
import matplotlib.figure
import missingno as msno
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
sns.set_palette("Set2") #set consistent color palette for all plots


# title and subtitle for the app
st.title("Prediction of the cardiovascular events using the baseline patient characteristics from the Framingham dataset")

st.markdown("""*Tom Einhaus: i6339207, Alisa Ovsiannikova: i6365923*  
*MAI3002: Introduction to Programming*  
*Faculty of Health, Medicine, and Life Sciences*  
*Maastricht University*

***December 16th, 2025***""")


# section 1
with st.expander("1. Background and Research Question"):

    st.header("1. Background and Research Question") #background on the dataset and our RQ

    st.write("### Framingham Heart Study Dataset:")
    st.write("""- **Extensive follow-up study** dataset on **cardiovascular health** ongoing **since 1948 (1):**
    - Up to **3 follow-up periods** (fewer for some patients): **Period 1 = baseline**
    - **11,627 examination records** (rows) from **4,434 patients**
    - **39 variables:**
        - Demographics (sex, age)
        - Clinical health data (blood pressure, diabetes)
        - Lifestyle (smoking, BMI)
        - Occurrence of cardiovascular diseases (stroke, coronary heart disease (CHD))""")

    st.write("### Research Question (RQ):")
    st.write("**To what extent** can **baseline patient characteristics from the Framingham dataset** be used by machine-learning (ML) models to **reliably predict the occurrence of major cardiovascular events (CVD)** (stroke, CHD, myocardial infarction (MI), and coronary insufficiency)?")

    st.write("### Previous Research Findings:")
    st.write("Previously reported research shows that BMI, cholesterol and blood pressure can have an influence on the CVD risk (2).")


# section 2
with st.expander("2. Data Preparation"):
    st.header("2. Data Preparation")  #exploration of raw dataset and subsetting

    st.write('### Raw Dataset Preview:')
    # load dataset
    data_heart = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
    data_heart
    st.write(f'**Shape** (n of rows and columns) of the raw dataset: :blue-background[{data_heart.shape}]')
    st.write("""
    - *Each **row represents one exam**, so there are **more rows than number of patients** (RANDIDs repeat).*
    - *Some cells contain 'None', therefore, **some missing values can already be expected and will have to be handled.***
    """)

    st.write('### Raw Dataset Descriptive Statistics:')
    st.write(data_heart.describe())
    st.write("""
    - ***RANDID variable is not interesting**, therefore, it **will be removed** in the following steps.*
    - *Interestingly, **sex was encoded with 1 and 2**. It **introduces artificial ordering** and numerical meaning and may **bias ML model** learning and interpretation.
             Therefore, we should **be careful** and encode it with 0 and 1.*
    """)
    # not visualized in the app, but the raw dataset was also checked for missing values and duplicates
    print(data_heart.isna().sum()) #in the terminal output it can be seen that there are some missing values that have to be checked for and handled
    print(data_heart.duplicated().sum()) #in the terminal output it can be seen that there are no duplicates

    st.write('### Raw Dataset Variables:')
    st.write(data_heart.columns)
    st.write(f'Number of all variables: :blue-background[{len(data_heart.columns)}]')
    st.write("""
    - *To answer the RQ, the **subset will be created** in 2 steps:*
        1. ***Filtering:***
            - *Keep **only Period 1 records** for each participant to define baseline 'at-risk' population.*
            - ***Filter out participants with prevalent disease** (PREVCHD=1, PREVSTRK=1) at Period 1 to ensure we are keeping only the population 'at risk' for a first event.*
            - ***Exclude participants who died during follow-up without a recorded cardiovascular event**, as their disease status beyond death cannot be determined.*
        2. ***Including** only 14 **variables of interest:***
            - *Demographics: **SEX, AGE, educ***
            - *Clinical health data: **TOTCHOL, SYSBP, DIABP, DIABETES, HEARTRTE, GLUCOSE***
            - *Lifestyle: **CURSMOKE, CIGPDAY, BMI***
            - *Occurrence of cardiovascular diseases: **ANYCHD, STROKE***
        - *The following 25 variables will therefore be **excluded:***
            - ***RANDID, PERIOD, PREVCHD, PREVSTRK, TIME, DEATH**: do not contain any valuable information as the data were already filtered.*
            - ***ANGINA, PREVAP, TIMEAP**: angina pectoris was outside the scope of the CVD outcomes investigated.*
            - ***HOSPMI, MI_FCHD, CVD, PREVMI, TIMECVD, TIMEMIFC, TIMEMI**: the information they contain is captured by other retained variables (ANYCHD, STROKE, PREVCHD, PREVSTRK, TIMECHD, TIMESTRK).*
            - ***HDLC, LDLC**: lack of baseline (Period 1) measurements.*
            - ***PREVHYP, HYPERTEN, TIMECHD, TIMESTRK, TIMEDTH, TIMEHYP, BPMEDS**: contain post-baseline or outcome-related information that would introduce data leakage.*
    """)

    # copy raw dataset to a new subset variable with a short name to make changes in a new variable and make it easier to call it
    dhs = data_heart.copy()
    # step 1: filtering
    dhs = dhs.loc[dhs['PERIOD'] == 1].copy() #keep only period 1
    dhs = dhs.loc[
        (dhs['PREVCHD'] == 0) & (dhs['PREVSTRK'] == 0)
        ].copy() #filter out participants with prevalent disease
    dhs = dhs.loc[~((dhs['ANYCHD'] == 0) & (dhs['DEATH'] == 1))] #filter out deceased participants with no disease
    dhs = dhs.loc[~((dhs['STROKE'] == 0) & (dhs['DEATH'] == 1))] 
    # step 2: keeping variables of interest
    dhs = dhs[['SEX', 'AGE', 'educ', 'TOTCHOL', 'SYSBP', 'DIABP', 'DIABETES',
    'HEARTRTE', 'GLUCOSE', 'CURSMOKE', 'CIGPDAY', 'BMI', 'ANYCHD', 'STROKE']]


    st.write("### Created Subset Preview:")
    dhs
    st.write(f'**Shape** (n of rows and columns) of the subset: :blue-background[{dhs.shape}]')

    st.write("### Created Subset Variables:")
    """Overview of all columns of the created subset:"""
    st.write(dhs.columns)

    # not visualized in the app, but the subset descriptive statistics were also checked
    print(dhs.describe()) #it can be seen in the terminal output

with st.expander("3. Exploratory Data Analysis (EDA), Cleaning, and Feature Engineering"):
    st.header("3. Exploratory Data Analysis (EDA), Cleaning, and Feature Engineering") #exploration of our subset and cleaning

    # define function for plotting (for all plots to be formatted in a same way)
    def render_plot(obj, title="", *args, **kwargs):
        figsize=(12, 8)

        # for pandas Series
        if isinstance(obj, pd.Series):
            fig = obj.plot(kind='hist', figsize=figsize, **kwargs).get_figure()
        
        # for callables
        elif callable(obj):
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            obj(*args, **kwargs)
        
        # for figures
        elif isinstance(obj, matplotlib.figure.Figure):
            fig = obj
        
        # for single axes
        elif isinstance(obj, matplotlib.axes.Axes):
            fig = obj.get_figure()
        
        # for array of axes (e.g. df.hist())
        elif isinstance(obj, (np.ndarray, list)) and all(isinstance(a, matplotlib.axes.Axes) for a in np.ravel(obj)):
            axes = np.ravel(obj)
            fig = axes[0].get_figure()
        
        # in case of not supported types show an error
        else:
            raise TypeError(f"render_plot received an unsupported type: {type(obj)}")
        
        # layout
        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title, fontsize=30)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        for ax in fig.axes:
            ax.tick_params(axis='x', rotation=90)
        
        # show in streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        return fig


    st.write("### Subset Variable Distributions")
    selectedVariable = st.selectbox("Select variable to plot:", dhs.select_dtypes(include='number').columns)
    render_plot(dhs[selectedVariable].hist, bins=30, alpha=0.4, edgecolor='black', label=selectedVariable)
    st.write("""
    - *More females*
    - *Age has some dips*
    - *educ, TOTCHOL, SYSBP, HEARTRTE, GLUCOSE, CIGPDAY, BMI seem to be right skwewed*
    - *DIABP, CURSMOKE: approximately symmetric*
    - *DIABETES, ANYCHD, STROKE: no event/comorbidity is more represented*
    """)

    st.write("### Create 1 Target Variable")
    st.write("""
    **Before data pre-processing**, the data should be **split into train and test sets** to avoid data leakage.
    To enable this and **formulate the problem as a binary classification** task, a **single target variable has to be defined** by combining
    2 CVD event variables (ANYCHD, STROKE). In the Framingham dataset, **ANYCHD=1 and STROKE=1** in the Period 1 record **indicate that the
    event happened at any time during the follow-up** after Period 1.
    Therefore, we **can create one target variable** for binary classification that would allow us to do train/test split.
    """)
    dhs['Disease'] = (
        (dhs['ANYCHD'] == 1) | (dhs['STROKE'] == 1)
    ).astype(int)
    st.write('Values of the new **target variable**:')
    st.write(dhs.Disease.value_counts())
    # Drop the original target columns
    dhs = dhs.drop('ANYCHD', axis=1)
    dhs = dhs.drop('STROKE', axis=1)

    #splitting data
    X = dhs.drop('Disease', axis=1) 
    y = dhs['Disease']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=1,
        stratify=y 
    )
    # allocate all categorical and numerical variables to corresponding separate variables
    categorical = ['SEX', 'educ', 'DIABETES', 'CURSMOKE']
    numerical = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE', 'CIGPDAY', 'BMI']
    # create 2 dataframes with different types of variables
    num_df = X_train[numerical]
    cat_df = X_train[categorical]

    rawX_train = X_train.copy() #keep a copy of raw X_train for later visualization comparison
    rawX_test = X_test.copy() #keep a copy of raw X_test for later visualization comparison
    """With the target variable defined, the **dataset is ready for subsequent data cleaning and preprocessing**."""


    st.write("### Outlier Detection and Handling")
    st.write("It is important to only act on the training set, and only use baseline features to prevent data leakage.")
    st.write("A boxplot provides an overview over the outliers of the data:")
    render_plot(sns.boxplot, data=X_train[numerical], orient='h')
    st.write("""
    - *Most of the outliers are on the right hand side due to skewness.*
    - *Age has no outliers.*
    - *Outermost physiological limits are: TOTCHOL (50-500 mg/dL), SYSBP (40-350 mmHg), GLUCOSE (20-1000 mg/dL) (3,4,5).*
    - *Comparing those limits to our outliers, all are theoretically possible due to extreme medical conditions, therefore, decided to keep them.*
    """)

    # not visualized in the app, but also checked if there are many values outside whiskers (1.5*IQR)
    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))
    outlier_counts = outlier_mask.sum().sort_values(ascending=False)
    print(outlier_counts) #can be seen in terminal output

    # not visualized in the app, but also checked if there are many values above z score of 3 (another way of detecting outliers)
    from scipy.stats import zscore
    z_scores = num_df.apply(zscore)
    outliers_z = (abs(z_scores) > 3).sum().sort_values(ascending=False)
    print(outliers_z) #can be seen in terminal output

    # not visualized in the app, but also checked: How much influence do these outliers have on mean and std when being included or disregarded?
    #without outliers
    clean_df = num_df[~outlier_mask.any(axis=1)]
    #mean and std for with outliers vs without outliers and put in pd dataframe
    compare = pd.DataFrame({
        'Mean (all)': num_df.mean(),
        'Mean (no outliers)': clean_df.mean(),
        'Std (all)': num_df.std(),
        'Std (no outliers)': clean_df.std()
    })
    print(compare.round(2)) #can be seen in terminal output, no drastic change, however, decided to keep them as they may represent real clinical values


    st.write("### Missing Data Handling")
    st.write(pd.DataFrame({
        'Missing Count': X_train.isna().sum(),
        'Missing Percentage %': ((X_train.isna().sum() / len(X_train)) * 100).round(2)
    })
    )
    st.write("""
    - *5 Variables: educ, TOTCHOL, GLUCOSE, CIGPDAY, and BMI have missing values.*
    - *GLUCOSE has >5% missing, which automatically makes it a candidate for model-based imputation.*
    """)

    st.write("""Another visualization of missing data is the missingno matrix, which shows percentage and absolute number of present data:""")
    render_plot(msno.bar(X_train).figure)

    st.write("""
             - *GLUCOSE will be model-based imputed due to its high missingness.*
             """)
    st.write("""
             Another reason for model based imputation is if type of missingness is missing at random (MAR).  
             To check if variables are correlated with missingness of GLUCOSE and TOTCHOL, correlation heatmap and missingno heatmap are plotted below:*
    """)

    #add correlations between variables
    render_plot(sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8}))

    st.write("""
    Missingno heatmap relates missingness of variables to each other:""")

    render_plot(msno.heatmap(X_train).figure)

    st.write("""
    - *It can be seen that TOTCHOL has slight correlations with missingness in BMI and GLUCOSE.*
    - *This makes it a candidate for model based imputation as well, as its missingness may depend on other variables.*""")

    st.write("""
    We will use MICE (Multiple Imputation by Chained Equations) for imputation of TOTCHOL and GLUCOSE.""")

    st.write("""
    - *For all other variables, as the missingness is less than 5% in each variable, we will do simple imputation.*
    - *educ is categorical, therefore, the mode imputation will be used.*
    - *For CIGPDAY and BMI, the median imputation will be used to be more robust against outliers.*
    """)

    # mode imputation (educ)
    mode_value = X_train['educ'].mode()[0]
    X_train['educ'] = X_train['educ'].fillna(mode_value)
    X_test['educ'] = X_test['educ'].fillna(mode_value)

    # median imputation (CIGPDAY, BMI)
    median_cigpday = X_train['CIGPDAY'].median() #compute median only using train data to avoid data leakage
    X_train['CIGPDAY'] = X_train['CIGPDAY'].fillna(median_cigpday) #use computed median for both train and test sets
    X_test['CIGPDAY'] = X_test['CIGPDAY'].fillna(median_cigpday)

    median_bmi = X_train['BMI'].median()
    X_train['BMI'] = X_train['BMI'].fillna(median_bmi)
    X_test['BMI'] = X_test['BMI'].fillna(median_bmi)

    median_heartrate = X_train['HEARTRTE'].median() #was imputed due to its missingness (1 value) in the test set
    X_train['HEARTRTE'] = X_train['HEARTRTE'].fillna(median_heartrate)
    X_test['HEARTRTE'] = X_test['HEARTRTE'].fillna(median_heartrate)

    
    # MICE Imputation (TOTCHOl, GLUCOSE) 
    mice_cols = ['AGE', 'SEX', 'educ', 'TOTCHOL', 'SYSBP', 'DIABP',
    'DIABETES', 'HEARTRTE', 'GLUCOSE', 'CURSMOKE', 'CIGPDAY', 'BMI']
    X_train_temp = X_train[mice_cols].copy() #create the temporary DataFrame X_train_temp, preserving the original index
    imp = IterativeImputer(max_iter=10, random_state=42) #create the imputation model (imp) that we will reuse
    X_imputed_train_array = imp.fit_transform(X_train_temp) #fit and transform only on train data 
    X_imputed_train = pd.DataFrame(
        X_imputed_train_array, 
        index=X_train_temp.index, # Index Alignment
        columns=mice_cols
    ) #convert back to DataFrame using the original index
    X_train['TOTCHOL'] = X_imputed_train['TOTCHOL'] #update the original DataFrame (X_train)
    X_train['GLUCOSE'] = X_imputed_train['GLUCOSE']

    X_test_temp = X_test[mice_cols].copy() #create the temporary DataFrame X_test_temp
    X_imputed_test_array = imp.transform(X_test_temp) # No 'fit' here! Only transform the test set (using the SAME 'imp' model fitted on X_train)
    X_imputed_test = pd.DataFrame(
        X_imputed_test_array, 
        index=X_test_temp.index, # Index Alignment
        columns=mice_cols
    ) #convert back to DataFrame using the original index
    X_test['TOTCHOL'] = X_imputed_test['TOTCHOL'] #update the original DataFrame (X_test)
    X_test['GLUCOSE'] = X_imputed_test['GLUCOSE']

    st.write("Check for the missing data again after imputation (there should be no missing values):")
    st.write("Train set:")
    st.write(X_train.isna().sum())
    st.write("Test set:")
    st.write(X_test.isna().sum())
    st.write("""
    - *X_train and X_test are now fully imputed and cleaned for further modeling.*
    """)


    st.write("### Apply transformations against skewness")
    st.write("Check skewness (calculated on X_train):")
    skewness_results = X_train[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    st.write(skewness_results)
    st.write("""
    - *From the skewness results we can identify these variables as highly skewed: GLUCOSE, CIGPDAY, SYSBP, BMI.*
    - *Log transformation will be applied to them.*
    """)
    st.write("""
    Check distributions after log tranformation:""")
    log_transform_cols = ['GLUCOSE', 'CIGPDAY', 'SYSBP', 'BMI']
    for col in log_transform_cols:
        X_train[col] = np.log1p(X_train[col]) #transform X_train
    for col in log_transform_cols:
        X_test[col] = np.log1p(X_test[col]) #transform X_train
    # make a on/off button to select train/test set for visualization
    on = st.toggle('Turn on to see test set.')
    if on:
        dataset = X_test
        dataset_name = "X_test set"
    else:
        dataset = X_train
        dataset_name = "X_train set"
    selectedVariable = st.selectbox("Select variable to plot:", log_transform_cols) #select a variable from the columns that were transformed
    render_plot(dataset[selectedVariable].hist, f'{dataset_name}', bins=30, alpha=0.4, edgecolor='black', label=selectedVariable)
    st.write("""
    - *Distributions are symmetric, therefore, the transformation worked.*
    """)


    st.write("### One Hot Encoding for SEX variable")
    st.write("""*As the SEX variable was encoded with 1 and 2, we map original value 1 (Male) to 0
    and value 2 (Female) to 1 to remove artificial ordering.*""")
    # create a new binary column
    X_train['SEX_Female'] = X_train['SEX'].replace({
        1: 0, # map original value 1 (Male) to 0
        2: 1  # map original value 2 (Female) to 1
    })
    X_test['SEX_Female'] = X_test['SEX'].replace({
        1: 0, # map original value 1 (Male) to 0
        2: 1  # map original value 2 (Female) to 1
    })
    st.write(f'*Values of new **SEX_Female variable** after encoding: :blue-background[{set(X_train.SEX_Female)}]*')
    X_train = X_train.drop('SEX', axis=1) #drop original SEX variable
    X_test = X_test.drop('SEX', axis=1) #drop original SEX variable

    # Do the same on the raw data copies for later visualization comparison
    # create a new binary column
    rawX_train['SEX_Female'] = rawX_train['SEX'].replace({
        1: 0, # map original value 1 (Male) to 0
        2: 1  # map original value 2 (Female) to 1
    })
    rawX_test['SEX_Female'] = rawX_test['SEX'].replace({
        1: 0, # map original value 1 (Male) to 0
        2: 1  # map original value 2 (Female) to 1
    })
    rawX_train = rawX_train.drop('SEX', axis=1) #drop original SEX variable
    rawX_test = rawX_test.drop('SEX', axis=1) #drop original SEX variable


    st.write("### Scaling")
    st.write("""Since the outlires have been already handled, the standard scaling can be used to prepare data for modeling.""")
    st.write("""StandardScaler was fitted on the training set and subsequently applied to both the training and test sets to prevent data leakage.""")

    scaler = StandardScaler() #since we already handled outliers we can use standard scaling
    scaler.fit(X_train[numerical]) #fit only on train data (leakage prevention), calculating mean and standard deviation of X_train
    X_train[numerical] = scaler.transform(X_train[numerical]) #transform train data
    X_test[numerical] = scaler.transform(X_test[numerical]) #transform test data

    # do the same on the raw data copies for later visualization comparison
    scaler.fit(rawX_train[numerical]) #fit only on train data (leakage prevention), calculating mean and standard deviation of X_train
    rawX_train[numerical] = scaler.transform(rawX_train[numerical]) #transform train data
    rawX_test[numerical] = scaler.transform(rawX_test[numerical]) #transform test data
    st.write("Data preparation is now complete, and the data is ready for modeling.")


with st.expander("4. Visualization of the Final Data"):

    st.header("4. Visualization of the Final Data")
    st.write("### Final Data Variable Distributions")

    on = st.toggle('Turn on to see test set')
    if on:
        dataset = X_test
        dataset_name = "X_test"
        raw_dataset = rawX_test
    else:
        dataset = X_train
        dataset_name = "X_train"
        raw_dataset = rawX_train

    overlap_raw = st.toggle("Overlap RAW (scaled)")
    selectedVariable = st.selectbox("Select variable to plot:", dataset.columns)

    if overlap_raw:
        def plot_overlaid():
            # Raw (red) with KDE
            sns.histplot(raw_dataset[selectedVariable], bins=30, kde=True, 
                        color='red', alpha=0.3, stat='density', label='Raw')
            # Cleaned (blue) with KDE  
            sns.histplot(dataset[selectedVariable], bins=30, kde=True, 
                        color='blue', alpha=0.3, stat='density', label='Cleaned')
            plt.legend()
        
        render_plot(plot_overlaid, title=f"{dataset_name} vs Raw - {selectedVariable}")
        st.write(f"Total training samples (X_train): :blue-background[{X_train.shape[0]}]")
    else:
        def plot_single():
            # Single distribution with stats
            sns.histplot(dataset[selectedVariable], bins=30, kde=True, 
                        color='teal', alpha=0.3, stat='density')
            plt.legend()
        
        render_plot(plot_single, title=f"{dataset_name} - {selectedVariable}")
        st.write(f"Total training samples (X_train): :blue-background[{X_train.shape[0]}]")

    on = st.toggle('Turn on to see test target set')
    if on:
        dataset = y_test
        dataset_name = "y_test"
    else:
        dataset = y_train
        dataset_name = "y_train"
    render_plot(dataset.value_counts().plot, kind='pie', autopct='%1.1f%%', title=f'Training Target Disease Distribution ({dataset_name})')
    st.write(f'Training target {dataset_name} shape: {dataset.shape} with {dataset.value_counts()[0]} CVD patients and {dataset.value_counts()[1]} healthy patients.')
    st.write("""
    - *The target distribution indicates imbalanced dataset, thus, it should be kept in mind during the ML models training.*
    - *The target class distribution in the training and test sets remained consistent after pre-processing.*
             """)


with st.expander("5. ML Models Training and Prediction Evaluation"):
    st.header("5. ML Models Training and Prediction Evaluation") #4 algorithms, evaluation also includes CV and feature importance

    st.write('### *Predicting CVD from baseline patient characteristics*')


    # interactive button to select cross-validation and number of folds
    cv = st.checkbox("Cross-validation",
                            value=True)
    cv_value = st.selectbox("Select number of folds for cross-validation:",
                                    [5, 10])
        

    # model evaluation function (so that it can be reused for all models)
    def model_evaluation(model_name, model, prediction):
        # accuracy and classification report
        with st.expander("Accuracy and Classification Report"):
            accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
            report = classification_report(y_true=y_test, y_pred=prediction, output_dict = True)
            st.write(f'Accuracy: {accuracy:.2f}')
            st.dataframe(pd.DataFrame(report).transpose())

        # confusion matrix
        with st.expander("Confusion Matrix"):
            cm = confusion_matrix(y_true=y_test, normalize='true', y_pred=prediction)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, ax=ax_cm)
            st.pyplot(fig_cm)

        # ROC curve and AUC (since we have binary classification)
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_value = auc(fpr, tpr)
        with st.expander("AUC"):
            fig_auc, ax_auc = plt.subplots()
            ax_auc.plot([0, 1], [0, 1], 'k--')
            ax_auc.plot(fpr, tpr, label=f'{model_name} (area = {auc_value:.3f})')
            ax_auc.set(
            xlabel='False positive rate',
            ylabel='True positive rate (Recall)',
            title='ROC curve')
            ax_auc.legend(loc='best');
            st.pyplot(fig_auc)

        # Cross-validation
        if cv:
            with st.expander("Cross-validation"):
                cv_scores = cross_val_score(model, X_train, y_train, scoring='f1', cv=cv_value)
                prediction_cv = cross_val_predict(model, X_train, y_train, cv=cv_value)
                st.write(f'Cross-validation f1-score for class 1 is {cv_scores.mean().round(2)} with a standard deviation of {cv_scores.std().round(2)}')
                # confusion matrix
                cm_cv = confusion_matrix(y_true=y_train, normalize='true', y_pred=prediction_cv)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm_cv, annot=True, ax=ax_cm)
                st.pyplot(fig_cm)
                # AUC
                y_prob_cv = model.predict_proba(X_test)[:,1]
                fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_test, y_prob_cv)
                auc_cv = auc(fpr_cv, tpr_cv)
                fig_cv, ax_cv = plt.subplots()
                ax_cv.plot([0, 1], [0, 1], 'k--')
                ax_cv.plot(fpr_cv, tpr_cv, label=f'{model_name} (cv) (area = {auc_cv:.3f})')
                ax_cv.set(
                xlabel='False positive rate',
                ylabel='True positive rate (Recall)',
                title='ROC curve')
                ax_cv.legend(loc='best');
                st.pyplot(fig_cv)


    st.write('### Algorithm 1: Logistic regression')
    model_name = "Logistic regression"
    # specify parameters 
    class_weight_lr = st.selectbox('Select class_weight parameter:',
                                ['balanced', None], #the first value is always a default here and in the following select boxes unless user specifies it
                                key = 'class_weight_lr') #added to avoid duplicated class_weight select boxes (bc use them below as well)
    penalty_lr = st.selectbox('Select penalty parameter (lbfgs solver by default):',
                            ['l2', None]) #lbfgs solver by default can only have these penalties
    st.write('Fine-tuned parameters are: balanced and l2 penalty.')

    # TRAINING AND PREDICTION
    # define logistic regression (lr) classifier
    model_lr = LogisticRegression(max_iter=1000,
                                class_weight = class_weight_lr,
                                penalty = penalty_lr)
    # train lr
    model_lr = model_lr.fit(X_train, y_train)
    # predict using test data
    prediction_lr = model_lr.predict(X_test)
    # EVALUATION
    model_evaluation(model_name, model_lr, prediction_lr)


    st.write('### Algorithm 2: Decision Tree')
    model_name = "Decision Tree"
    # specify parameters 
    class_weight_dt = st.selectbox('Select class_weight parameter:',
                                ['balanced', None],
                                key = 'class_weight_dt')
    splitter_dt = st.selectbox('Select splitter:',
                            ['best', 'random'])
    max_depth_dt = st.slider('Specify maximum tree depth:',
                            min_value = 1, #1 is chosen bc this way user can see that most probalby this value is too low to get valuable predictions
                            max_value = X_train.shape[0]-1, #this specific value is the max possible value for depth (number of samples - 1)
                            value = 10, #chosen as default as the best depth found
                            key = 'max_depth_dt')
    min_samples_leaf_dt = st.slider('Specify minimum number ' \
                                    'of samples in a leaf node ' \
                                    'after splitting to avoid overfitting:',
                                    min_value = 1, #should not be used as leads to overfitting
                                    max_value = X_train.shape[0] // 2, #this is a max possible value for that (half of number of samples)
                                    value = 20, #chosen as default as the best found
                                    key = 'min_samples_leaf_dt')
    st.write('Fine-tuned parameters are: balanced, best splitter, depth of 10, and 20 minimum samples per leaf node.')

    # TRAINING AND PREDICTION
    # define decision tree (dt) classifier
    model_dt = tree.DecisionTreeClassifier(random_state=42,
                                        max_depth=max_depth_dt,
                                        splitter = splitter_dt,
                                        class_weight = class_weight_dt,
                                        min_samples_leaf = min_samples_leaf_dt)
    # train dt
    model_dt = model_dt.fit(X_train, y_train)
    # predict using test data
    prediction_dt = model_dt.predict(X_test)
    # EVALUATION
    model_evaluation(model_name, model_dt, prediction_dt)


    st.write('### Algorithm 3: Random Forest')
    model_name = "Random Forest"
    # specify parameters 
    class_weight_rf = st.selectbox('Select class_weight parameter:',
                                ['balanced', 'balanced_subsample', None],
                                key = 'class_weight_rf')
    n_estimators_rf = st.slider('Select number of decision trees in the forest:',
                                min_value = 10,
                                max_value = 500, #to not make it too computationally heavy
                                value = 100) #chosen as default as the best found
    max_depth_rf = st.slider('Specify maximum tree depth:',
                            min_value = 1,
                            max_value = X_train.shape[0]-1,
                            value = 6, #chosen as default as the best found
                            key = 'max_depth_rf')
    min_samples_leaf_rf = st.slider('Specify minimum number ' \
                                    'of samples in a leaf node ' \
                                    'after splitting to avoid overfitting:',
                                    min_value = 1,
                                    max_value = X_train.shape[0] // 2,
                                    value = 14,
                                    key = 'min_samples_leaf_rf')        
    st.write('Fine-tuned parameters are: balanced, 200 trees, depth of 6, and 14 minimum samples per leaf node.')
             

    # TRAINING AND PREDICTION
    # define random forest classifier
    model_rf = RandomForestClassifier(n_estimators = n_estimators_rf,
                                    max_depth=max_depth_rf,
                                    random_state=0,
                                    class_weight = class_weight_rf,
                                    min_samples_leaf = min_samples_leaf_rf)
    # train rf
    model_rf = model_rf.fit(X_train, y_train)
    # predict using test data
    prediction_rf = model_rf.predict(X_test)
    # EVALUATION
    model_evaluation(model_name, model_rf, prediction_rf)


    st.write('### Algorithm 4: KNN')
    model_name = "KNN"
    # specify parameters
    n_neighbors_KNN = st.slider('Select number of neighbors:',
                                min_value = 1, #can lead to overfitting
                                max_value = 500, #not to make it computationally heavy, can lead to underfitting
                                value = 5, #chosen as default as the best found 
                                step = 2) #only odd numbers to avoid ties in a classification majority vote
    weights_KNN = st.selectbox('Select weight function:', 
                            ['uniform', 'distance', None]) 
    st.write('Fine-tuned parameters are: 5 neighbours and uniform function.')
    # TRAINING AND PREDICTION
    # define KNN classifier
    model_KNN = KNeighborsClassifier(n_neighbors = n_neighbors_KNN,
                                    weights = weights_KNN,
                                    algorithm='auto',
                                    leaf_size=30, #default
                                    metric='minkowski') #since p=2 is default in sklearn knn --> metric is Euclidean distance
    # train rf
    model_KNN = model_KNN.fit(X_train, y_train)
    # predict using test data
    prediction_KNN = model_KNN.predict(X_test)
    # EVALUATION
    model_evaluation(model_name, model_KNN, prediction_KNN)


with st.expander("6. Comparing ML Models"):
    st.header("6. Comparing ML Models") #selecting the best model out of 4 fine-tuned models
    #table with best fine-tuned models comparison
    # and explanation why we think which one is better
    model_comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ],
        "Accuracy": [
            0.61,
            0.61,
            0.66,
            0.76
        ],
        "Recall (CVD)": [
            0.55,
            0.54,
            0.56,
            0.19
        ],
        "Precision (Healthy)": [
            0.83,
            0.83,
            0.85,
            0.80
        ],
        "F1-score (CVD)": [
            0.38,
            0.38,
            0.42,
            0.26
        ],
        "ROC–AUC": [
            0.65,
            0.61,
            0.66,
            0.60
        ],
        "CV F1-score (CVD)": [
            "0.47 ± 0.02",
            "0.40 ± 0.02",
            "0.44 ± 0.02",
            "0.21 ± 0.02"
        ]
    })

    st.dataframe(model_comparison, use_container_width=True)

    st.write("""
             **Models evaluation:**  
             Four ML models were evaluated for **predicting future cardiovascular disease (CVD)** using baseline patient
             characteristics: **Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), and K-Nearest Neighbors (KNN).**
             Model performance was assessed using accuracy, class-specific precision and recall, F1-score for the positive class (CVD),
             ROC–AUC, and cross-validated F1-score. Given that the data were imbalanced, **accuracy was interpreted with caution** and was
             not used as the primary criterion for model comparison. **F1-score and recall for the positive class were prioritised** as they
             **better reflect the ability to identify CVD cases.**""")
    st.write("""
    **Final model choice: Logistic Regression.**  
             Logistic Regression was selected as the best model for cardiovascular disease prediction in this imbalanced dataset.
             Although **Random Forest achieved slightly higher test-set F1-score** and **less overfitting according to the cross-validation** F1-score,
             Logistic Regression maintained **strong recall for CVD cases relatively to other models.** In addition, its **better interpretability** compared to, for example, Random Forest make it the **most
             appropriate choice for clinical risk prediction.** Whereas, the **KNN model was excluded** due to its inability to identify positive cases despite high accuracy,
             which is not a reliable metric for imbalanced datasets.
             """)
    st.write("""
    **Possible reasons for limited model performance:**  

    **1. Timing of events was ignored**  
    - *A CVD event after 1 year was treated the same as an event after 20 years.*  
    - *Long follow-up periods allow substantial divergence between individuals with identical baseline profiles (especially for lifestyle features).*  

    **2. Limited follow-up**  
    - *Follow-up duration varied substantially across participants.*  

    **3. Patients at the beginning of the follow-up are younger**  
    - *According to the literature, in young patients at baseline the most predictable markers for future CVD events are: lipoprotein A, hereditary CVD,
    familiary hypercholesteremia, and ApoB value (6,7,8,9). Including these in the baseline could mitigate the problems with model performance.*  

    **4. Features used**  
    - *Baseline features are single measures, sometimes self-reported and thus inherently have low precision.*  

    **5. Heterogeneous target label**  
    - *Combined ANYCHD and STROKE included multiple cardiovascular outcomes with partly distinct underlying mechanisms and risk profiles. 
    This increases class overlap and introduces label noise, which inherently limits the ability of models to distinguish future CVD cases
    from non-cases based on baseline features.*  

    **6. Feature engineering**  
    - In the future, SYSBP and DIABP could be combined in one feature as they are highly correlated.
    """)
             

with st.expander("7. Conclusion"):
    st.header("7. Conclusion")

    st.write("""
    **Research Question:**  
    - **To what extent** can **baseline patient characteristics from the Framingham dataset** be used by machine-learning models to **reliably predict the occurrence of major cardiovascular events** (stroke, CHD, myocardial infarction (MI), and coronary insufficiency)?  
             
    **Conclusion:**  
    - **Baseline patient characteristics** from the Framingham dataset allow only **limited predictive discrimination** and are **insufficient** on their own **for reliable prediction of future cardiovascular events.**""")  

with st.expander("8. References"):
    st.header("8. References") #include genAI statement

    st.write("""
    1. Framingham Heart Study Longitudinal Data Documentation [Internet]. Available from: https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf?link_time=2024-05-26_10:36:20.705109  
    2. Bays H. Ten things to know about ten cardiovascular disease risk factors. American Journal of Preventive Cardiology [Internet]. 2021 Mar 1;5(100149):100149. Available from: https://www.sciencedirect.com/science/article/pii/S2666667721000040#  
    3. Pejic RN. Familial hypercholesterolemia. The Ochsner Journal [Internet]. 2014 [cited 2023 Mar 23];14(4):669–72. Available from: https://pubmed.ncbi.nlm.nih.gov/25598733/  
    4. Hörber S, Hudak S, Kächele M, Overkamp D, Fritsche A, Häring HU, et al. Unusual high blood glucose in ketoacidosis as first presentation of type 1 diabetes mellitus. Endocrinology, Diabetes & Metabolism Case Reports. 2018 Sep 24;2018.  
    5. Ketch T, Biaggioni I, Robertson R, Robertson D. Four Faces of Baroreflex Failure. Circulation. 2002 May 28;105(21):2518–23.  
    6. Fatemeh Vazirian, Sadeghi M, Theodoros Kelesidis, Budoff MJ, Zandi Z, Samadi S, et al. Predictive value of lipoprotein(a) in coronary artery calcification among asymptomatic cardiovascular disease subjects: A systematic review and meta-analysis. Nutrition Metabolism and Cardiovascular Diseases [Internet]. 2023 Jul 14 [cited 2025 Dec 15];33(11):2055–66. Available from: https://www.nmcd-journal.com/article/S0939-4753(23)00285-5/abstract  
    7. Lloyd-Jones DM, Nam BH, D’Agostino, Sr RB, Levy D, Murabito JM, Wang TJ, et al. Parental Cardiovascular Disease as a Risk Factor for Cardiovascular Disease in Middle-aged Adults. JAMA [Internet]. 2004 May 12;291(18):2204. Available from: https://jamanetwork.com/journals/jama/fullarticle/198726  
    8. Akioyamen LE, Genest J, Chu A, Inibhunu H, Ko DT, Tu JV. Risk factors for cardiovascular disease in heterozygous familial hypercholesterolemia: A systematic review and meta-analysis. Journal of Clinical Lipidology. 2019 Jan;13(1):15–30.  
    9. Epstein E, Ekpo E, Evans D, Varughese E, Hermel M, Jeschke S, et al. Apolipoprotein B outperforms low density lipoprotein particle number as a marker of cardiovascular risk in the UK Biobank. European journal of preventive cardiology [Internet]. 2025 Jan;zwaf554. Available from: https://pubmed.ncbi.nlm.nih.gov/40887080/  

    **Generative AI statement:**  
    When Streamlit documentation and online forums were unclear, generative AI tools (e.g. ChatGPT and Perplexity) were used in a limited and supportive manner to help interpret and understand debugging errors,
    particularly when transferring code from Google Colab to Streamlit.
    All problem-solving decisions, code implementation, and final outputs were performed and critically assessed by the project team.
    """)