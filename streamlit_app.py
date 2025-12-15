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


# title and subtitle for the app
st.title("Prediction of the cardiovascular events using the baseline patient characteristics from the Framingham dataset")

st.markdown("""*Tom Einhaus: i6339207, Alisa Ovsiannikova: i6365923*  
*MAI3002: Introduction to Programming*  
*Faculty of Health, Medicine, and Life Sciences*  
*Maastricht University*

***December 16th, 2025***""")


# contents and anchor links to different sections (headers) in the sidebar and the page itself

st.write("## Contents:")

st.sidebar.markdown("[1. Background and Research Question](#background-and-research-question)")
st.sidebar.markdown("[2. Data Preparation](#data-preparation)")
st.sidebar.markdown("[3. Exploratory Data Analysis (EDA), Cleaning, and Feature Engineering](#exploratory-data-analysis-eda-leaning-and-feature-ngineering)") 
st.sidebar.markdown("[4. Visualization of the Final Clean Data](#visualization-of-the-final-clean-data)")
st.sidebar.markdown("[5. ML Models Training and Prediction Evaluation](#ml-models-training-and-prediction-evaluation)") 
st.sidebar.markdown("[6. Comparing ML Models](#comparing-ml-models)")
st.sidebar.markdown("[7. Conclusion](#conclusion)")
st.sidebar.markdown("[8. References](#references)")

st.markdown("[1. Background and Research Question](#background-and-research-question)")
st.markdown("[2. Data Preparation](#data-preparation)")
st.markdown("[3. Exploratory Data Analysis (EDA), Cleaning, and Feature Engineering](#exploratory-data-analysis-eda-leaning-and-feature-ngineering)") 
st.markdown("[4. Visualization of the Final Clean Data](#visualization-of-the-final-clean-data)")
st.markdown("[5. ML Models Training and Prediction Evaluation](#ml-models-training-and-prediction-evaluation)") 
st.markdown("[6. Comparing ML Models](#comparing-ml-models)")
st.markdown("[7. Conclusion](#conclusion)")
st.markdown("[8. References](#references)")


# section 1
st.header("1. Background and Research Question") #background on the dataset and our RQ

st.write("### Framingham Heart Study Dataset:")
st.write("""- ***Extensive follow-up study** dataset on **cardiovascular health** ongoing **since 1948***
- *Up to **3 follow-up periods** (fewer for some patients): **Period 1 = baseline***
- ***11,627 examination records** (rows) from **4,434 patients***
- ***39 variables:***
    - *Demographics (sex, age)*
    - *Clinical health data (blood pressure, diabetes)*
    - *Lifestyle (smoking, BMI)*
    - *Occurrence of cardiovascular diseases (stroke, coronary heart disease (CHD))*""")

st.write("### Research Question:")
st.write("***To what extent** can **baseline patient characteristics from the Framingham dataset** be used by machine-learning models to **reliably predict the occurrence of major cardiovascular events** (stroke, CHD, myocardial infarction (MI), and coronary insufficiency)?*")

st.write("### Previous Research Findings:")
st.write("*Previously reported research shows BMI, cholesterol and blood pressure can have an influence on CVD risk (Bays et al., 2021): → Which modifiable factors are correlated with CVD the strongest, and are thus worth minimizing?*")


# section 2
st.header("2. Data Preparation")  #exploration of raw dataset and subsetting

st.write('### Raw Dataset Preview:')
# load dataset
data_heart = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
data_heart
st.write(f'**Shape** (n of rows and columns) of the raw dataset: :blue-background[{data_heart.shape}]')
st.write("""
- *Each **row represents one exam**, so there are **more rows than number of patients** (RANDIDs repeat)*
- *Some cells contain 'None', therefore, **some missing values can already be expected and will have to be handled***
""")

st.write('### Raw Dataset Descriptive Statistics:')
st.write(data_heart.describe())
st.write("""
- ***RANDID variable is not interesting**, therefore, it **will be removed** in the following steps*
- *Interestingly, **sex was encoded with 1 and 2** so we should **be careful** and encode it to 0 and 1 before applying ML, to preven learning hierachy*
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
        - *Keep **only Period 1 records** for each participant to define baseline 'at-risk' population*
        - ***Filter out participants with prevalent disease** (PREVCHD=1, PREVSTRK=1) at Period 1 to ensure we are keeping only the population 'at risk' for a first event*
        - ***Filter out the patients that never developed a disease and died during the follow-up** as if they died we cannot assume that they would have not developed a disease later*
    2. ***Including** only 14 **variables of interest:***
        - *Demographics: **SEX, AGE, educ***
        - *Clinical health data: **TOTCHOL, SYSBP, DIABP, DIABETES, HEARTRTE, GLUCOSE***
        - *Lifestyle: **CURSMOKE, CIGPDAY, BMI***
        - *Occurrence of cardiovascular diseases: **ANYCHD, STROKE***
    - *The following 25 variables will therefore be **excluded:***
        - ***RANDID, PERIOD, PREVCHD, PREVSTRK, TIME, DEATH**: do not contain any valuable information as the data were already filtered*
        - ***ANGINA, PREVAP, TIMEAP**: not looking into that disease*
        - ***HOSPMI, MI_FCHD, CVD, PREVMI, TIMECVD, TIMEMIFC, TIMEMI**: included in other variables that will be left in the subset (ANYCHD, STROKE, PREVCHD, PREVSTRK, TIMECHD, TIMESTRK)*
        - ***HDLC, LDLC**: only available in Period 3 while we want to look into the baseline (Period 1)*
        - ***PREVHYP, HYPERTEN, TIMECHD, TIMESTRK, TIMEDTH, TIMEHYP, BPMEDS**: contain information that would create data leakage when using ML*
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
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
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
- *Before data cleaning, the data should be split into train and test to avoid data leakage.
Thus, the target variable is required for which we currently have 2 variables (ANYCHD, STROKE).
In the Framingham dataset, ANYCHD=1 and STROKE=1 in the Period 1 record indicate that the
event happened at any time during the follow-up after Period 1.
Therefore, we can create one target variable for binary classification that would allow us to do train/test split.*
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
"""Now on this new dataset, with a target variable, we can proceed to cleaning"""


st.write("### Outlier Detection and Handling")
st.write("It is important to only act on Training Set, and only use baseline features to prevent data leakage.")
st.write("A quick boxplot gives us an overview over the outliers of our data:")
render_plot(sns.boxplot, data=X_train[numerical], orient='h')
st.write("""
- *Most of the outliers are on the right hand side due to skewness*
- *Age has no outliers*
- *Outermost physiological limits are: TOTCHOL (50-500 mg/dL), SYSBP (40-350 mmHg), GLUCOSE (20-1000 mg/dL)*
- *Comparing those limits to our outliers, all are theoretically possible due to extreme medical conditions, therefore, decided to keep them*
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
- *5 Variables: educ, TOTCHOL, GLUCOSE, CIGPDAY, and BMI have missing values*
""")
st.write("""
- *GLUCOSE has >5% missing, which automatically makes it a candidate for model based imputation*
""")

st.write("""Another visualization of missing data is the missingno matrix, which shows percentage and absolute number of present data:""")
render_plot(msno.bar(X_train).figure)

st.write("""GLUCOSE will be model based imputed because of its high missingness.
         Another reason for model based imputation is if type of missingness is MAR (Missing at Random)
         
- *To check if variables are correlated with missingness of GLUCOSE and TOTCHOL, correlation heatmap and missingno heatmap are plotted below*
""")

#add correlations between variables
render_plot(sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt='.2f'))

st.write("""
Missingno heatmap relates missingness of variables to each other.""")

render_plot(msno.heatmap(X_train).figure)

st.write("""
- *You can see that TOTCHOL has slight correlations with missingness in BMI and GLUCOSE*
- *This makes it a candidate for model based imputation as well, as its missingness may depend on other variables*""")


#render_plot(sns.pairplot(X_train[['AGE','TOTCHOL','SYSBP','DIABP','GLUCOSE','BMI','HEARTRTE']],corner=True,diag_kind='hist').fig)


#st.write("""
#- *From the correlation heatmap we can see that TOTCHOL is MAR, and should thus also be model based imputed.*
#""")

st.write("""
- For all other, as the missingness is less than 5% in each variable, we will do simple imputation*
- *educ is categorical, therefore, will use mode for imputation*
- *For CIGPDAY and BMI will use median imputation to be more robust against outliers*
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

st.write("""
- *We will use MICE (Multiple Imputation by Chained Equations) for imputation of TOTCHOL and GLUCOSE.*""")

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
- *X_train and X_test are now fully imputed and cleaned for further modeling*
""")


#add also checking distributions before and after imputation - should not change much


st.write("### Apply transformations against skewness")
st.write("Check skewness (calculated on X_train):")
skewness_results = X_train[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
st.write(skewness_results)
st.write("""
- *From the skewness results we can identify these variables as highly skewed: GLUCOSE, CIGPDAY, SYSBP, BMI*
- *Log transformation will be applied to them*
- *Check distributions after log tranformation:*
""")
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
- *Distributions are symmetric, therefore, the transformation worked*
""")


st.write("### One Hot Encoding for SEX variable")
st.write("""*As the SEX variable was encoded with 1 and 2, we map original value 1 (Male) to 0
and value 2 (Female) to 1.*""")
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
st.write("""*Since we have already handled outliers, we can use standard scaling to prepare data for modeling.*""")
st.write("""*We applied StandardScaler calculate on X_train, and transformed X_train and X_test using this scaling to prevent data leakage*""")

scaler = StandardScaler() #since we already handled outliers we can use standard scaling
scaler.fit(X_train[numerical]) #fit only on train data (leakage prevention), calculating mean and standard deviation of X_train
X_train[numerical] = scaler.transform(X_train[numerical]) #transform train data
X_test[numerical] = scaler.transform(X_test[numerical]) #transform test data

#Do the same on the raw data copies for later visualization comparison
scaler.fit(rawX_train[numerical]) #fit only on train data (leakage prevention), calculating mean and standard deviation of X_train
rawX_train[numerical] = scaler.transform(rawX_train[numerical]) #transform train data
rawX_test[numerical] = scaler.transform(rawX_test[numerical]) #transform test data
st.write("Data preparation is now complete, and the data is ready for modeling.")

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

overlap_raw = st.toggle("Overlap RAW")
selectedVariable = st.selectbox("Select variable to plot:", dataset.columns)

if overlap_raw:
    def plot_overlaid():
        # Raw (red) with KDE
        sns.histplot(raw_dataset[selectedVariable], bins=30, kde=True, 
                    color='red', alpha=0.6, stat='density', label='Raw')
        # Cleaned (blue) with KDE  
        sns.histplot(dataset[selectedVariable], bins=30, kde=True, 
                    color='blue', alpha=0.6, stat='density', label='Cleaned')
        plt.legend()
    
    render_plot(plot_overlaid, title=f"{dataset_name} vs Raw - {selectedVariable}")
    st.write(f"Total training samples (X_train): :blue-background[{X_train.shape[0]}]")
else:
    def plot_single():
        # Single distribution with stats
        sns.histplot(dataset[selectedVariable], bins=30, kde=True, 
                    color='teal', alpha=0.6, stat='density')
        plt.legend()
    
    render_plot(plot_single, title=f"{dataset_name} - {selectedVariable}")
    st.write(f"Total training samples (X_train): :blue-background[{X_train.shape[0]}]")

render_plot(y_train.value_counts().plot, kind='pie', autopct='%1.1f%%', title='Training Target Disease Distribution (y_train)')
st.write("""
- *The target distribution indicates imbalanced dataset, thus, it should be kept in mind during the ML models training.*
""")

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

    # visualization (only for decision tree)
    if model_name == "Decision Tree":
        with st.expander("Decision Tree Visualization"):
            fig_dt, ax_dt = plt.subplots(figsize=(16,10))
            tree.plot_tree(model, ax=ax_dt);
            st.pyplot(fig_dt)

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
            cv_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv_value)
            prediction_cv = cross_val_predict(model, X_test, y_test, cv=cv_value)
            st.write(f'Cross-validation accuracy is {cv_scores.mean().round(2)} with a standard deviation of {cv_scores.std().round(2)}')
            # confusion matrix
            cm_cv = confusion_matrix(y_true=y_test, normalize='true', y_pred=prediction_cv)
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
                         value = 10, #chosen as default as the best found
                         key = 'max_depth_rf')
min_samples_leaf_rf = st.slider('Specify minimum number ' \
                                'of samples in a leaf node ' \
                                'after splitting to avoid overfitting:', 1, X_train.shape[0] // 2, 20,
                                key = 'min_samples_leaf_rf')        
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
                            value = int((X_train.shape[0])**0.5), #chosen as default as the best found 
                            step = 2) #only odd numbers to avoid ties in a classification majority vote
weights_KNN = st.selectbox('Select weight function:', 
                           ['uniform', 'distance', None]) 
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



st.header("6. Comparing ML Models") #selecting the best model out of 4 fine-tuned models
#table with best fine-tuned models comparison
# and explanation why we think which one is better
"Algorithm choice and what to mention when comparing them:"
"For example, do we need to explain how the algorithm's"
"choice was made? Why is the algorithm advising this treatment for you?"
"(whitebox vs blackbox algorithms)"
"Or perhaps some algorithms just take too much time to train,"
"even with todays computational power"

st.header("7. Conclusion")

"""Limitations: 
- Patients were followed-up in incomparable timespans (e.g. 1 patient 2 years, another 6 years) --> can bias predictions
- Maybe should have also feature engineered blood pressure"""
"""Answer on RQ based on ML, feature importance"""

st.header("8. References") #include genAI statement


st.markdown("*Streamlit* is **really** ***cool***.")
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')

multi = '''If you end a line with two spaces,
a soft return is used for the next line.

st.markdown("This is black and :red[this is red!]")

Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)