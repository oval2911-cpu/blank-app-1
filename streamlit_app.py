#import libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.axes
import matplotlib.figure
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler



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






st.title("Prediction of the cardiovascular events using the baseline patient characteristics from the Framingham dataset")

st.markdown("*Tom Einhaus: i6339207, Alisa Ovsiannikova: i6365923*")
st.markdown("*MAI3002: Introduction to Programming*")
st.markdown("*Faculty of Health, Medicine, and Life Sciences*")
st.markdown("*Maastricht University*")
st.markdown("***December 16th, 2025***")


def render_plot(obj, title="", *args, **kwargs):
    """
    Universal Streamlit plot wrapper.
    
    Accepts:
    - Figure objects
    - Axes objects
    - Array of Axes (e.g., df.hist())
    - Callable plotting functions
    - pandas Series (automatically plotted as histogram)
    
    Automatically:
    - Creates figure if needed
    - Applies layout fixes
    - Displays the figure in Streamlit
    """
    figsize=(12, 8)
    # ----------------------------
    # CASE 0 — pandas Series
    # ----------------------------
    if isinstance(obj, pd.Series):
        fig = obj.plot(kind='hist', figsize=figsize, **kwargs).get_figure()
    
    # ----------------------------
    # CASE 1 — Callable
    # ----------------------------
    elif callable(obj):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        obj(*args, **kwargs)
    
    # ----------------------------
    # CASE 2 — Figure
    # ----------------------------
    elif isinstance(obj, matplotlib.figure.Figure):
        fig = obj
    
    # ----------------------------
    # CASE 3 — Single Axes
    # ----------------------------
    elif isinstance(obj, matplotlib.axes.Axes):
        fig = obj.get_figure()
    
    # ----------------------------
    # CASE 4 — Array of Axes (e.g., df.hist())
    # ----------------------------
    elif isinstance(obj, (np.ndarray, list)) and all(isinstance(a, matplotlib.axes.Axes) for a in np.ravel(obj)):
        axes = np.ravel(obj)
        fig = axes[0].get_figure()
    
    else:
        raise TypeError(f"render_plot received an unsupported type: {type(obj)}")
    
    # ----------------------------
    # Layout adjustments
    # ----------------------------
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title, fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    for ax in fig.axes:
        ax.tick_params(axis='x', rotation=30)
    
    # ----------------------------
    # Display in Streamlit
    # ----------------------------
    st.pyplot(fig)
    
    return fig

#load dataset
data_heart = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

"""Prints Dataset Head"""
data_heart

"""--> each row represents one exam, so there are more rows than number of patients (RANDIDs repeat)

--> can already expect some missing values
"""

"""see shape (n of rows and columns) of the dataset"""
st.write(data_heart.shape)

"""descriptive statistics on full dataset"""
st.write(data_heart.describe())

"""look at all column names"""
st.write(data_heart.columns)

"""summarize information on missing values per feature"""
st.write(data_heart.isna().sum())

"""--> Missing values have to be handled"""

st.write(data_heart.duplicated().sum())

"""--> No duplicates to handle

Identify main research question:
**Main RQ & subquestions:**
How accurately can a machine-learning model predict the occurrence of cardiovascular events (stroke, CHD, MI, or coronary insufficiency) using baseline patient characteristics from the Framingham dataset?
##Select rows and columns relevant to the research question:
"""

#We want to exclude the following columns (number in brackets is index):
# ANGINA (24), HOSPMI (25), MI_FCHD (26), TIME..(31-38)
data_heart_subset = data_heart.drop(columns = ['ANGINA', 'HOSPMI', 'MI_FCHD', 'CVD', 'PREVAP', 'PREVMI', 'TIMECVD', 'TIMEMIFC', 'TIMEMI', 'TIMEAP', 'HDLC', 'LDLC', 'BPMEDS']) #drop these columns hdlc, ldlc only in period3 available, thus inappropriate to calculate risk
#                                    + [col for col in data_heart.columns if col.startswith('PREV')]) #drop columns that start with PREV... when iterating over all columns
#                                    + [col for col in data_heart.columns if col.startswith('TIME')]) #drop columns that start with TIME... when iterating over all columns


#rename subset to make it easier to call it
dhs = data_heart_subset

"""check all columns of the created subset"""
st.write(dhs.columns)

"""see the subset"""
st.write(dhs.head(100))

#check the shape of the subset
st.write(dhs.shape)

# 1. Keep only Period 1 records for each participant to define baseline 'at-risk' population.
dhs = dhs.loc[dhs['PERIOD'] == 1].copy()

# 2. Filter out participants with prevalent disease (PREVCHD=1 or PREVSTRK=1) at Period 1.
# This ensures we are left with the population 'at risk' for a first event.
dhs = dhs.loc[
    (dhs['PREVCHD'] == 0) & (dhs['PREVSTRK'] == 0)
].copy()

# 3. Calculate the Incident Target Variable.
# For this 'at-risk' group, ANYCHD=1 or STROKE=1 in the Period 1 record indicates an
# INCIDENT event after Period 1, because the 1 is filled in each period
dhs['targetDisease'] = (
    (dhs['ANYCHD'] == 1) | (dhs['STROKE'] == 1)
).astype(int)

"""Data exploration (distributions and descriptive statistics)"""
selectedVariable = st.selectbox("Select variable to plot:", dhs.select_dtypes(include='number').columns)

render_plot(dhs[selectedVariable].hist, bins=30, alpha=0.4, edgecolor='black', label=selectedVariable)


"""plot distributions of variables that are relevant to describe population characteristics at baseline"""

render_plot(dhs.hist(figsize=(15, 10), bins=30, edgecolor='black'))


"""--> SYSBP, TOTCHOL, BMI, HEARTRTE, GLUCOSE seem to be right skwewed.

--> Other look normal for what they are

--> Sex was encoded with 1 and 2 - keep in mind

--> Age has some dips?
"""

"""One Hot Encoding for SEX variable"""
# Create the new binary column
dhs['SEX_Female'] = dhs['SEX'].replace({
    1: 0, # Map original value 1 (Male) to 0
    2: 1  # Map original value 2 (Female) to 1
})


# Drop the patients that never developed chd or stroke and that died during the follow-up because if they died we cannot say that they would have not developed a disease
dhs = dhs.loc[~((dhs['targetDisease'] == 0) & (dhs['DEATH'] == 1))]


# Drop the original sex column, original target columns, and other columns that are not needed
dhs = dhs.drop('SEX', axis=1)
dhs = dhs.drop('ANYCHD', axis=1)
dhs = dhs.drop('STROKE', axis=1)
dhs = dhs.drop('PREVCHD', axis=1)
dhs = dhs.drop('PREVSTRK', axis=1)
dhs = dhs.drop('TIME', axis=1)
dhs = dhs.drop('PERIOD', axis=1)
dhs = dhs.drop('HYPERTEN', axis=1)
dhs = dhs.drop('TIMEHYP', axis=1)
dhs = dhs.drop('DEATH', axis=1)
dhs = dhs.drop('TIMECHD', axis=1)
dhs = dhs.drop('TIMESTRK', axis=1)
dhs = dhs.drop('TIMEDTH', axis=1)


"""##Outlier detection and handling, Impute only Period 1, and only training set to prevent data leakage"""

X = dhs.drop('targetDisease', axis=1) 
y = dhs['targetDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=1,
    stratify=y 
)

#allocate all categorical and numerical variables to corresponding separate variables
categorical = ['SEX_Female', 'CURSMOKE', 'DIABETES', 'educ']

numerical = ['TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE', 'CIGPDAY', 'GLUCOSE', 'AGE']

#create 2 dataframes with different types of variables
num_df = X_train[numerical]
cat_df = X_train[categorical]

"""plot boxplots for numerical data to see outliers and dstributions"""
render_plot(sns.boxplot, data=X_train[numerical], orient='h')

"""--> most of the outliers are on the right hand side due to skewness"""

##Missing data handling

"""check number of missing values in each column"""
st.write(X_train.isna().sum())

"""Mode Impute educ, since it is categorical"""
mode_value = X_train['educ'].mode()[0]
X_train['educ'] = X_train['educ'].fillna(mode_value)
X_test['educ'] = X_test['educ'].fillna(mode_value)

"""Median Impute CIGPDA, to be more robust against outliers"""
median_cigpday = X_train['CIGPDAY'].median()
X_train['CIGPDAY'] = X_train['CIGPDAY'].fillna(median_cigpday)
X_test['CIGPDAY'] = X_test['CIGPDAY'].fillna(median_cigpday)

"""Median Impute BMI and Heartrate, to be more robust against outliers"""
median_bmi = X_train['BMI'].median()
X_train['BMI'] = X_train['BMI'].fillna(median_bmi)
X_test['BMI'] = X_test['BMI'].fillna(median_bmi)

median_heartrate = X_train['HEARTRTE'].median()
X_train['HEARTRTE'] = X_train['HEARTRTE'].fillna(median_heartrate)
X_test['HEARTRTE'] = X_test['HEARTRTE'].fillna(median_heartrate)

"""check if there are many values outside whiskers (1.5*IQR) - outlier detection"""
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)
st.write(outlier_counts)

"""check if there are many values above z score of 3 (another way of detecting outliers)"""
from scipy.stats import zscore
z_scores = num_df.apply(zscore)
outliers_z = (abs(z_scores) > 3).sum().sort_values(ascending=False)
st.write(outliers_z)

#How much influence do these outliers have on mean and std when being included or disregarded?

#without outliers
clean_df = num_df[~outlier_mask.any(axis=1)]

#st.write mean and std for with outliers vs without outliers and put in pd dataframe
compare = pd.DataFrame({
    'Mean (all)': num_df.mean(),
    'Mean (no outliers)': clean_df.mean(),
    'Std (all)': num_df.std(),
    'Std (no outliers)': clean_df.std()
})
st.write(compare.round(2))

"""check for ranges of the heart rate"""
st.write("Highest heart rate:", X_train['HEARTRTE'].max())
st.write("Lowest heart rate:", X_train['HEARTRTE'].min())

#check for heartrate: how many values are beyond physiological limits (below 40 or above 180)
low = (X_train['HEARTRTE'] < 40).sum()
high = (X_train['HEARTRTE'] > 180).sum()
st.write("Low (<40):", low, "High (>180):", high)

st.write("Highest SYSBP:", X_train['SYSBP'].max())
st.write("Lowest SYSBP:", X_train['SYSBP'].min())

X_train['SYSBP'] = X_train['SYSBP'].clip(lower=70.0, upper=250.0)

st.write("Highest TOTCHOL:", X_train['TOTCHOL'].max())
st.write("Lowest TOTCHOL:", X_train['TOTCHOL'].min())

X_train['TOTCHOL'] = X_train['TOTCHOL'].clip(lower=70.0, upper=450.0)

#MICE Imputation for TOTCHOl / GLUCOSE as they are correlated and higher percentage missing
# (doing after outlier and other missing values imputation bc totchol/glucose imputation depends on them)

# PART A: FIT and TRANSFORM X_TRAIN 
# 1. Define the set of columns for the MICE model:
# Both Imputation Targets (TOTCHOL, GLUCOSE) and all Predictors
mice_cols = [
    'TOTCHOL', 'GLUCOSE', 'AGE', 'SEX_Female', 'SYSBP', 'DIABP', 'BMI',
    'educ', 'CIGPDAY', 'HEARTRTE'
]

# Create the temporary DataFrame X_train_temp, preserving the original index
X_train_temp = X_train[mice_cols].copy() 

# 2. Initialize and FIT the MICE model ONLY on the training data
# This creates the imputation model (imp) that we will reuse.
imp = IterativeImputer(max_iter=10, random_state=42)
X_imputed_train_array = imp.fit_transform(X_train_temp) 

# 3. CORRECTED STEP: Convert back to DataFrame using the original index
X_imputed_train = pd.DataFrame(
    X_imputed_train_array, 
    index=X_train_temp.index, # Index Alignment
    columns=mice_cols
)

# 4. Update the original DataFrame (X_train)
X_train['TOTCHOL'] = X_imputed_train['TOTCHOL']
X_train['GLUCOSE'] = X_imputed_train['GLUCOSE']

print("X_train successfully imputed.")

# PART B: TRANSFORM X_TEST (LEAKAGE-FREE)

# 1. Create the temporary DataFrame X_test_temp
X_test_temp = X_test[mice_cols].copy() 

# 2. TRANSFORM the test set (using the SAME 'imp' model fitted on X_train)
X_imputed_test_array = imp.transform(X_test_temp) # No 'fit' here!

# 3. Convert back to DataFrame using the original index
X_imputed_test = pd.DataFrame(
    X_imputed_test_array, 
    index=X_test_temp.index, # Index Alignment
    columns=mice_cols
)

# 4. Update the original DataFrame (X_test)
X_test['TOTCHOL'] = X_imputed_test['TOTCHOL']
X_test['GLUCOSE'] = X_imputed_test['GLUCOSE']

st.write(X_train.isna().sum())
st.write(X_test.isna().sum())

# X_train and X_test are now fully imputed and cleaned for further modeling.
# Now transformations against skewness can be applied if needed.

skewness_results = X_train[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

st.write("--- Skewness Check (Calculated on X_train) ---")
st.write(skewness_results)

# From the skewness results we can identify these variables as highly skewed:
log_transform_cols = ['GLUCOSE', 'CIGPDAY', 'SYSBP', 'BMI']

# --- Transform X_train ---
for col in log_transform_cols:
    # Overwrite the original column with the transformed values
    X_train[col] = np.log1p(X_train[col])

# --- Transform X_test ---
# Apply the EXACT SAME transformation to X_test (leakage-free)
for col in log_transform_cols:
    # Overwrite the original column with the transformed values
    X_test[col] = np.log1p(X_test[col])

# now we have imputed and transformed datasets X_train and X_test ready for scaling

# since we already handled outliers we can use standard scaling

# 1. Initialize the Scaler
scaler = StandardScaler()

# 2. FIT the Scaler ONLY on the X_train data (Leakage Prevention)
# This calculates the mean and standard deviation of X_train
scaler.fit(X_train[numerical])

# 3. TRANSFORM the X_train data
# X_train is updated in place
X_train[numerical] = scaler.transform(X_train[numerical])

# 4. TRANSFORM the X_test data (using the X_train fitted scaler)
# X_test is updated in place
X_test[numerical] = scaler.transform(X_test[numerical])



"""Final Data exploration (distributions and descriptive statistics)"""

# Make a on/off button to select train/test set
on = st.toggle('Turn on to see test set')
if on:
    dataset = X_test
    dataset_name = "X_test set"
    
else:
    dataset = X_train
    dataset_name = "X_train set"

# 1. Select a variable from the columns of the training set (X_train)

selectedVariable = st.selectbox("Select variable to plot:", dataset.columns)
# 2. Plot the histogram using the training data for the selected variable
render_plot(dataset[selectedVariable].hist, f'{dataset_name}', bins=30, alpha=0.4, edgecolor='black', label=selectedVariable)

st.write(X_train.describe())

# Print the size and outcome distribution of the training set

st.write(f"Total Training Samples: {X_train.shape[0]}")
outcome_distribution = y_train.value_counts(normalize=True) * 100
st.write("\nTarget Disease Distribution (y_train):")
st.write(outcome_distribution.round(2))

st.subheader("Categorical Feature Distributions")

# Use a visualization (example using Streamlit/render_plot)
render_plot(y_train.value_counts().plot, kind='pie', autopct='%1.1f%%', 
            title='Target Disease Distribution (y_train)')

# 2. Gender Distribution (Bar Chart)
st.markdown("##### SEX_Female Counts (0=Male, 1=Female)")
render_plot(X_train['SEX_Female'].value_counts().plot, 
            kind='bar', 
            alpha=0.8, 
            edgecolor='black')
##############################################################################

"""Limitations: 
- Patients were followed-up in incomparable timespans (e.g. 1 patient 2 years, another 6 years) --> can bias predictions"""