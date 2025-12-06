# -*- coding: utf-8 -*-
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

"""
st.title("My new app")
st.write(
    "I now changed somethign"
)"""


def render_plot(obj, figsize=(12, 8), *args, **kwargs):
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

##Identify main research question:
**Main RQ & subquestions:**
How accurately can a machine-learning model predict the occurrence of cardiovascular events (stroke, CHD, MI, or coronary insufficiency) using baseline patient characteristics from the Framingham dataset?
##Select rows and columns relevant to the research question:
"""

#We want to exclude the following columns (number in brackets is index):
# ANGINA (24), HOSPMI (25), MI_FCHD (26), TIME..(31-38)
data_heart_subset = data_heart.drop(columns = ['ANGINA', 'HOSPMI', 'MI_FCHD', 'CVD', 'PREVAP', 'PREVMI', 'PREVHYP', 'TIMECVD', 'TIMEMIFC', 'TIMEMI', 'TIMEAP', 'HDLC', 'LDLC', 'BPMEDS']) #drop these columns hdlc, ldlc only in period3 available, thus inappropriate to calculate risk
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
# INCIDENT event after Period 1
dhs['targetDisease'] = (
    (dhs['ANYCHD'] == 1) | (dhs['STROKE'] == 1)
).astype(int)

"""Data exploration (distributions and descriptive statistics)"""
selectedVariable = st.selectbox("Select variable to plot:", dhs.select_dtypes(include='number').columns)
render_plot(dhs[selectedVariable].hist, bins=30, alpha=0.4, edgecolor='black', label=selectedVariable)


"""plot distributions of variables that are relevant to describe population characteristics at baseline"""

render_plot(dhs.loc[dhs['PERIOD']==1].hist(figsize=(15, 10), bins=30, edgecolor='black'))


"""--> SYSBP, TOTCHOL, BMI, HEARTRTE, GLUCOSE, HEARTRTE seem to be right skwewed.

--> Other look normalfor what they are

--> Sex was encoded with 1 and 2 --> be careful

--> Age has some dips?
"""

"""##Outlier detection and handling, Impute only Period 1, to prevent data leakage"""

#allocate all categorical and numerical variables to corresponding separate variables
categorical = ['SEX', 'CURSMOKE', 'DIABETES',
                    'ANYCHD', 'STROKE', 'HYPERTEN', 'PERIOD', 'educ']

numerical = ['TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE', 'CIGPDAY', 'GLUCOSE']

#create 2 dataframes with different types of variables
num_df = dhs[numerical]
cat_df = dhs[categorical]

"""plot boxplots for numerical data to see outliers and dstributions"""
render_plot(sns.boxplot, data=dhs[numerical], orient='h')

"""--> most of the outliers are on the right hand side due to skewness"""

"""check for ranges of the heart rate"""
st.write("Highest heart rate:", dhs['HEARTRTE'].max())
st.write("Lowest heart rate:", dhs['HEARTRTE'].min())

#check for heartrate: how many values are beyond physiological limits (below 40 or above 180)
low = (dhs['HEARTRTE'] < 40).sum()
high = (dhs['HEARTRTE'] > 180).sum()
st.write("Low (<40):", low, "High (>180):", high)

st.write("Highest SYSBP:", dhs['SYSBP'].max())
st.write("Lowest SYSBP:", dhs['SYSBP'].min())

dhs['SYSBP'] = dhs['SYSBP'].clip(lower=70.0, upper=250.0)

st.write("Highest TOTCHOL:", dhs['TOTCHOL'].max())
st.write("Lowest TOTCHOL:", dhs['TOTCHOL'].min())

dhs['TOTCHOL'] = dhs['TOTCHOL'].clip(lower=70.0, upper=450.0)

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

##Missing data handling

"""check number of missing values in each column"""
st.write(dhs.isna().sum())

"""Mode Impute educ, since it is categorical"""
mode_value = dhs['educ'].mode()[0]
dhs['educ'] = dhs['educ'].fillna(mode_value)

"""Median Impute CIGPDA, to be more robust against outliers"""
median_cigpday = dhs['CIGPDAY'].median()
dhs['CIGPDAY'] = dhs['CIGPDAY'].fillna(median_cigpday)

"""Mean Impute BMI, since missing value is very low and distribution looks normal enough"""
mean_bmi = dhs['BMI'].mean()
dhs['BMI'] = dhs['BMI'].fillna(mean_bmi)


#MICE Imputation for remaining numerical variables as they are correlated and higher percentage missing

#dhs_imputed.isna().sum()

#dhs_imputed.to_csv('dhs_imputed_clean.csv', index=False)



##############################################################################
