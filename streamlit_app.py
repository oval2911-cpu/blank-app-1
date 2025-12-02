"""import streamlit as st

st.title("🎈 My new app")
st.write(
    "I now changed somethign"
)"""
#import libraries
import pandas as pd

#load dataset
data_heart = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

#see the dataset
print(data_heart.head())

#see shape (n of rows and columns) of the dataset
print(data_heart.shape)

#descriptive statistics on full dataset
print(data_heart.describe())

#look at all column names
print(data_heart.columns)

#summarize information on missing values per feature
print(data_heart.isna().sum())

print(data_heart.duplicated().sum()) #does not work

#--> No duplicates to handle