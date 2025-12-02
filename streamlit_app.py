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
data_heart.head()