import streamlit as st
import pandas as pd

st.title('Hormoniq')

st.info('This app can help you predict whether or not you have PCOS.')

df = pd.read_csv("https://raw.githubusercontent.com/its-bhavya/PCOS-Prediction/refs/heads/master/Final_PCOS_Dataset.csv")

X = df.drop("PCOS", axis = 1)
y = df.PCOS
y
