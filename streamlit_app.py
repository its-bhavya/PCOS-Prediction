import streamlit as st
import pandas as pd

st.title('Hormoniq')

st.info('This app can help you predict whether or not you have PCOS.')

df = pd.read_csv("https://raw.githubusercontent.com/its-bhavya/PCOS-Prediction/refs/heads/master/Final_PCOS_Dataset.csv")

X = df.drop("PCOS", axis = 1)
y = df.PCOS

with st.sidebar:
  st.header("Input Features")
  #Height(Cm) ,BMI,Pulse rate(bpm) ,Hb(g/dl),Cycle length(days),FSH(mIU/mL),Hip(inch),
  #Waist(inch),AMH(ng/mL),Follicle No. (L),Follicle No. (R),Avg. F size (L) (mm),Avg. F size (R) (mm),Blood Group,
  #Cycle(R/I),Pregnant(Y/N),Weight gain(Y/N),hair growth(Y/N),Skin darkening (Y/N),Hair loss(Y/N),Pimples(Y/N),
  Fast food (Y/N),Reg.Exercise(Y/N)
  Age = st.slider("How old are you?", 14, 50, 25)
