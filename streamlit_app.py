import streamlit as st
import pandas as pd

st.title('Hormoniq')

st.info('This app can help you predict whether or not you have PCOS.')

df = pd.read_csv("https://raw.githubusercontent.com/its-bhavya/PCOS-Prediction/refs/heads/master/Final_PCOS_Dataset.csv")

X = df.drop("PCOS", axis = 1)
y = df.PCOS

with st.sidebar:
  st.header("Input Features")
  #Pulse rate(bpm) ,Hb(g/dl),Cycle length(days),FSH(mIU/mL),Hip(inch),
  #Waist(inch),AMH(ng/mL),Follicle No. (L),Follicle No. (R),Avg. F size (L) (mm),Avg. F size (R) (mm)
  #Pregnant(Y/N),hair growth(Y/N),Skin darkening (Y/N),Hair loss(Y/N),Pimples(Y/N),
  #Fast food (Y/N),Reg.Exercise(Y/N)
  Age = st.slider("How old are you?", 14, 50, 25)
  Weight = st.slider("How much do you weigh (In Kg)", 25, 150, 60)
  Height = st.slider("How tall are you? (In CM)", 100, 220, 155)
  BMI = st.text_input("Please enter your BMI.", 24.55)
  BloodGroup = st.selectbox("What's your blood group?", ("A+","B+","AB+", "O+", "O-", "A-", "B-", "AB-"))
  CycleReg = st.radio("Is your menstrual cycle regular?", ("Yes", "No"))
  WeightGain = st.radio("Have you experienced recent weight gain?", ("Yes", "No"))
  HairGrowth = st.radio("Have you noticed excessive facial or body hair growth?", ("Yes", "No"))
  
