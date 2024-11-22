import streamlit as st
import pandas as pd

st.title('Hormoniq')

st.info('This app can help you predict whether or not you have PCOS.')

df = pd.read_csv("https://raw.githubusercontent.com/its-bhavya/PCOS-Prediction/refs/heads/master/Final_PCOS_Dataset.csv")

X = df.drop("PCOS", axis = 1)
y = df.PCOS
#Follicle No. (L),Follicle No. (R),Avg. F size (L) (mm),Avg. F size (R) (mm)
Age = st.slider("How old are you?", 14, 50, 25)
Weight = st.slider("How much do you weigh (In Kg)", 25, 150, 60)
Height = st.slider("How tall are you? (In CM)", 100, 220, 155)
CycleLength = st.slider("How long does your period last on an average?", 2, 8, 5)
Hip = st.slider("Please enter your hip measurements (in Inches): ", 24, 56,  38)
Waist = st.slider("Please enter your waist measurements (in Inches): ",20, 56, 34)
CycleReg = st.radio("Is your menstrual cycle regular?", ("Yes", "No"))
WeightGain = st.radio("Have you experienced recent weight gain?", ("Yes", "No"))
HairGrowth = st.radio("Have you noticed excessive facial or body hair growth?", ("Yes", "No"))
SkinDarkening = st.radio("Have you experienced recent skin darkening in areas?", ("Yes", "No"))
HairLoss = st.radio("Have you experienced recent Hair Loss?", ("Yes", "No"))
Pimples = st.radio("Have you been struggling with acne or pimples?", ("Yes", "No"))
FastFood = st.radio("Do you regularly consume fast food? (More than three times a week)", ("Yes", "No"))
RegExercise = st.radio("Do you exercise regularly?", ("Yes", "No"))
BloodGroup = st.selectbox("What's your blood group?", ("A+","B+","AB+", "O+", "O-", "A-", "B-", "AB-"))
BMI = st.number_input("Please enter your BMI.", 24.55)
Pulse = st.number_input("Enter your pulse rate: ", min_value = 50, max_value = 100, value = 72, placeholder = "Enter here...")
Haemoglobin = st.number_input("Please enter your Haemoglobin Levels (in g/dL)", 1, 30, 11)
FSH = st.number_input("Enter your FSH levels (in mIU/mL): ", 12.5)
AMH = st.number_input("Enter your AMH levels (in ng/mL): ", 6)
LeftFollicleNumbers = st.number_input("Enter your Left Follicle Number: ", 6)
RightFollicleNumbers = st.number_input("Enter your Right Follicle Number: ", 6)
AvgRFollicleSize = st.number_input("Enter your average right follicle size: ", 15)
AvgLFollicleSize = st.number_input("Enter your average left follicle size:", 15)

  
  
