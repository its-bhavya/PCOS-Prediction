import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn. tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


st.title('Ask PCOS 🌷')

st.info('This app uses Artificial Intelligence to help you predict whether or not you have PCOS.')

df = pd.read_csv("https://raw.githubusercontent.com/its-bhavya/PCOS-Prediction/refs/heads/master/Final_PCOS_Dataset.csv")

X = df.drop("PCOS", axis = 1)

y = df.PCOS

Age = st.slider("How old are you?", 14, 50, 25)
Weight = st.slider("How much do you weigh (In Kg)", 25, 150, 60)
Height = st.slider("How tall are you? (In CM)", 100, 220, 155)
CycleLength = st.slider("How long does your period last on an average?", 2, 8, 5)
Hip = st.slider("Please enter your hip measurements (in Inches): ", 24, 56,  38)
Waist = st.slider("Please enter your waist measurements (in Inches): ",20, 56, 34)
Pregnant = st.radio("Are you currently pregnant?", ("Yes", "No"))
CycleReg = st.radio("Is your menstrual cycle regular?", ("Yes", "No"))
WeightGain = st.radio("Have you experienced recent weight gain?", ("Yes", "No"))
HairGrowth = st.radio("Have you noticed excessive facial or body hair growth?", ("Yes", "No"))
SkinDarkening = st.radio("Have you experienced recent skin darkening in areas?", ("Yes", "No"))
HairLoss = st.radio("Have you experienced recent Hair Loss?", ("Yes", "No"))
Pimples = st.radio("Have you been struggling with acne or pimples?", ("Yes", "No"))
FastFood = st.radio("Do you regularly consume fast food? (More than three times a week)", ("Yes", "No"))
RegExercise = st.radio("Do you exercise regularly?", ("Yes", "No"))
BloodGroup = st.selectbox("What's your blood group?", ("A+","B+","AB+", "O+", "O-", "A-", "B-", "AB-"))
BMI = st.number_input("Please enter your BMI.", value = 24.55, step = 1.00)
Pulse = st.number_input("Enter your pulse rate: ", min_value = 50, max_value = 100, value = 72, step =1)
Haemoglobin = st.number_input("Please enter your Haemoglobin Levels (in g/dL)", min_value = 1.00, max_value = 30.00, value = 11.00, step = 1.00)
FSH = st.number_input("Enter your FSH levels (in mIU/mL): ", value = 12.50, step =1.00)
AMH = st.number_input("Enter your AMH levels (in ng/mL): ", value = 6.00, step =1.00)
LeftFollicleNumbers = st.number_input("Enter your Left Follicle Number: ", value = 6)
RightFollicleNumbers = st.number_input("Enter your Right Follicle Number: ", value = 6)
AvgRFollicleSize = st.number_input("Enter your average right follicle size: ", value = 15.00, step =1.00)
AvgLFollicleSize = st.number_input("Enter your average left follicle size:", value = 15.00, step =1.00)

#Creating a dataframe for the input features
data = {' Age (yrs)': Age,
        'Weight (Kg)': Weight,
        'Height(Cm) ': Height,
        'BMI': BMI,
        'Pulse rate(bpm) ': Pulse,
        'Hb(g/dl)': Haemoglobin,
        'Cycle length(days)': CycleLength,
        'FSH(mIU/mL)': FSH,
        'Hip(inch)': Hip,
        'Waist(inch)': Waist,
        'AMH(ng/mL)': AMH,
        'Follicle No. (L)': LeftFollicleNumbers,
        'Follicle No. (R)':RightFollicleNumbers,
        'Avg. F size (L) (mm)': AvgLFollicleSize,
        'Avg. F size (R) (mm)': AvgRFollicleSize,
        'Blood Group': BloodGroup,
        'Cycle(R/I)': CycleReg,
        'Pregnant(Y/N)': Pregnant,
        'Weight gain(Y/N)': WeightGain,
        'hair growth(Y/N)': HairGrowth,
        'Skin darkening (Y/N)': SkinDarkening,
        'Hair loss(Y/N)': HairLoss,
        'Pimples(Y/N)': Pimples,
        'Fast food (Y/N)': FastFood,
        'Reg.Exercise(Y/N)': RegExercise}

input_df = pd.DataFrame(data, index = [0])


#Encoding our input features
encode = ['Cycle(R/I)', 'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)',  'Skin darkening (Y/N)',
          'Hair loss(Y/N)', 'Pimples(Y/N)','Fast food (Y/N)', 'Reg.Exercise(Y/N)']

binary_mapping = {"Yes": 1, "No": 0}

for col in encode:
    input_df[col] = input_df[col].replace(binary_mapping)

blood_group_mapping = {
    'A+':11,
    'A-':12,
    'B+':13,
    'B-':14,
    'O+':15,
    'O-':16,
    'AB+':17,
    'AB-':18
}

input_df['Blood Group'] = input_df['Blood Group'].replace(blood_group_mapping)

input_PCOS = pd.concat([input_df, X], axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = joblib.load("decision_tree_model.pkl")

if st.button('Predict'):
    prediction = model.predict(input_df)  # Use the model to make a prediction
    st.write(f'{"You might have PCOS, kindly consult a doctor." if prediction[0] == 1 else "No, you don't have PCOS. But always consult a doctor for proper diagnosis."}')

