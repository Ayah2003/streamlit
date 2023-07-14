#/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# create streamlit interface, asome info about the app
st.write("""
         ## In few seconds, you can calculate your risk of developing heart disease!
      the app is built using a machine learning algorithm called Random Forest, with an accuracy of 88%.  
         """)
st.write("""
         ### To predict your heart disease risk:
         ###### 1- Enter the parameters that best describe you.
         ###### 2- Press the "Predict" button and wait for the result.
         """)


# In[9]:


# Thesidebar function from streamlit is used to create a sidebar for users to input their information.

st.sidebar.title('Please, fill in your information to predict your heart condition')

gender = st.sidebar.selectbox("Select your gender", ("", "Female", "Male"))
age = st.sidebar.selectbox("Select your age", [""] + list(map(str, range(1, 111))))
cigsPerDay = st.sidebar.selectbox("How many cigarettes do you smoke a day?", [""] + list(map(str, range(101))))

BPMeds = st.sidebar.selectbox("Are you on blood pressure medication?", ["", "No", "Yes"])

prevalentStroke = st.sidebar.selectbox("Did you have a stroke?", ["", "No", "Yes"])

prevalentHypertension = st.sidebar.selectbox("Do you have hypertension?", ["", "No", "Yes"])
diabetes = st.sidebar.selectbox("Do you have diabetes?", ["", "No", "Yes"])
totalCholesterolLevel = st.sidebar.selectbox("Enter your cholesterol level", [""] + list(map(str, range(1001))))
systolicBP = st.sidebar.selectbox("Enter your systolic blood pressure (mm Hg)", [""] + list(map(str, range(401))))
diastolicBP = st.sidebar.selectbox("Enter your diastolic blood pressure (mm Hg)", [""] + list(map(str, range(401))))
BMI = st.sidebar.selectbox("Enter your BMI", [""] + list(map(str, range(201))))
heartRate = st.sidebar.selectbox("Enter your heart rate", [""] + list(map(str, range(501))))
glucose = st.sidebar.selectbox("Enter your glucose level (mg/dL)", [""] + list(map(str, range(201))))

# Function to convert formatted number with comma to float
def convert_to_float(value):
    if value:
        value = value.replace(',', '.')
        return float(value)
    return None

dataToPredict = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "cigsPerDay": [cigsPerDay],
    "BPMeds": [BPMeds],
    "prevalentStroke": [prevalentStroke],
    "prevalentHypertension": [prevalentHypertension],
    "diabetes": [diabetes],
    "totalCholesterolLevel": [convert_to_float(totalCholesterolLevel)],
    "systolicBP": [convert_to_float(systolicBP)],
    "diastolicBP": [convert_to_float(diastolicBP)],
    "BMI": [convert_to_float(BMI)],
    "heartRate": [convert_to_float(heartRate)],
    "glucose": [convert_to_float(glucose)]
})

filename = 'random_forest.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

if st.button('PREDICT'):
    if (gender == "") or (age == "") or (cigsPerDay == "") or (BPMeds == "") or (prevalentStroke == "") or \
            (prevalentHypertension == "") or (diabetes == "") or (totalCholesterolLevel == "") or \
            (systolicBP == "") or (diastolicBP == "") or (BMI == "") or (heartRate == "") or (glucose == ""):
        st.write("Please fill in all the information before predicting.")
    else:
        # Mapping the data as explained in the script above
        dataToPredict.replace("Female", 0, inplace=True)
        dataToPredict.replace("Male", 1, inplace=True)

        dataToPredict.replace("Yes", 1, inplace=True)
        dataToPredict.replace("No", 0, inplace=True)

        prediction = loaded_model.predict(dataToPredict)
        probability = loaded_model.predict_proba(dataToPredict)

        risk_percentage = probability[0][1] * 100
        if risk_percentage > 5:
            st.write(f"There is a {risk_percentage:.2f}% risk of Heart Disease.")
        else:
            st.write("No Heart Disease Risk.")
