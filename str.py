#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries

import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier



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



# Thesidebar function from streamlit is used to create a sidebar for users to input their information.

st.sidebar.title('Please, fill your information to predict your heart condition')

gender = st.sidebar.selectbox("Select your gender", (None, "Female", "Male"))
age = st.sidebar.number_input("Select your age", min_value=0, max_value=110, value=None)
cigsPerDay = st.sidebar.number_input("How many cigarettes do you smoke a day?", min_value=0, max_value=100, value=None)

BPMeds = st.sidebar.selectbox("Are you on blood pressure medication?", (None, "No", "Yes"))

prevalentStroke = st.sidebar.selectbox("Did you have a stroke?", (None, "No", "Yes"))

prevalentHypertension = st.sidebar.selectbox("Do you have Hypertension?", (None, "No", "Yes"))
diabetes = st.sidebar.selectbox("Do you have diabetes?", (None, "No", "Yes"))
totalCholesterolLevel = st.sidebar.number_input("Enter your cholesterol level", min_value=0, max_value=1000, value=None)
systolicBP = st.sidebar.number_input("Enter your systolic blood pressure (mm Hg)", min_value=0, max_value=400, value=None)
diastolicBP = st.sidebar.number_input("Enter your diastolic blood pressure (mm Hg)", min_value=0, max_value=400, value=None)
BMI = st.sidebar.number_input("Enter your BMI", min_value=0, max_value=200, value=None)
heartRate = st.sidebar.number_input("Enter your heart rate", min_value=0, max_value=500, value=None)
glucose = st.sidebar.number_input("Enter your glucose level (mg/dL)", min_value=0, max_value=200, value=None)

dataToPredict = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "cigsPerDay": [cigsPerDay],
    "BPMeds": [BPMeds],
    "prevalentStroke": [prevalentStroke],
    "prevalentHypertension": [prevalentHypertension],
    "diabetes": [diabetes],
    "totalCholesterolLevel": [totalCholesterolLevel],
    "systolicBP": [systolicBP],
    "diastolicBP": [diastolicBP],
    "BMI": [BMI],
    "heartRate": [heartRate],
    "glucose": [glucose]
})

filename = 'random_forest.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

if st.button('PREDICT'):
    if (gender is None) or (age is None) or (BPMeds is None) or (prevalentStroke is None) or \
            (prevalentHypertension is None) or (diabetes is None) or (totalCholesterolLevel is None) or \
            (systolicBP is None) or (diastolicBP is None) or (BMI is None) or (heartRate is None) or (glucose is None):
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
