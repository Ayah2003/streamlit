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
age = st.sidebar.number_input("Select your age", min_value=0, max_value=110, value="")
cigsPerDay = st.sidebar.selectbox("How many cigarettes do you smoke a day?", ["", "0"] + list(range(1, 100)))

BPMeds = st.sidebar.selectbox("Are you on blood pressure medication?", ["", "No", "Yes"])

prevalentStroke = st.sidebar.selectbox("Did you have a stroke?", ["", "No", "Yes"])

prevalentHypertension = st.sidebar.selectbox("Do you have hypertension?", ["", "No", "Yes"])
diabetes = st.sidebar.selectbox("Do you have diabetes?", ["", "No", "Yes"])
totalCholesterolLevel = st.sidebar.number_input("Enter your cholesterol level", min_value=0, max_value=1000, value="")
systolicBP = st.sidebar.number_input("Enter your systolic blood pressure (mm Hg)", min_value=0, max_value=400, value="")
diastolicBP = st.sidebar.number_input("Enter your diastolic blood pressure (mm Hg)", min_value=0, max_value=400, value="")
BMI = st.sidebar.number_input("Enter your BMI", min_value=0, max_value=200, value="")
heartRate = st.sidebar.number_input("Enter your heart rate", min_value=0, max_value=500, value="")
glucose = st.sidebar.number_input("Enter your glucose level (mg/dL)", min_value=0, max_value=200, value="")

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
