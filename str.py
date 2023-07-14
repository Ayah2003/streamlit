#!/usr/bin/env python
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
      the app is built based on the "Framingham dataset", using machine learning algorithm called Random Forest, with an accuracy of 88%.  
         """)
st.write("""
         ### To predict your heart disease risk:
         ###### 1- Enter the parameters that best describe you.
         ###### 2- Press the "Predict" button and wait for the result.
         """)


# In[9]:


# Thesidebar function from streamlit is used to create a sidebar for users to input their information.

st.sidebar.title('Please, fill your information to predict your heart condition')

gender=st.sidebar.selectbox("Select your gender", ("Female", "Male" ))
age = st.sidebar.number_input("Select your age", min_value=0, max_value=110, value=24)
cigsPerDay = st.sidebar.selectbox("How many cigarettes do you smoke a day?",  range(0, 100))

BPMeds= st.sidebar.selectbox("Are you in blood pressure medication?", options=("No", "Yes"))

prevalentStroke = st.sidebar.selectbox("Did you have a stroke?", options=("No", "Yes"))

prevalentHypertension=st.sidebar.selectbox("Do you have Hypertension?", options=("No", "Yes"))
diabetes=st.sidebar.selectbox("Do you have diabetes?", options=("No", "Yes"))
totalCholesterolLevel=st.sidebar.number_input("Enter your cholesterol level",  min_value=0, max_value=1000, value=180)
systolicBP =st.sidebar.number_input("Enter your systolic blood pressure (mm Hg)",  min_value=0, max_value=400, value=120)
diastolicBP =st.sidebar.number_input("Enter your diastolic blood pressure (mm Hg)",  min_value=0, max_value=400, value=80)
BMI =st.sidebar.number_input("Select your BMI",  min_value=0, max_value=200, value=22)
heartRate = st.sidebar.number_input("Heart Rate", min_value=0, max_value=500, value=300)
glucose = st.sidebar.number_input("Glucose in mg/dL", min_value=0, max_value=200, value=90)



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


# In[12]:



# Mapping the data as explained in the script above
dataToPredict.replace("Female",0,inplace=True)
dataToPredict.replace("Male",1,inplace=True)


dataToPredict.replace("Yes",1,inplace=True)
dataToPredict.replace("No",0,inplace=True)


# In[14]:


filename = 'random_forest.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict(dataToPredict)
probability = loaded_model.predict_proba(dataToPredict)

if st.button('PREDICT'):
    risk_percentage = probability[0][1] * 100
    if risk_percentage > 5:
        st.write(f"There is a {risk_percentage:.2f}% risk of Heart Disease.")
    else:
        st.write(f"No Heart Disease Risk.")

