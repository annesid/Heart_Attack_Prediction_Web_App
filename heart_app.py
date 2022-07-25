#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:52:01 2022

@author: anne
"""
import pickle
import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


#%% Machine Learning
MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')
with open(MODEL_PATH,'rb') as file:
    model=pickle.load(file)

#%% Test Deployment 

patience_info = pd.DataFrame({"age": [65,61,45,40,48,41,36,45,57,69],
                 "sex": [1,1,0,0,1,1,0,1,1,1],
                 "cp": [3,0,1,1,2,0,2,0,0,2],
                 "trtbps": [142,140,128,125,132,108,121,111,155,179],
                 "chol": [220,207,204,307,254,165,214,198,271,273],
                 "fbs": [1,0,0,0,0,0,0,0,0,1],
                  "restecg": [0,0,0,1,1,0,1,0,0,0],
                 "thalachh": [158,138,172,162,180,115,168,176,112,151],
                 "exng": [0,1,0,0,0,1,0,0,1,1],
                 "oldpeak": [2.3,1.9,1.4,0,0,2,0,0,0.8,1.6],
                 "slp": [1,2,2,2,2,1,2,2,2,1],
                 "caa": [0,1,0,0,0,0,0,1,0,0],
                 "thall": [1,3,2,2,2,3,2,2,3,3],
                 "True output": [1,0,1,1,1,0,1,0,0,0]
                 })

heart_attack = {0: "less chance of heart attack",
                 1: "more chance of heart attack"}

# split into X_true and y_true
X_true = patience_info.drop(labels=["sex","trtbps","chol","fbs","restecg","slp","True output"], axis=1)
y_true = np.expand_dims(patience_info["True output"],-1)

#Predict the data one by one
new_pred = model.predict(X_true)
if np.argmax(new_pred) == 0:
    new_pred = [0,1]
    print(heart_attack[np.argmax(new_pred)])
else:
    new_pred = [1,0]
    print(heart_attack[np.argmax(new_pred)])


#%% Streamlit

st.set_page_config( page_title="Heart Attack Prediction App", page_icon=":muscle:", layout="wide")


col1, col2 = st.columns(2)

with col1:
    st.markdown("![Alt Text](https://media.giphy.com/media/j13ierzOVDYTwTgMNj/giphy.gif)")
with col2:
    st.title('Heart Attack Prediction App')
    st.write('This simple application may help you to detect if you are at risk of having heart attack, however this cannot be used as a substitute for real medical advice.')
    my_expander = st.expander(label='To book appointment with nearest doctor')
    with my_expander:'Contact +6 03 123456789 for full upgrade on Health App for only $0.99'
    my_expander = st.expander(label='The Data')
    with my_expander:'The data for the following example is originally from Rashik Rahman [link](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)'
    my_expander = st.expander(label='The Awesome Team Behind This App')
    with my_expander:'N & Co.'

col1, col2 = st.columns(2)

with st.form('Heart Disease Prediction Form'):
    with col1:
        st.subheader("Patient's Info")
        age     = st.slider("Age in Years", 1, 110, 25, 1)
        # sex     = st.radio("Gender",options=("Female","Male"))
        cp      = st.selectbox("Chest Pain Type",options=("Typical Angina","Atypical Angina","Non-Angina Pain","Asymptomatic"))
        # trtbps  = st.slider("Resting blood pressure (in mm Hg)",90, 200, 120, 1)
        # chol    = st.slider("Cholestoral in mg/dl",100, 570, 130, 1)
        # fbs     = st.radio("(Fasting blood sugar > 120 mg/dl)",options= ("False","True"))
        thalachh = st.slider("Max heart rate", 60, 300, 130, 1)
        exng    = st.radio("Exercise induced angina",options = ("No","Yes"))
    with col2:
        st.subheader("Patient's History")
        # restecg = st.selectbox("Resting ECG",options=("Normal","Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"))

        oldpeak = st.number_input("Previous peak")
        # slp     = st.radio("Exercise ST",options = ('0','1','2'))
        caa     = st.radio("Number of major vessels",options = ('0','1','2','3'))
        thall   = st.radio("Thallium Stress Test",options = ('0','1','2','3'))
        
    
    submitted = st.form_submit_button('Predict')
        
# cp options
    if cp == "Typical Angina":
        cp = 0
    elif cp == "ATypical Angina":
    	cp = 1
    elif cp == "Non-Angina Pain":
    	cp = 2
    else:
    	cp = 3

# exng options
    if exng == "Yes":
        exng = 1
    else:
        exng = 0

    row = [age,thalachh,oldpeak,cp,exng,caa,thall]
    
    if submitted:
        patience_columns=np.expand_dims(row,axis=0)
        output=model.predict(patience_columns)[0]

        if output==0:
            st.subheader("Wow! Congratulations! You are not at risk of heart attack. Barney is happy for you!")
            st.markdown("![Alt Text](https://media.giphy.com/media/f7GXfUvpLXNTyDnMS2/giphy.gif)")

            
        else:
            st.subheader('You have a **risk of heart attack.** Kindly consult a doctor immediately!')
            st.markdown("![Alt Text](https://media.giphy.com/media/5UfsOgBaXNQgE/giphy.gif)")



                