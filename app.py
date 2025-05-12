#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 04:05:01 2025

@author: merajalam
"""
# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
try:
    with open('/Users/merajalam/Desktop/churn_prediction_model/trained_model.sav', 'rb') as f:
        model = pickle.load(f)
    with open('/Users/merajalam/Desktop/churn_prediction_model/scaler.pkl', 'rb') as s:
        scaler = pickle.load(s)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.set_page_config(page_title="Churn Predictor", page_icon=":bar_chart:")
st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn or not based on input features.")

# Input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)
    submit = st.form_submit_button("Predict")

# Encoding and prediction
if submit:
    data = pd.DataFrame([{
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': SeniorCitizen,
        'Partner': 1 if Partner == 'Yes' else 0,
        'Dependents': 1 if Dependents == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])
    scaled = scaler.transform(data)
    result = model.predict(scaled)[0]
    st.success(f"Prediction: {'Churn' if result == 1 else 'No Churn'}")