import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('insurance_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Insurance Cost Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", options=['male', 'female'])
smoker = st.selectbox("Smoker", options=['yes', 'no'])
region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'])

input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == 'male' else 0],
    'smoker_yes': [1 if smoker == 'yes' else 0],
    'region_northwest': [1 if region == 'northwest' else 0],
    'region_southeast': [1 if region == 'southeast' else 0],
    'region_southwest': [1 if region == 'southwest' else 0]
})

if st.button("Predict"):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f"Predicted Insurance Cost: ${prediction[0][0]:.2f}")