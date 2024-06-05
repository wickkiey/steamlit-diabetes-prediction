import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the model
with open('random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a function to check if input is numerical
def check_numerical(*args):
    for arg in args:
        try:
            float(arg)
        except ValueError:
            return False
    return True

# Create input fields
st.title('Diabetes Prediction App')
pregnancies = st.text_input('Pregnancies')
glucose = st.text_input('Glucose')
blood_pressure = st.text_input('Blood Pressure')
skin_thickness = st.text_input('Skin Thickness')
insulin = st.text_input('Insulin')
bmi = st.text_input('BMI')
diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function')
age = st.text_input('Age')

# Check if input is numerical and if the Predict button is clicked
if st.button('Predict'):
    if check_numerical(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
        df = pd.DataFrame(data=np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]), 
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        if prediction[0] == 0:
            st.markdown(f"""
                        <div style="background-color: #F5F5F5; padding: 10px; border-radius: 10px;">
                            <h2 style="color: #4F8BF9;">Prediction: No Diabetes</h2>
                            <h2 style="color: #4F8BF9;">Probability: {prediction_proba[0][0]*100:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                        <div style="background-color: #F5F5F5; padding: 10px; border-radius: 10px;">
                            <h2 style="color: #F63366;">Prediction: Diabetes</h2>
                            <h2 style="color: #F63366;">Probability: {prediction_proba[0][1]*100:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.write('Please enter numerical values.')