import streamlit as st
import joblib
import numpy as np
from PIL import Image # Importing the Image module from PIL


# Load the logo image
logo = Image.open("C:\\Users\\chinedu\\Desktop\\App_Model\\health-care-stroke.jpg")
st.image(logo,  caption="Stroke Prediction App", use_container_width=True)

# Load the trained model
model = joblib.load("stroke_prediction_model.pkl")

st.title("🧠 Stroke Prediction App")
st.write("Enter patient details below to predict stroke risk.")

# Define input fields
gender = st.selectbox("Gender", ["Male", "Female" ])
age = st.number_input("Age", min_value=1, max_value=120)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Encode categorical inputs manually (adjust this to match your model's training encoding)
gender_map = {"Male": 1, "Female": 0}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 1, "Never_worked": 4}
residence_type_map = {"Urban": 1, "Rural": 0}
smoking_map = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}

# Create input vector
input_data = np.array([[
    0,  # ID (not used in prediction)
    gender_map[gender],
    age,
    hypertension_map[hypertension],
    heart_disease_map[heart_disease],
    ever_married_map[ever_married],
    work_type_map[work_type],
    residence_type_map[residence_type],
    avg_glucose_level,
    bmi,
    smoking_map[smoking_status]
]])

# Predict button
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {probability:.2f})")