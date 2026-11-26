
import streamlit as st
import pandas as pd
import joblib
import json
import os

# Locate saved model and default values
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "G09_model.joblib")
DEFAULTS_PATH = os.path.join(PROJECT_DIR, "models", "G09_defaults.json")

# Load model and default values
model = joblib.load(MODEL_PATH)

with open(DEFAULTS_PATH, "r") as file:
    defaults = json.load(file)

# Configure application
st.set_page_config(page_title="G9 Staff Attrition Predictor")

# Project information
st.title("G9 Staff Attrition Prediction")
st.subheader("CSC 4110 - Machine Learning")

st.write("**Group:** G9")

st.write("**Group Members:**")
st.write("""
- Clare Nabwile
- Abel Morara
- Owiye Sylvia
- Jack Otuko
- Obed Kiprono
""")

st.write("**Dataset:** IBM HR Analytics Employee Attrition Dataset")

st.divider()

# Employee input section
st.header("Employee Information")

overtime = st.selectbox("OverTime", ["No", "Yes"])

job_satisfaction = st.selectbox(
    "Job Satisfaction", [1, 2, 3, 4]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=20000,
    value=5000
)

years_at_company = st.number_input(
    "Years at Company",
    min_value=0,
    max_value=40,
    value=5
)

distance_from_home = st.number_input(
    "Distance From Home",
    min_value=1,
    max_value=29,
    value=5
)

work_life_balance = st.selectbox(
    "Work-Life Balance", [1, 2, 3, 4]
)

business_travel = st.selectbox(
    "Business Travel",
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
)

# Generate prediction
if st.button("Predict Attrition Risk"):

    # Start with all 30 model features
    input_data = pd.DataFrame([defaults])

    # Replace seven default values with user inputs
    input_data["OverTime"] = overtime
    input_data["JobSatisfaction"] = job_satisfaction
    input_data["MonthlyIncome"] = monthly_income
    input_data["YearsAtCompany"] = years_at_company
    input_data["DistanceFromHome"] = distance_from_home
    input_data["WorkLifeBalance"] = work_life_balance
    input_data["BusinessTravel"] = business_travel

    # Calculate attrition probability
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.metric(
        "Estimated Attrition Probability",
        f"{probability:.2%}"
    )

    st.info(
        "This is a model estimate, not a certainty or a basis "
        "for employment decisions."
    )

    st.subheader("Suggested Support Conversation")

    st.write(
        "Would you like to discuss your workload, work-life balance, "
        "travel demands, or any support that could improve your "
        "experience at work?"
    )
