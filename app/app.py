import json
import os
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HR Analytics | Staff Attrition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. LOAD THE SAVED MODEL AND DEFAULT FEATURE VALUES
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "G09_model.joblib")
DEFAULTS_PATH = os.path.join(PROJECT_DIR, "models", "G09_defaults.json")


@st.cache_resource
def load_model(model_path: str) -> Any:
    """Load the trained model once per Streamlit server session."""
    return joblib.load(model_path)


@st.cache_data
def load_defaults(defaults_path: str) -> dict:
    """Load the baseline values used for model features not shown in the form."""
    with open(defaults_path, "r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        raise ValueError("G09_defaults.json must contain a JSON object.")
    return values


try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    if not os.path.isfile(DEFAULTS_PATH):
        raise FileNotFoundError(f"Defaults file was not found: {DEFAULTS_PATH}")

    model = load_model(MODEL_PATH)
    defaults = load_defaults(DEFAULTS_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support predict_proba().")

    expected_features = list(getattr(model, "feature_names_in_", []))
    if expected_features:
        missing_features = [name for name in expected_features if name not in defaults]
        if missing_features:
            raise ValueError(
                "The defaults file is missing model features: "
                + ", ".join(missing_features)
            )
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.exception(exc)
    st.stop()


# =========================================================
# 3. CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #f1f4f9;
        color: #17283e;
    }
    [data-testid="stHeader"] {
        background: rgba(241, 244, 249, 0.96);
    }
    [data-testid="stSidebar"] {
        background: #132943;
    }
    [data-testid="stSidebar"] * {
        color: #e8eff7;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 10px;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stRadio label p {
        color: #e8eff7 !important;
        font-size: 13px;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 8px 0 22px 0;
        border-bottom: 1px solid rgba(255,255,255,.15);
        margin-bottom: 20px;
    }
    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: #159b9a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
        color: white;
    }
    .brand-name { font-size: 17px; font-weight: 800; line-height: 1.2; }
    .brand-sub { font-size: 10px; color: #b9c8d8 !important; margin-top: 3px; }
    .side-label {
        color: #9db0c4 !important;
        font-size: 10px;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin: 18px 0 8px;
    }
    .side-team {
        color: #dce7f1 !important;
        font-size: 12px;
        line-height: 2;
    }
    .crumb {
        color: #778596;
        font-size: 12px;
        padding: 3px 0 15px;
    }
    .main-title {
        color: #17283e;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -.7px;
        margin: 0;
    }
    .subtitle {
        color: #718092;
        font-size: 12px;
        margin: 3px 0 18px;
    }
    .metric-card {
        background: white;
        border: 1px solid #dce3eb;
        border-radius: 12px;
        padding: 15px 16px 13px;
        min-height: 101px;
    }
    .metric-label {
        color: #687789;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .5px;
        text-transform: uppercase;
    }
    .metric-value {
        color: #17283e;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 3px;
    }
    .metric-note { color: #7b8795; font-size: 11px; margin-top: 3px; }
    .section-title {
        color: #1b2c41;
        font-size: 18px;
        font-weight: 800;
        margin: 0 0 3px;
    }
    .section-subtitle {
        color: #7b8795;
        font-size: 11px;
        margin-bottom: 13px;
    }

    /* Explicitly use a light control surface and dark readable text. */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stSelectbox label,
    .stNumberInput label {
        color: #26364a !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        border-radius: 8px !important;
        background: #f8fafc !important;
        border-color: #d5dee8 !important;
        color: #26364a !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-baseweb="base-input"] *,
    input {
        color: #26364a !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #526579 !important;
    }
    [data-testid="stNumberInput"] button {
        background: #edf2f7 !important;
        color: #26364a !important;
        border-color: #d5dee8 !important;
    }
    [data-testid="stNumberInput"] button svg {
        fill: #26364a !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: #147b89;
        color: white !important;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
        min-height: 42px;
        width: 100%;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #106977;
        color: white !important;
        border: 0;
    }
    .result-box {
        background: #eef7f9;
        border-radius: 9px;
        padding: 12px;
        color: #385465;
        font-size: 12px;
        line-height: 1.7;
    }
    .small-note { color: #7b8795; font-size: 11px; line-height: 1.6; }
    .bottom-panel {
        background: white;
        border: 1px solid #dce3eb;
        border-radius: 12px;
        padding: 15px 18px;
    }
    .bar-track {
        width: 100%;
        height: 13px;
        background: #e9edf2;
        border-radius: 20px;
        overflow: hidden;
        display: flex;
        margin: 8px 0 9px;
    }
    .bar-retained { background: #159b8d; height: 100%; }
    .bar-left { background: #e9a33d; height: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. SIDEBAR AND NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">↗</div>
            <div>
                <div class="brand-name">HR Analytics</div>
                <div class="brand-sub">GROUP G09</div>
            </div>
        </div>
        <div class="side-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Attrition prediction",
            "Salary analysis",
            "Model performance",
            "Fairness analysis",
            "About the project",
        ],
        index=1,
        label_visibility="collapsed",
        key="page_navigation",
    )

    st.markdown(
        """
        <div class="side-label" style="margin-top:28px;">PROJECT TEAM</div>
        <div class="side-team">
            <span style="color:#31c9bd">●</span> Clare Nabwile<br>
            <span style="color:#31c9bd">●</span> Abel Morara<br>
            <span style="color:#31c9bd">●</span> Owiye Sylvia<br>
            <span style="color:#31c9bd">●</span> Jack Otuko<br>
            <span style="color:#31c9bd">●</span> Obed Kiprono
        </div>
        <div style="height:22px"></div>
        <div style="border-top:1px solid rgba(255,255,255,.15); padding-top:15px;
                    color:#aebed0; font-size:10px; line-height:1.8;">
            Kibabii University<br>
            CSC 4110 • Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. COMMON DASHBOARD HEADER AND DATASET SUMMARY
# =========================================================
st.markdown(
    '<div class="crumb">HR Management / Attrition Prediction</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-title">Staff Attrition Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">IBM HR Analytics Employee Attrition • '
    'Early-warning and staff support system</div>',
    unsafe_allow_html=True,
)

# These are fixed descriptive totals for the IBM HR attrition dataset,
# not live counts of employees entered into the prediction form.
summary_metrics = [
    ("TOTAL EMPLOYEES", "1,470", "Dataset records", "#3478e5"),
    ("STAFF RETAINED", "1,233", "83.9% of records", "#159b8d"),
    ("STAFF WHO LEFT", "237", "Historical attrition", "#e47726"),
    ("ATTRITION RATE", "16.1%", "Overall dataset", "#8a4de0"),
]

metric_columns = st.columns(4)
for column, (label, value, note, accent) in zip(metric_columns, summary_metrics):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    <span style="color:{accent};font-size:16px;">▍</span> {label}
                </div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# =========================================================
# 6. ATTRITION PREDICTION PAGE
# =========================================================
def make_prediction(
    overtime: str,
    job_satisfaction: int,
    monthly_income: int,
    years_at_company: int,
    distance_from_home: int,
    work_life_balance: int,
    business_travel: str,
) -> float:
    """
    Combine the seven form values with the other saved defaults,
    then return the model probability for the attrition class.
    """
    input_data = pd.DataFrame([defaults])

    input_data.loc[0, "OverTime"] = overtime
    input_data.loc[0, "JobSatisfaction"] = job_satisfaction
    input_data.loc[0, "MonthlyIncome"] = monthly_income
    input_data.loc[0, "YearsAtCompany"] = years_at_company
    input_data.loc[0, "DistanceFromHome"] = distance_from_home
    input_data.loc[0, "WorkLifeBalance"] = work_life_balance
    input_data.loc[0, "BusinessTravel"] = business_travel

    # Keep the same feature order used when the model was trained.
    if expected_features:
        input_data = input_data.reindex(columns=expected_features)

    probabilities = model.predict_proba(input_data)[0]
    classes = list(getattr(model, "classes_", []))

    # The training target is expected to encode Attrition as 1 (Yes).
    if 1 not in classes:
        raise ValueError(
            f"Could not find the positive attrition class (1). Model classes: {classes}"
        )

    positive_index = classes.index(1)
    return float(probabilities[positive_index])


if page == "Attrition prediction":
    left_column, right_column = st.columns([1.55, 1], gap="medium")

    with left_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">Employee risk assessment</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Enter anonymous staff information.</div>',
                unsafe_allow_html=True,
            )

            with st.form("attrition_assessment", clear_on_submit=False):
                input_left, input_right = st.columns(2)

                with input_left:
                    overtime = st.selectbox("Works overtime", ["No", "Yes"])
                    monthly_income = st.number_input(
                        "Monthly income ($)",
                        min_value=1000,
                        max_value=20000,
                        value=5000,
                        step=100,
                    )
                    distance_from_home = st.number_input(
                        "Distance from home (dataset units)",
                        min_value=1,
                        max_value=29,
                        value=5,
                    )
                    business_travel = st.selectbox(
                        "Business travel",
                        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                        format_func=lambda value: {
                            "Non-Travel": "Non-travel",
                            "Travel_Rarely": "Travel rarely",
                            "Travel_Frequently": "Travel frequently",
                        }[value],
                    )

                with input_right:
                    job_satisfaction = st.selectbox(
                        "Job satisfaction",
                        [1, 2, 3, 4],
                        format_func=lambda value: {
                            1: "1 – Low",
                            2: "2 – Medium",
                            3: "3 – High",
                            4: "4 – Very high",
                        }[value],
                    )
                    years_at_company = st.number_input(
                        "Years at company",
                        min_value=0,
                        max_value=40,
                        value=5,
                    )
                    work_life_balance = st.selectbox(
                        "Work-life balance",
                        [1, 2, 3, 4],
                        format_func=lambda value: {
                            1: "1 – Low",
                            2: "2 – Good",
                            3: "3 – High",
                            4: "4 – Very high",
                        }[value],
                    )

                submitted = st.form_submit_button("Predict attrition risk")

    with right_column:
        with st.container(border=True):
            st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Illustrative model estimate only.</div>',
                unsafe_allow_html=True,
            )

            if submitted:
                try:
                    result = make_prediction(
                        overtime=overtime,
                        job_satisfaction=job_satisfaction,
                        monthly_income=monthly_income,
                        years_at_company=years_at_company,
                        distance_from_home=distance_from_home,
                        work_life_balance=work_life_balance,
                        business_travel=business_travel,
                    )
                    st.session_state["attrition_probability"] = result
                    st.session_state.pop("attrition_prediction_error", None)
                except Exception as exc:
                    st.session_state["attrition_prediction_error"] = str(exc)
                    st.session_state.pop("attrition_probability", None)

            prediction_error = st.session_state.get("attrition_prediction_error")
            probability = st.session_state.get("attrition_probability")

            if prediction_error:
                st.error(f"Prediction failed: {prediction_error}")
            elif probability is None:
                st.markdown(
                    """
                    <div style="text-align:center;padding:34px 8px 30px;">
                        <div style="font-size:17px;font-weight:700;color:#718092;">
                            No assessment yet
                        </div>
                        <div class="small-note">
                            Complete the form and select Predict attrition risk.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                percent = int(round(probability * 100))
                # Half-circle SVG gauge; dash length follows the model probability.
                arc_length = 282.74
                dash_length = arc_length * max(0.0, min(1.0, probability))
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:7px 0 0;">
                        <svg viewBox="0 0 220 128" width="100%" style="max-width:260px;">
                            <path d="M 20 110 A 90 90 0 0 1 200 110"
                                  fill="none" stroke="#e3e9f0" stroke-width="15"
                                  stroke-linecap="round"/>
                            <path d="M 20 110 A 90 90 0 0 1 200 110"
                                  fill="none" stroke="#e9a33d" stroke-width="15"
                                  stroke-linecap="round"
                                  stroke-dasharray="{dash_length:.2f} {arc_length:.2f}"/>
                            <text x="110" y="88" text-anchor="middle"
                                  font-size="34" font-weight="800" fill="#17283e">
                                {percent}%
                            </text>
                            <text x="110" y="109" text-anchor="middle"
                                  font-size="10" fill="#718092">
                                Estimated attrition probability
                            </text>
                        </svg>
                        <div style="display:flex;justify-content:space-between;
                                    color:#7b8795;font-size:10px;margin:0 18px 8px;">
                            <span>0%</span><span>100%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <hr style="border:0;border-top:1px solid #e5eaf0;margin:5px 0 12px;">
                <div style="font-size:13px;font-weight:800;color:#1b2c41;margin-bottom:8px;">
                    Suggested support conversation
                </div>
                <div class="result-box">
                    Discuss the employee's:<br>
                    • Workload and overtime<br>
                    • Job satisfaction<br>
                    • Work-life balance<br>
                    • Travel demands
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
        <div class="bottom-panel">
            <div style="display:flex;align-items:center;gap:24px;">
                <div style="flex:1;">
                    <div style="font-size:13px;font-weight:800;color:#1b2c41;">
                        Historical staff distribution
                    </div>
                    <div class="bar-track">
                        <div class="bar-retained" style="width:83.9%"></div>
                        <div class="bar-left" style="width:16.1%"></div>
                    </div>
                    <div style="display:flex;gap:28px;font-size:11px;color:#718092;">
                        <span><span style="color:#159b8d">●</span> Retained: 1,233</span>
                        <span><span style="color:#e9a33d">●</span> Left: 237</span>
                    </div>
                </div>
                <div style="width:220px;border-left:1px solid #e1e7ee;padding-left:18px;">
                    <div style="font-size:12px;font-weight:800;color:#1b2c41;">
                        Responsible use
                    </div>
                    <div class="small-note" style="margin-top:4px;">
                        Use only for learning and staff-support discussions.
                        Never use this estimate for dismissal, promotion, pay,
                        or other employment decisions.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. OTHER NAVIGATION PAGES
# =========================================================
elif page == "Overview":
    st.subheader("Project overview")
    st.write(
        "This project demonstrates how a machine-learning model can estimate "
        "employee attrition probability using selected HR-related information."
    )
    overview_left, overview_right = st.columns(2)
    with overview_left:
        with st.container(border=True):
            st.markdown("**Dataset summary**")
            st.write("1,470 employee records")
            st.write("1,233 retained employees")
            st.write("237 employees recorded as having left")
    with overview_right:
        with st.container(border=True):
            st.markdown("**What this app does**")
            st.write("Collects selected, anonymous input values.")
            st.write("Uses the saved G09 model to calculate a probability.")
            st.write("Displays a support-oriented interpretation.")

elif page == "Salary analysis":
    st.subheader("Salary analysis")
    st.info(
        "This section is a navigation placeholder. A salary analysis chart "
        "cannot be produced from the model and defaults files alone. "
        "The approved dataset or prepared salary-analysis outputs must be "
        "added before displaying real salary statistics."
    )

elif page == "Model performance":
    st.subheader("Model performance")
    st.info(
        "This section is a navigation placeholder. Add the verified evaluation "
        "metrics from the group's model-evaluation report before presenting "
        "accuracy, precision, recall, F1-score, or a confusion matrix here."
    )

elif page == "Fairness analysis":
    st.subheader("Fairness analysis")
    st.warning(
        "A prediction model can perform differently across employee groups. "
        "Group-level fairness metrics must be calculated and reviewed before "
        "drawing conclusions about model fairness."
    )
    st.write(
        "Do not use individual predictions or sensitive attributes to make "
        "dismissal, promotion, pay, or other employment decisions."
    )

elif page == "About the project":
    st.subheader("About this project")
    st.write("**Project:** G09 Staff Attrition Prediction")
    st.write("**Institution:** Kibabii University")
    st.write("**Course:** CSC 4110 – Machine Learning")
    st.write("**Dataset:** IBM HR Analytics Employee Attrition Dataset")
    st.markdown("**Project team**")
    st.write(
        "Clare Nabwile · Abel Morara · Owiye Sylvia · Jack Otuko · Obed Kiprono"
    )
    st.warning(
        "This is an educational demonstration. Predictions are estimates, "
        "not certainties, and must not be used as the basis for employment decisions."
    )
