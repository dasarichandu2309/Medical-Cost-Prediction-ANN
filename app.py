import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MedCost AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_ann():

    model = load_model("medical_cost_ann.keras")

    preprocessor = joblib.load("preprocessor.pkl")

    feature_names = joblib.load("feature_names.pkl")

    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_ann()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""

<style>

html,
body,
[data-testid="stAppViewContainer"]{

background:linear-gradient(
135deg,
#0F172A,
#172554,
#0F172A
);

background-size:400% 400%;

animation:gradient 15s ease infinite;

}

@keyframes gradient{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}

/* Hide Streamlit Footer */

footer{

visibility:hidden;

}

header{

visibility:hidden;

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#08131F;

border-right:1px solid rgba(255,255,255,.08);

}

/* Main Title */

.main-title{

font-size:46px;

font-weight:800;

color:white;

text-align:center;

margin-top:10px;

margin-bottom:0px;

}

/* Subtitle */

.subtitle{

font-size:18px;

color:#CBD5E1;

text-align:center;

margin-bottom:30px;

}

/* Glass Card */

.glass{

background:rgba(255,255,255,.10);

backdrop-filter:blur(18px);

padding:24px;

border-radius:20px;

border:1px solid rgba(255,255,255,.15);

box-shadow:0 8px 30px rgba(0,0,0,.35);

margin-bottom:20px;

}

/* Prediction Card */

.metric{

background:linear-gradient(
135deg,
rgba(14,165,233,.25),
rgba(30,41,59,.65)
);

padding:30px;

border-radius:18px;

border:1px solid rgba(255,255,255,.12);

text-align:center;

color:white;

box-shadow:0px 8px 24px rgba(0,0,0,.30);

}

/* Prediction Number */

.pred{

font-size:60px;

font-weight:900;

color:#22C55E;

margin-top:10px;

}

/* Button */

.stButton>button{

width:100%;

height:55px;

font-size:20px;

font-weight:bold;

background:#0EA5E9;

color:white;

border:none;

border-radius:12px;

transition:.25s;

}

.stButton>button:hover{

background:#0284C7;

transform:scale(1.02);

}

/* Metrics */

[data-testid="metric-container"]{

background:rgba(255,255,255,.08);

border-radius:15px;

padding:18px;

border:1px solid rgba(255,255,255,.08);

}

/* Section Heading */

.section{

font-size:24px;

font-weight:700;

color:white;

margin-bottom:15px;

}

</style>

""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div class="main-title">

🏥 MedCost AI

</div>

<div class="subtitle">

AI-Powered Medical Insurance Cost Prediction

<br>

Artificial Neural Network • TensorFlow • Streamlit • Optuna

</div>

""",
unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
"https://img.icons8.com/color/512/hospital-3.png",
width=110
)

st.sidebar.title("🏥 MedCost AI")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model")

st.sidebar.success("Artificial Neural Network")

st.sidebar.subheader("⚙️ Framework")

st.sidebar.info("TensorFlow + Keras")

st.sidebar.subheader("🎯 Hyperparameter Tuning")

st.sidebar.success("Optuna")

st.sidebar.subheader("📊 Dataset")

st.sidebar.info("100,000+ Medical Records")

st.sidebar.subheader("📈 Performance")

st.sidebar.success("R² Score : 0.675")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Dasari Chandu")

# =====================================================
# DEFAULT VALUES
# =====================================================

defaults = {

    "urban_rural":"Suburban",

    "education":"Graduate",

    "marital_status":"Married",

    "employment_status":"Employed",

    "household_size":3,

    "dependents":1,

    "alcohol_freq":"Occasional",

    "network_tier":"Bronze",

    "policy_term_years":4,

    "policy_changes_last_2yrs":0,

    "provider_quality":3.8,

    "risk_score":0.55,

    "claims_count":1,

    "avg_claim_amount":4500,

    "total_claims_paid":4500,

    "chronic_count":0,

    "asthma":0,

    "copd":0,

    "cardiovascular_disease":0,

    "cancer_history":0,

    "kidney_disease":0,

    "liver_disease":0,

    "arthritis":0,

    "mental_health":0,

    "proc_imaging_count":1,

    "proc_surgery_count":0,

    "proc_physio_count":0,

    "proc_consult_count":1,

    "proc_lab_count":1,

    "is_high_risk":0,

    "had_major_procedure":0

}

# =====================================================
# RISK FUNCTION
# =====================================================

def risk_level(cost):

    if cost < 5000:

        return "🟢 LOW"

    elif cost < 15000:

        return "🟡 MODERATE"

    else:

        return "🔴 HIGH"


def recommendation(cost):

    if cost < 5000:

        return (
            "Excellent health profile.\n\n"
            "✔ Maintain healthy BMI\n"
            "✔ Continue regular exercise\n"
            "✔ Annual health check-up recommended"
        )

    elif cost < 15000:

        return (
            "Moderate healthcare expenditure expected.\n\n"
            "✔ Improve diet quality\n"
            "✔ Exercise regularly\n"
            "✔ Schedule preventive health screenings"
        )

    else:

        return (
            "Higher healthcare expenditure expected.\n\n"
            "✔ Consider comprehensive insurance\n"
            "✔ Consult your physician regularly\n"
            "✔ Monitor chronic health conditions closely"
        )
# =====================================================
# PATIENT INFORMATION
# =====================================================

st.markdown(
"""
<div class='section'>

📝 Patient Information

</div>
""",
unsafe_allow_html=True
)

# =====================================================
# ROW 1
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# PERSONAL DETAILS
# =====================================================

with col1:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("👤 Personal Details")

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    sex = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West"
        ]
    )

    income = st.number_input(
        "Annual Income (₹)",
        min_value=1000,
        max_value=500000,
        value=50000,
        step=1000
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HEALTH DETAILS
# =====================================================

with col2:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("❤️ Health Details")

    bmi = st.slider(
        "BMI",
        15.0,
        45.0,
        24.5
    )

    smoker = st.selectbox(
        "Smoking Status",
        [
            "Never",
            "Former",
            "Current"
        ]
    )

    alcohol_freq = st.selectbox(
        "Alcohol Frequency",
        [
            "Never",
            "Rare",
            "Occasional",
            "Frequent"
        ]
    )

    diabetes = st.selectbox(
        "Diabetes",
        [
            "No",
            "Yes"
        ]
    )

    hypertension = st.selectbox(
        "Hypertension",
        [
            "No",
            "Yes"
        ]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ROW 2
# =====================================================

col3, col4 = st.columns(2)

# =====================================================
# MEDICAL HISTORY
# =====================================================

with col3:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("🏥 Medical History")

    visits = st.slider(
        "Doctor Visits (Last Year)",
        0,
        25,
        2
    )

    hospitalizations = st.slider(
        "Hospitalizations (Last 3 Years)",
        0,
        10,
        0
    )

    medication = st.slider(
        "Medication Count",
        0,
        20,
        2
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# INSURANCE
# =====================================================

with col4:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("💳 Insurance Details")

    plan_type = st.selectbox(
        "Insurance Plan",
        [
            "HMO",
            "PPO",
            "EPO"
        ]
    )

    deductible = st.selectbox(
        "Deductible",
        [
            500,
            1000,
            1500,
            2000,
            3000
        ]
    )

    st.info(
        "Remaining insurance features are automatically generated by the AI model."
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button(
    "🚀 Predict Annual Medical Cost"
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    # ---------------------------------------------
    # Create Default Dictionary
    # ---------------------------------------------

    data = defaults.copy()

    # Numeric Defaults

    data["days_hospitalized_last_3yrs"] = 0
    data["systolic_bp"] = 120
    data["diastolic_bp"] = 80
    data["ldl"] = 110
    data["hba1c"] = 5.4
    data["copay"] = 20

    # ---------------------------------------------
    # User Inputs
    # ---------------------------------------------

    data["age"] = age
    data["sex"] = sex
    data["region"] = region
    data["income"] = income
    data["bmi"] = bmi

    data["smoker"] = smoker
    data["alcohol_freq"] = alcohol_freq

    data["plan_type"] = plan_type
    data["deductible"] = deductible

    data["visits_last_year"] = visits
    data["hospitalizations_last_3yrs"] = hospitalizations
    data["medication_count"] = medication

    data["diabetes"] = 1 if diabetes == "Yes" else 0
    data["hypertension"] = 1 if hypertension == "Yes" else 0

    # ---------------------------------------------
    # DataFrame
    # ---------------------------------------------

    input_df = pd.DataFrame([data])

    input_df = input_df[feature_names]

    # ---------------------------------------------
    # Transform
    # ---------------------------------------------

    input_processed = preprocessor.transform(input_df)

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = float(
        model.predict(
            input_processed,
            verbose=0
        )[0][0]
    )

    prediction = round(prediction,2)

    # ---------------------------------------------
    # Dashboard
    # ---------------------------------------------

    st.markdown("---")

    st.markdown("## 📊 AI Prediction Dashboard")

    c1,c2,c3 = st.columns(3)

    # ---------------------------------------------
    # Cost Card
    # ---------------------------------------------

    with c1:

        st.metric(

            label="💰 Annual Medical Cost",

            value=f"₹ {prediction:,.2f}"

        )

    # ---------------------------------------------
    # Risk
    # ---------------------------------------------

    risk = risk_level(prediction)

    with c2:

        st.metric(

            label="❤️ Health Risk",

            value=risk

        )

    # ---------------------------------------------
    # BMI
    # ---------------------------------------------

    if bmi < 18.5:

        bmi_status="Underweight"

    elif bmi <25:

        bmi_status="Healthy"

    elif bmi <30:

        bmi_status="Overweight"

    else:

        bmi_status="Obese"

    with c3:

        st.metric(

            label="⚖ BMI Status",

            value=bmi_status

        )

    st.write("")

    # ---------------------------------------------
    # Health Summary
    # ---------------------------------------------

    st.markdown("### 📋 Health Summary")

    h1,h2,h3,h4 = st.columns(4)

    h1.metric("Smoking",smoker)

    h2.metric(
        "Diabetes",
        diabetes
    )

    h3.metric(
        "Hypertension",
        hypertension
    )

    h4.metric(
        "Doctor Visits",
        visits
    )

    st.write("")

    # ---------------------------------------------
    # Recommendation
    # ---------------------------------------------

    st.markdown("### 🤖 AI Recommendation")

    if prediction < 5000:

        st.success(recommendation(prediction))

    elif prediction <15000:

        st.warning(recommendation(prediction))

    else:

        st.error(recommendation(prediction))