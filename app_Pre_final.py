import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import altair as alt

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Classification & Support System",
    page_icon="🧠",
    layout="wide",
)

# ------------------ STYLES ------------------
st.markdown("""
<style>
body { background-color:#0E1117; color:#FAFAFA; }
h1,h2,h3,h4,h5 { color:#E0E0E0; }
.stButton>button { background:#2E7D32; color:white; font-weight:600; border-radius:10px; }
.stButton>button:hover { background:#43A047; color:white; }
.question-box { padding:10px; border-radius:5px; background:#1A1D22; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MODELS SAFELY
# -------------------------------------------------------------
MODEL_FILES = {
    "Anxiety": "best_model_Anxiety_Label_Logistic_Regression.joblib",
    "Stress": "best_model_Stress_Label_Logistic_Regression.joblib",
    "Depression": "best_model_Depression_Label_CatBoost.joblib",
}

ENCODER_FILES = {
    "Anxiety": "final_anxiety_encoder.joblib",
    "Stress": "final_stress_encoder.joblib",
    "Depression": "final_depression_encoder.joblib",
}


def load_models():
    models = {}
    for key, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            st.error(f"Model missing: {path}")
            models[key] = None
    return models


def load_encoders():
    encoders = {}
    for key, path in ENCODER_FILES.items():
        if os.path.exists(path):
            encoders[key] = joblib.load(path)
        else:
            encoders[key] = None
    return encoders


# -------------------------------------------------------------
# ALIGN INPUT FEATURES
# -------------------------------------------------------------
def align_features(df, model):
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return df

    # Add missing features
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    return df[expected]


# -------------------------------------------------------------
# QUESTION SETS (Original from RAW DATASET)
# -------------------------------------------------------------

GAD7 = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

PHQ9 = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or failure",
    "Trouble concentrating",
    "Moving/speaking slowly or restlessness",
    "Thoughts of self-harm",
]

PSS10 = [
    "Upset because of unexpected events",
    "Unable to control important things in life",
    "Felt nervous and stressed",
    "Confident about handling problems",
    "Things going your way",
    "Could not cope with tasks",
    "Able to control irritations",
    "Felt on top of things",
    "Angry because things were out of control",
    "Felt difficulties piling up too high",
]

ALL = {
    "Anxiety": GAD7,
    "Depression": PHQ9,
    "Stress": PSS10,
}

# -------------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------------
def predict_condition(model, encoder, answers):
    df = pd.DataFrame([answers])
    df = align_features(df, model)

    try:
        pred = model.predict(df)[0]
        if encoder:
            pred = encoder.inverse_transform([pred])[0]
    except:
        pred = "Unknown"

    return str(pred)


# -------------------------------------------------------------
# SEVERITY MAPPING
# -------------------------------------------------------------
def severity_value(label):
    label = label.lower()
    if "minimal" in label: return 1
    if "mild" in label: return 2
    if "moderate" in label: return 3
    if "severe" in label: return 4
    return 0


# -------------------------------------------------------------
# SUGGESTIONS
# -------------------------------------------------------------
def suggestion_text(condition, severity):
    if severity <= 1:
        return "আপনার অবস্থা স্থিতিশীল। নিয়মিত ঘুম, পানি পান, পরিবারে সময় কাটান।"
    if severity == 2:
        return "মাঝারি স্তরের সমস্যা। ঘুম ও রুটিন ঠিক করুন, বন্ধুর সাথে কথা বলুন।"
    if severity == 3:
        return "উচ্চ ঝুঁকি। কাউন্সেলিং নেওয়া উচিত, কাজ/স্টাডি লোড কমান।"
    if severity == 4:
        return "জরুরি সহায়তা প্রয়োজন। মানসিক স্বাস্থ্য বিশেষজ্ঞের সাথে কথা বলুন।\n\n📞 জাতীয় হেল্পলাইন: 16263\n📞 মনোরোগ সহায়তা: 09666774455"
    return "পর্যাপ্ত তথ্য নেই।"


# -------------------------------------------------------------
# UI — MAIN SCREEN
# -------------------------------------------------------------
st.title("🧠 Mental Health Classification & Support System")
st.caption("ML-based Screening • Bangla + English • Developed by Team Dual Core")

models = load_models()
encoders = load_encoders()

tab1, tab2 = st.tabs(["📋 Screening", "📊 Dashboard"])


# -------------------------------------------------------------
# TAB 1 — SCREENING
# -------------------------------------------------------------
with tab1:
    st.header("📝 Answer the following questions")

    user_answers = {}

    for cond, qs in ALL.items():
        st.subheader(f"{cond} Questionnaire")

        answers = []
        for i, q in enumerate(qs):
            val = st.slider(
                f"{q} (1=Not at all • 5=Nearly everyday)",
                1, 5, 3,
                key=f"{cond}_{i}"
            )
            answers.append(val)

        user_answers[cond] = answers

    if st.button("🔍 Predict Overall Mental Health"):
        anxiety_label = predict_condition(models["Anxiety"], encoders["Anxiety"], user_answers["Anxiety"])
        stress_label = predict_condition(models["Stress"], encoders["Stress"], user_answers["Stress"])
        depression_label = predict_condition(models["Depression"], encoders["Depression"], user_answers["Depression"])

        st.subheader("📌 Individual Predictions")
        st.write(f"**Anxiety:** {anxiety_label}")
        st.write(f"**Stress:** {stress_label}")
        st.write(f"**Depression:** {depression_label}")

        sev = {
            "Anxiety": severity_value(anxiety_label),
            "Stress": severity_value(stress_label),
            "Depression": severity_value(depression_label),
        }

        main_issue = max(sev, key=sev.get)

        st.success(f"### 🧭 Your Main Mental Health Concern: **{main_issue}**")

        st.markdown("### 🎯 Suggested Actions (Bangla)")
        st.info(suggestion_text(main_issue, sev[main_issue]))

        # Save to log
        log = pd.DataFrame([{
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Anxiety": anxiety_label,
            "Stress": stress_label,
            "Depression": depression_label,
            "Main_Issue": main_issue
        }])

        if os.path.exists("prediction_log.csv"):
            log.to_csv("prediction_log.csv", mode="a", header=False, index=False)
        else:
            log.to_csv("prediction_log.csv", index=False)

        st.success("Your response has been saved.")


# -------------------------------------------------------------
# TAB 2 — DASHBOARD
# -------------------------------------------------------------
with tab2:
    st.header("📊 Insights Dashboard")

    if not os.path.exists("prediction_log.csv"):
        st.warning("No data yet.")
    else:
        df = pd.read_csv("prediction_log.csv")
        st.dataframe(df)

        chart = df["Main_Issue"].value_counts().reset_index()
        chart.columns = ["Issue", "Count"]
        st.altair_chart(
            alt.Chart(chart).mark_bar().encode(
                x="Issue:N", y="Count:Q", color="Issue:N"
            ),
            use_container_width=True
        )


# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("🔧 Developed by **Team Dual Core** © All Rights Reserved")
