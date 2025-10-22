# ==========================================
# 🧠 AI-based Mental Health Detection & Support System (v3)
# Thesis + Real-World Edition | 2025
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import altair as alt
from datetime import datetime

# -----------------------------
# ⚙️ App Setup
# -----------------------------
st.set_page_config(page_title="AI-based Mental Health Detection System", layout="wide", page_icon="🧠")

st.markdown("""
<style>
body {background-color:#0E1117;color:#FAFAFA;}
h1,h2,h3,h4,h5{color:#E0E0E0;}
.stButton>button{background:#111;color:white;font-weight:600;border-radius:10px;}
.stButton>button:hover{background:#2E8B57;color:white;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧩 Model Loader
# -----------------------------
@st.cache_resource
def load_model(target):
    """Load model + encoder for selected target safely."""
    model_files = {
        "Anxiety": "final_anxiety_model.joblib",
        "Stress": "final_stress_model.joblib",
        "Depression": "best_model_Depression_Label_SVM.joblib"
    }
    encoder_files = {
        "Anxiety": "final_anxiety_encoder.joblib",
        "Stress": "final_stress_encoder.joblib",
        "Depression": "final_depression_encoder (1).joblib"
    }

    m_path = model_files.get(target)
    e_path = encoder_files.get(target)
    if not os.path.exists(m_path) or not os.path.exists(e_path):
        raise FileNotFoundError(f"Missing: {m_path} or {e_path}")

    model = joblib.load(m_path)
    encoder = joblib.load(e_path)
    return model, encoder


def risk_tier_map(label):
    tiers = {
        "Minimal": "Low", "Mild": "Moderate",
        "Moderate": "High", "Severe": "Critical"
    }
    for k,v in tiers.items():
        if k.lower() in label.lower(): return v
    return "Unknown"


def save_prediction_log(row):
    df = pd.DataFrame([row])
    if os.path.exists("prediction_log.csv"):
        df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_log.csv", index=False)


# -----------------------------
# 🧭 Navigation
# -----------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🧩 Prediction", "📊 Dashboard"])

# -----------------------------
# 🧩 Prediction Page
# -----------------------------
if page == "🧩 Prediction":
    st.title("🧠 AI-based Mental Health Detection & Support System")
    st.caption("Developed for Thesis & Real-World Use | 2025")

    target = st.selectbox("Select what you want to predict:", ["Anxiety","Stress","Depression"])

    try:
        model, encoder = load_model(target)
        st.success(f"✅ {target} model loaded successfully.")
    except Exception as e:
        st.error(f"⚠️ Could not load {target} model: {e}")
        st.stop()

    st.markdown(f"### 🧾 {target} Screening Form")
    st.info("Rate each statement from 1 (Not at all) to 5 (Nearly every day).")

    questions = {
        "Anxiety":[
            "Feeling nervous, anxious, or on edge",
            "Not being able to stop or control worrying",
            "Worrying too much about different things",
            "Trouble relaxing","Being so restless that it is hard to sit still",
            "Becoming easily annoyed or irritable",
            "Feeling afraid as if something awful might happen"],
        "Stress":[
            "Upset because of unexpected events",
            "Unable to control important things in life",
            "Felt nervous and stressed",
            "Confident about handling problems",
            "Things going your way",
            "Could not cope with all the things you had to do",
            "Able to control irritations in your life",
            "Felt on top of things",
            "Angry because things were out of control",
            "Felt difficulties piling up too high"],
        "Depression":[
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or a failure",
            "Trouble concentrating on things",
            "Moving/speaking slowly or restlessness",
            "Thoughts of self-harm or death"]
    }[target]

    answers = [st.slider(q,1,5,3) for q in questions]

    if st.button("🔍 Predict Mental Health Status"):
        try:
            df = pd.DataFrame([answers])
            pred = model.predict(df)[0]
            decoded = encoder.inverse_transform([pred])[0] if encoder else str(pred)
            risk = risk_tier_map(decoded)

            st.success(f"🎯 Predicted: **{decoded}**")
            st.info(f"🩺 Risk Level: **{risk}**")

            advice = {
                "Low":"Maintain routine • Sleep 7-9 h • Daily relaxation",
                "Moderate":"Exercise • Journaling • Healthy diet",
                "High":"Seek counseling • Reduce workload • Mindfulness",
                "Critical":"Consult professional immediately • Support network"
            }.get(risk,"Monitor your mental state regularly")
            st.markdown(f"**Suggested Actions:** {advice}")

            save_prediction_log({
                "datetime":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target":target,"predicted_label":decoded,"risk_tier":risk
            })
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# 📊 Dashboard
# -----------------------------
else:
    st.title("📊 Mental Health Dashboard")
    if not os.path.exists("prediction_log.csv"):
        st.warning("No predictions yet! Perform some first.")
        st.stop()

    df = pd.read_csv("prediction_log.csv")
    st.dataframe(df.tail(10), use_container_width=True)

    # --- Stats Summary ---
    st.subheader("📈 Summary")
    total = len(df)
    tiers = df["risk_tier"].value_counts(normalize=True).mul(100)
    for tier in ["Low","Moderate","High","Critical"]:
        val = float(tiers.get(tier,0))
        st.progress(int(val))
        st.markdown(f"**{tier}: {val:.1f}%**")

    # --- Trend Chart ---
    df["datetime"] = pd.to_datetime(df["datetime"])
    trend = df.groupby(df["datetime"].dt.date).size().reset_index(name="Predictions")
    st.altair_chart(
        alt.Chart(trend).mark_line(point=True,color="#00FFAA").encode(
            x="datetime:T",y="Predictions:Q"), use_container_width=True)

    # --- Distribution Chart ---
    dist = df["risk_tier"].value_counts().reset_index()
    dist.columns = ["Risk Tier","Count"]
    st.altair_chart(
        alt.Chart(dist).mark_bar().encode(
            x=alt.X("Risk Tier:N",sort="-y"),
            y="Count:Q",color="Risk Tier:N"), use_container_width=True)

    # --- Download Button ---
    st.download_button("⬇️ Download Prediction History", data=df.to_csv(index=False),
                       file_name="prediction_log.csv", mime="text/csv")
