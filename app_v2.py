# ==========================================
# 🌐 AI-based Mental Health Detection System (v2)
# Thesis + Real-world ready | 2025 Edition
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
from datetime import datetime

# -----------------------------
# 🔧 Page Config
# -----------------------------
st.set_page_config(
    page_title="AI-based Mental Health Detection & Support System",
    layout="wide",
    page_icon="🧠"
)

# -----------------------------
# 🌙 Custom Dark Theme Styling
# -----------------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: #FAFAFA;}
.sidebar .sidebar-content {background-color: #1E1E1E;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.stButton>button {border-radius: 10px; background-color: #111; color: white; font-weight: 600;}
.stButton>button:hover {background-color: #2E8B57; color: white;}
.stProgress .st-bo {background-color: #00FFAA;}
h1, h2, h3, h4, h5 {color: #E0E0E0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 📦 Helper Functions
# -----------------------------
@st.cache_resource
def load_model(target_name):
    """Load specific model + encoder"""
    models = {
        "Anxiety": "final_anxiety_model.joblib",
        "Stress": "final_stress_model.joblib",
        "Depression": "best_model_Depression_Label_SVM.joblib",
    }
    encoders = {
        "Anxiety": "final_anxiety_encoder.joblib",
        "Stress": "final_stress_encoder.joblib",
        "Depression": "final_depression_encoder (1).joblib",
    }

    try:
        model_path = models[target_name]
        encoder_path = encoders[target_name]

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Model or encoder missing: {model_path}, {encoder_path}")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder

    except Exception as e:
        st.error(f"❌ Could not load {target_name} files. {type(e).__name__}: {e}")
        return None, None


def save_prediction_log(data):
    """Append prediction data to CSV log"""
    df = pd.DataFrame([data])
    if os.path.exists("prediction_log.csv"):
        df.to_csv("prediction_log.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("prediction_log.csv", index=False)


def risk_tier_map(label):
    mapping = {
        "Minimal Anxiety": "Low",
        "Mild Anxiety": "Moderate",
        "Moderate Anxiety": "High",
        "Severe Anxiety": "Critical",
        "Minimal Stress": "Low",
        "Mild Stress": "Moderate",
        "Moderate Stress": "High",
        "Severe Stress": "Critical",
        "Minimal Depression": "Low",
        "Mild Depression": "Moderate",
        "Moderate Depression": "High",
        "Severe Depression": "Critical"
    }
    return mapping.get(label, "Unknown")


# -----------------------------
# 🧠 Sidebar Navigation
# -----------------------------
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to:", ["🧩 Prediction", "📊 Dashboard Analytics"])

# -----------------------------
# 🧩 Prediction Section
# -----------------------------
if menu == "🧩 Prediction":
    st.title("🧠 AI-based Mental Health Detection & Support System")
    st.caption("Developed for Thesis & Real-world Use | 2025")

    target = st.selectbox("Select what you want to predict:", ["Anxiety", "Stress", "Depression"])

    model, encoder = load_model(target)
    if model and encoder:
        st.success(f"✅ {target} model loaded successfully!")

        # ---------- Questionnaires ----------
        st.markdown(f"### 🧾 {target} Screening Form")
        st.info(f"Please rate the following statements on a scale of 1 (Not at all) to 5 (Nearly every day).")

        if target == "Anxiety":
            questions = [
                "Feeling nervous, anxious, or on edge",
                "Not being able to stop or control worrying",
                "Worrying too much about different things",
                "Trouble relaxing",
                "Being so restless that it is hard to sit still",
                "Becoming easily annoyed or irritable",
                "Feeling afraid as if something awful might happen"
            ]
        elif target == "Stress":
            questions = [
                "Upset because of unexpected events",
                "Unable to control important things in life",
                "Felt nervous and stressed",
                "Confident about handling problems",
                "Things going your way",
                "Could not cope with all you had to do",
                "Able to control irritations in your life",
                "Felt on top of things",
                "Angry because things were out of control",
                "Felt difficulties piling up too high"
            ]
        else:  # Depression
            questions = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep, or sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself or a failure",
                "Trouble concentrating on things",
                "Moving/speaking slowly or restlessness",
                "Thoughts of self-harm or death"
            ]

        # Sliders for each question
        responses = [st.slider(q, 1, 5, 3) for q in questions]

        if st.button("🔍 Predict Mental Health Status"):
            try:
                input_df = pd.DataFrame([responses])
                prediction = model.predict(input_df)[0]
                decoded_label = encoder.inverse_transform([prediction])[0]
                risk = risk_tier_map(decoded_label)

                st.success(f"✅ Predicted: {decoded_label}")
                st.info(f"Risk Level: {risk}")

                if risk == "Low":
                    advice = "Maintain routine • Sleep 7–9h • Daily relaxation"
                elif risk == "Moderate":
                    advice = "Consider mild therapy • Exercise • Reduce workload"
                elif risk == "High":
                    advice = "Seek counseling • Balance rest • Journaling"
                else:
                    advice = "Consult a psychologist or mental health expert immediately"

                st.markdown(f"**Suggested Actions:** {advice}")

                save_prediction_log({
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "target": target,
                    "predicted_label": decoded_label,
                    "risk_tier": risk
                })

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# 📊 Dashboard Analytics
# -----------------------------
else:
    st.title("📊 Dashboard Analytics")

    if not os.path.exists("prediction_log.csv"):
        st.warning("No predictions yet! Make some predictions first.")
    else:
        log_df = pd.read_csv("prediction_log.csv")

        st.subheader("📅 Recent Prediction History")
        st.dataframe(log_df.tail(10), use_container_width=True)

        st.markdown("---")

        if "datetime" in log_df.columns and "risk_tier" in log_df.columns:
            # Chart 1: Daily trend
            log_df["datetime"] = pd.to_datetime(log_df["datetime"])
            trend = log_df.groupby(log_df["datetime"].dt.date)["risk_tier"].count().reset_index()
            trend.columns = ["Date", "Predictions"]

            chart1 = alt.Chart(trend).mark_line(point=True).encode(
                x="Date:T",
                y="Predictions:Q",
                tooltip=["Date", "Predictions"]
            ).properties(title="Daily Prediction Trend")

            st.altair_chart(chart1, use_container_width=True)

            # Chart 2: Distribution by Risk Tier
            if "risk_tier" in log_df.columns:
                risk_counts = log_df["risk_tier"].value_counts().reset_index()
                risk_counts.columns = ["Risk Tier", "Count"]
                chart2 = alt.Chart(risk_counts).mark_bar().encode(
                    x=alt.X("Risk Tier:N", sort="-y"),
                    y="Count:Q",
                    tooltip=["Risk Tier", "Count"]
                ).properties(title="Distribution by Risk Level")

                st.altair_chart(chart2, use_container_width=True)
        else:
            st.warning("Prediction log missing required columns.")
