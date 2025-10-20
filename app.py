# ============================================================
# 🌿 Mental Health Prediction App (Anxiety, Stress, Depression)
# Version: Final (Thesis + Deployment)
# Developed by: Ovi Sarker & BM Sabbir Hossen Riad (2025)
# ============================================================
import joblib
import sklearn
from sklearn.utils import _joblib

# ✅ Compatibility patch for sklearn deserialization
import sys
import sklearn.compose._column_transformer as ct
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    ct._RemainderColsList = _RemainderColsList

import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mental Health AI App", layout="wide")

# ------------------------------------------------------------
# 1️⃣ Model Loader
# ------------------------------------------------------------
@st.cache_resource
def load_model(target_name):
    models = {
        "Anxiety": "final_anxiety_model.joblib",
        "Stress": "final_stress_model.joblib",
        "Depression": "final_depression_model.joblib"
    }
    encoders = {
        "Anxiety": "final_anxiety_encoder.joblib",
        "Stress": "final_stress_encoder.joblib",
        "Depression": "final_depression_encoder.joblib"
    }
    model = joblib.load(models[target_name])
    encoder = joblib.load(encoders[target_name])
    return model, encoder

# ------------------------------------------------------------
# 2️⃣ Intervention Plan (Prediction → Action)
# ------------------------------------------------------------
RISK_PLAN = {
    "Anxiety": {
        "0": ("Low", ["Keep routine", "Sleep 7–9h", "Exercise 30min/day"]),
        "1": ("Mild", ["Practice 4-7-8 breathing", "Journal thoughts"]),
        "2": ("Moderate", ["Peer support", "Counseling signup"]),
        "3": ("Severe", ["Seek counselor", "Follow-up within 48h"])
    },
    "Stress": {
        "0": ("Low", ["Pomodoro 25/5", "Take short walks"]),
        "1": ("Mild", ["Time-block tasks", "Talk to mentor"]),
        "2": ("Moderate", ["Meet advisor", "Use stress worksheet"]),
        "3": ("Severe", ["Contact student services", "Health check-in"])
    },
    "Depression": {
        "0": ("Low", ["Gratitude list", "Social activity"]),
        "1": ("Mild", ["Small daily tasks", "Positive reinforcement"]),
        "2": ("Moderate", ["Counseling referral", "Follow-up in 72h"]),
        "3": ("Severe", ["Immediate support", "Safety plan review"])
    }
}

def get_risk_actions(target, label):
    plan = RISK_PLAN.get(target, {})
    return plan.get(str(label), ("Unknown", ["Consult counselor"]))

# ------------------------------------------------------------
# 3️⃣ Explainability (XAI using SHAP)
# ------------------------------------------------------------
@st.cache_resource
def get_explainer(model):
    clf = model.named_steps["clf"]
    explainer = shap.Explainer(clf)
    return explainer

# ------------------------------------------------------------
# 4️⃣ AI Assistant (FAQ)
# ------------------------------------------------------------
FAQ = {
    "breathing": "Try 4–7–8 breathing: inhale 4s, hold 7s, exhale 8s, 4 rounds.",
    "sleep": "Aim 7–9h sleep; consistent schedule; reduce screen time.",
    "study": "Use Pomodoro 25m focus + 5m break for better productivity.",
    "stress": "Short walks, music, and deep breathing can help reduce stress.",
    "support": "Consider reaching out to your university counseling service.",
}

def ai_assistant(query):
    q = query.lower()
    for k, v in FAQ.items():
        if k in q:
            return v
    return "I can assist with breathing, sleep, study, stress, and support tips."

# ------------------------------------------------------------
# 5️⃣ Log Predictions (Safe)
# ------------------------------------------------------------
def log_prediction(data, target, pred_label, risk):
    log = pd.DataFrame({
        "timestamp": [datetime.utcnow().isoformat()],
        "target": [target],
        "prediction": [pred_label],
        "risk_tier": [risk]
    })
    if os.path.exists("prediction_log.csv"):
        log.to_csv("prediction_log.csv", mode="a", header=False, index=False)
    else:
        log.to_csv("prediction_log.csv", index=False)

# ------------------------------------------------------------
# 🧠 6️⃣ Streamlit UI
# ------------------------------------------------------------
st.title("🧠 AI-based Mental Health Detection & Support System")
st.markdown("### Developed for Thesis & Real-world Use | 2025")

target = st.selectbox("Select what you want to predict:", ["Anxiety", "Stress", "Depression"])

model, encoder = load_model(target)
st.success(f"✅ {target} Model Loaded Successfully!")

st.markdown("#### Enter your responses:")
cols = st.columns(3)

# Dynamic feature input based on target
if target == "Anxiety":
    questions = [f"GAD{i}" for i in range(1, 8)]
elif target == "Stress":
    questions = [f"PSS{i}" for i in range(1, 11)]
else:
    questions = [f"PHQ{i}" for i in range(1, 10)]

responses = []
for i, q in enumerate(questions):
    with cols[i % 3]:
        val = st.slider(f"{q} (1=Never, 5=Always)", 1, 5, 3)
        responses.append(val)

if st.button("🔍 Predict Mental Health Status"):
    X = pd.DataFrame([responses], columns=questions)
    pred_encoded = model.predict(X)[0]
    pred_label = encoder.inverse_transform([int(pred_encoded)])[0]
    risk, acts = get_risk_actions(target, pred_encoded)

    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Class:** {pred_label}")
    st.write(f"**Risk Level:** {risk}")
    st.write("**Suggested Actions:**")
    for act in acts:
        st.markdown(f"- {act}")

    log_prediction(X, target, pred_label, risk)

    # XAI Section
    try:
        st.subheader("🔍 Explainability (XAI)")
        explainer = get_explainer(model)
        shap_values = explainer(X)
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.info(f"Explainability not available for this model. ({e})")

# ------------------------------------------------------------
# 7️⃣ AI Assistant (Chat Section)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Ask the AI Assistant")
query = st.text_input("Ask something (e.g., 'How to reduce stress before exams?')")
if st.button("Ask"):
    st.write(ai_assistant(query))

# ------------------------------------------------------------
# 8️⃣ Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational and awareness purposes only. It is not a medical diagnostic system. For urgent mental health concerns, contact a qualified professional or local helpline.")
