import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px

# ---------------------------------------------------------------
# 🧩 Compatibility patch
# ---------------------------------------------------------------
import sklearn.compose._column_transformer as ct
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    ct._RemainderColsList = _RemainderColsList

# ---------------------------------------------------------------
# 🌈 Page Config
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Mental Health System v2", page_icon="🧠", layout="wide")

tabs = st.tabs(["🔮 Prediction", "📊 Dashboard Analytics"])

# ---------------------------------------------------------------
# COMMON FUNCTIONS
# ---------------------------------------------------------------
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

RISK_PLAN = {
    "Anxiety": {
        0: ("Low", ["Maintain routine", "Sleep 7–9h", "Daily relaxation"]),
        1: ("Mild", ["Breathing 4-7-8", "Journal for 10 minutes"]),
        2: ("Moderate", ["Peer support", "Try mindfulness exercise"]),
        3: ("Severe", ["Reach counselor", "Immediate support recommended"])
    },
    "Stress": {
        0: ("Low", ["Take breaks", "Walk 15 minutes"]),
        1: ("Mild", ["Time blocking", "Prioritize 3 key tasks"]),
        2: ("Moderate", ["Meet advisor", "Try relaxation training"]),
        3: ("Severe", ["Contact student wellbeing center", "Crisis support if needed"])
    },
    "Depression": {
        0: ("Low", ["Stay socially active", "Practice gratitude"]),
        1: ("Mild", ["Plan one small positive activity per day"]),
        2: ("Moderate", ["Consider counseling", "Monitor mood daily"]),
        3: ("Severe", ["Contact mental health professional", "Create safety plan"])
    }
}

def interpret_risk(target, label_index):
    label_index = int(label_index) if isinstance(label_index, (np.int64, np.int32, float)) else 0
    if label_index in RISK_PLAN[target]:
        return RISK_PLAN[target][label_index]
    return ("Unknown", ["Consult professional for personalized support"])

# ---------------------------------------------------------------
# TAB 1 — PREDICTION INTERFACE
# ---------------------------------------------------------------
with tabs[0]:
    st.title("🧠 AI-based Mental Health Detection & Support System")
    st.caption("Developed for Thesis & Real-world Use | 2025")

    target = st.selectbox("Select what you want to predict:", ["Anxiety", "Stress", "Depression"])
    model, encoder = load_model(target)
    st.success(f"✅ {target} model loaded successfully!")

    # Questions
    if target == "Anxiety":
        st.subheader("🧠 Anxiety Screening (GAD-7 Scale)")
        questions = {
            "GAD1": "Feeling nervous, anxious, or on edge",
            "GAD2": "Not being able to stop or control worrying",
            "GAD3": "Worrying too much about different things",
            "GAD4": "Trouble relaxing",
            "GAD5": "Being so restless that it is hard to sit still",
            "GAD6": "Becoming easily annoyed or irritable",
            "GAD7": "Feeling afraid as if something awful might happen"
        }
    elif target == "Stress":
        st.subheader("😣 Stress Screening (PSS-10 Scale)")
        questions = {
            "PSS1": "Upset because of unexpected events",
            "PSS2": "Unable to control important things in life",
            "PSS3": "Felt nervous and stressed",
            "PSS4": "Confident about handling problems",
            "PSS5": "Things going your way",
            "PSS6": "Could not cope with all the things you had to do",
            "PSS7": "Able to control irritations in your life",
            "PSS8": "Felt on top of things",
            "PSS9": "Angry because things were out of control",
            "PSS10": "Felt difficulties piling up too high"
        }
    else:
        st.subheader("😔 Depression Screening (PHQ-9 Scale)")
        questions = {
            "PHQ1": "Little interest or pleasure in doing things",
            "PHQ2": "Feeling down, depressed, or hopeless",
            "PHQ3": "Trouble falling or staying asleep",
            "PHQ4": "Feeling tired or having little energy",
            "PHQ5": "Poor appetite or overeating",
            "PHQ6": "Feeling bad about yourself",
            "PHQ7": "Trouble concentrating",
            "PHQ8": "Moving/speaking slowly or being restless",
            "PHQ9": "Thoughts of self-harm or death"
        }

    # Questionnaire
    inputs = {}
    for key, q in questions.items():
        inputs[key] = st.slider(f"{q} (1 = Not at all, 5 = Nearly every day)", 1, 5, 3)

    # Dummy demographic columns
    for col in ['Age','Current_CGPA','Gender','University','Department','Academic_Year','waiver_or_scholarship']:
        inputs[col] = 0 if 'CGPA' in col else 'N/A'

    # Predict button
    if st.button("🔍 Predict Mental Health Status"):
        try:
            X = pd.DataFrame([inputs])
            pred_encoded = model.predict(X)[0]
            pred_label = encoder.inverse_transform([pred_encoded])[0]
            risk_tier, suggestions = interpret_risk(target, pred_encoded)
            st.success(f"🧩 Predicted: **{pred_label}**")
            st.info(f"**Risk Level:** {risk_tier}\n\n**Suggested Actions:** " + " • ".join(suggestions))
            # Log
            log = pd.DataFrame({
                "timestamp": [datetime.utcnow().isoformat()],
                "target": [target],
                "prediction": [pred_label],
                "risk": [risk_tier]
            })
            log.to_csv("prediction_log.csv", mode='a', header=not os.path.exists("prediction_log.csv"), index=False)
            st.toast("✅ Logged successfully!")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------
# TAB 2 — DASHBOARD ANALYTICS
# ---------------------------------------------------------------
with tabs[1]:
    st.title("📊 Mental Health Prediction Dashboard")
    if os.path.exists("prediction_log.csv"):
        df = pd.read_csv("prediction_log.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x="target", color="prediction", title="Prediction Distribution by Type")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(df, names="risk", title="Risk Tier Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📅 Daily Prediction Activity")
        df['date'] = df['timestamp'].dt.date
        fig3 = px.line(df.groupby(['date','target']).size().reset_index(name='count'),
                       x='date', y='count', color='target', markers=True,
                       title='Daily Prediction Count by Category')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No predictions logged yet. Make some predictions first!")
