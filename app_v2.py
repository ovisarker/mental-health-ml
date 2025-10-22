# ===========================
# AI-based Mental Health Detection & Support System (v2)
# - Real-time forms (GAD-7, PSS-10, PHQ-9) + demographics
# - Loads joblib pipelines + separate label encoders
# - Risk mapping + intervention suggestions
# - Prediction logging to CSV
# - Dashboard: daily trend & risk distribution
# ===========================

import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# ---------------------------
# Page config & style
# ---------------------------
st.set_page_config(
    page_title="AI Mental Health Detection",
    page_icon="🧠",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stAlert { font-size: 0.95rem; }
    .stButton>button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Risk mapping / next steps
# ---------------------------
RISK_PLAN = {
    "Anxiety": {
        "tiers": {
            "Minimal Anxiety": ("Low", ["Maintain routine", "Sleep 7–9h", "30m physical activity"]),
            "Mild Anxiety": ("Mild", ["4-7-8 breathing", "10m journaling"]),
            "Moderate Anxiety": ("Moderate", ["Peer support", "Counseling sign-up"]),
            "Severe Anxiety": ("Severe", ["Contact counselor", "Follow-up within 48h"]),
            # numeric fallback
            "0": ("Low", ["Maintain routine", "Sleep 7–9h", "30m activity"]),
            "1": ("Mild", ["4-7-8 breathing", "10m journaling"]),
            "2": ("Moderate", ["Peer support", "Counseling sign-up"]),
            "3": ("Severe", ["Contact counselor", "Follow-up within 48h"]),
        }
    },
    "Stress": {
        "tiers": {
            "Low Stress": ("Low", ["Pomodoro 25/5", "15m walk"]),
            "Moderate Stress": ("Moderate", ["Time blocking", "Say no to one task"]),
            "High Perceived Stress": ("Severe", ["Escalate to student services", "Wellbeing check-in"]),
            # sometimes datasets carry older names:
            "Moderate": ("Moderate", ["Time blocking", "Say no to one task"]),
            # numeric fallback
            "0": ("Low", ["Pomodoro 25/5", "15m walk"]),
            "1": ("Moderate", ["Time blocking", "Say no to one task"]),
            "2": ("Severe", ["Escalate to services", "Wellbeing check-in"]),
            "3": ("Severe", ["Escalate to services", "Wellbeing check-in"]),
        }
    },
    "Depression": {
        "tiers": {
            "No Depression": ("Low", ["Gratitude (3 items)", "10m social contact"]),
            "Minimal Depression": ("Low", ["Gratitude (3 items)", "10m social contact"]),
            "Mild Depression": ("Mild", ["Small task/day (behavioral activation)"]),
            "Moderate Depression": ("Moderate", ["Counseling referral", "Follow-up in 72h"]),
            "Moderately Severe Depression": ("Severe", ["Immediate support", "Safety plan review"]),
            "Severe Depression": ("Severe", ["Immediate support", "Safety plan review"]),
            # numeric fallback
            "0": ("Low", ["Gratitude", "10m social contact"]),
            "1": ("Mild", ["Small task/day"]),
            "2": ("Moderate", ["Counseling referral", "Follow-up in 72h"]),
            "3": ("Severe", ["Immediate support", "Safety plan review"]),
            "4": ("Severe", ["Immediate support", "Safety plan review"]),
            "5": ("Severe", ["Immediate support", "Safety plan review"]),
        }
    },
}


def to_risk(target: str, label: str | int) -> tuple[str, list[str]]:
    key = str(label)
    plan = RISK_PLAN.get(target, {}).get("tiers", {})
    if key in plan:
        return plan[key]
    # a few generic fallbacks
    low = ("Low", ["Maintain healthy routine", "Sleep 7–9h"])
    if isinstance(label, (int, np.integer)) and label <= 1:
        return low
    return ("Unknown", ["Consult professional for personalized support"])


# ---------------------------
# Model loading with robust debug
# ---------------------------
@st.cache_data(show_spinner=False)
def list_repo_files():
    files = []
    for f in sorted(os.listdir(".")):
        try:
            files.append((f, os.path.getsize(f)))
        except Exception:
            files.append((f, None))
    return files


@st.cache_resource(show_spinner=False)
def load_model(target_name: str):
    models = {
        "Anxiety": "final_anxiety_model.joblib",
        "Stress": "final_stress_model.joblib",
        "Depression": "final_depression_model.joblib",
    }
    encoders = {
        "Anxiety": "final_anxiety_encoder.joblib",
        "Stress": "final_stress_encoder.joblib",
        "Depression": "final_depression_encoder.joblib",
    }

    model_file = models[target_name]
    enc_file = encoders[target_name]

    # Show visible files to help diagnose path/size/LFS issues
    repo_listing = "\n".join([f"• {n}  ({s} bytes)" for n, s in list_repo_files()])
    with st.expander("🔎 Debug: files visible to the app (click to open)"):
        st.text(repo_listing)

    def _check(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found in repo root.")
        sz = os.path.getsize(path)
        if sz == 0:
            raise OSError(f"{path} is 0 bytes (bad upload). Re-upload from Colab.")
        return sz

    try:
        _check(model_file)
        _check(enc_file)
        model = joblib.load(model_file)
        encoder = joblib.load(enc_file)
        return model, encoder
    except Exception as e:
        st.error(f"❌ Could not load {target_name} files.\n{type(e).__name__}: {e}")
        st.code("Traceback:\n" + traceback.format_exc(), language="text")
        raise


# ---------------------------
# Input schemas
# ---------------------------
GAD7_ITEMS = [
    ("GAD1", "Feeling nervous, anxious, or on edge"),
    ("GAD2", "Not being able to stop or control worrying"),
    ("GAD3", "Worrying too much about different things"),
    ("GAD4", "Trouble relaxing"),
    ("GAD5", "Being restless (hard to sit still)"),
    ("GAD6", "Becoming easily annoyed or irritable"),
    ("GAD7", "Feeling afraid something awful might happen"),
]

PSS10_ITEMS = [
    ("PSS1", "Upset because of unexpected events"),
    ("PSS2", "Unable to control important things in life"),
    ("PSS3", "Felt nervous and stressed"),
    ("PSS4", "Confident about handling problems"),
    ("PSS5", "Things going your way"),
    ("PSS6", "Could not cope with all you had to do"),
    ("PSS7", "Able to control irritations in life"),
    ("PSS8", "Felt on top of things"),
    ("PSS9", "Angry because things were out of control"),
    ("PSS10", "Felt difficulties piling up too high"),
]

PHQ9_ITEMS = [
    ("PHQ1", "Little interest or pleasure in doing things"),
    ("PHQ2", "Feeling down, depressed, or hopeless"),
    ("PHQ3", "Trouble falling/staying asleep, or sleeping too much"),
    ("PHQ4", "Feeling tired or having little energy"),
    ("PHQ5", "Poor appetite or overeating"),
    ("PHQ6", "Feeling bad about yourself / failure feelings"),
    ("PHQ7", "Trouble concentrating"),
    ("PHQ8", "Moving/speaking slowly or fidgety/restless"),
    ("PHQ9", "Thoughts that you would be better off dead / self-harm"),
]

# Demographics your training used (names must match training columns)
DEMOGRAPHICS = [
    ("Age", 18, 60, 20),
    ("Current_CGPA", 0.0, 4.0, 3.2),
]
DEMOGRAPHICS_SELECT = [
    ("Gender", ["Male", "Female", "Other"], "Male"),
    ("University", ["DIU", "Other"], "DIU"),
    ("Department", ["CSE", "EEE", "BBA", "Other"], "CSE"),
    ("Academic_Year", ["1st", "2nd", "3rd", "4th", "Other"], "2nd"),
    ("waiver_or_scholarship", ["Yes", "No"], "No"),
]


def form_section(target: str) -> pd.DataFrame:
    """
    Render the form inputs, return a one-row DataFrame with all expected columns.
    Scale: 1..5 for all symptom items.
    """
    cols = {}
    st.markdown("**Answer the following (1 = Not at all, 5 = Nearly every day):**")

    if target == "Anxiety":
        items = GAD7_ITEMS
    elif target == "Stress":
        items = PSS10_ITEMS
    else:
        items = PHQ9_ITEMS

    # sliders for symptoms
    for code, label in items:
        val = st.slider(label, 1, 5, 1, key=f"{target}_{code}")
        cols[code] = val

    st.markdown("---")
    st.markdown("**Demographics** (used by the model; your input is not stored with identity):")

    for name, lo, hi, default in DEMOGRAPHICS:
        if isinstance(default, float):
            val = st.number_input(name, min_value=float(lo), max_value=float(hi), value=float(default), step=0.1)
        else:
            val = st.number_input(name, min_value=int(lo), max_value=int(hi), value=int(default), step=1)
        cols[name] = val

    for name, options, default in DEMOGRAPHICS_SELECT:
        choice = st.selectbox(name, options, index=options.index(default), key=f"{target}_{name}")
        cols[name] = choice

    # Put everything into one row DataFrame
    df = pd.DataFrame([cols])
    return df


# ---------------------------
# Predict helper
# ---------------------------
def predict_single(target: str, model, encoder, df_row: pd.DataFrame) -> tuple[str, str, list[str]]:
    """
    Returns: (label_str, risk_name, actions)
    """
    # model outputs can be numeric or strings depending on training;
    # try numeric first, then fallback to string
    y_pred = model.predict(df_row)[0]
    label_str = None

    # try decoding with encoder (if numeric codes)
    if encoder is not None and hasattr(encoder, "inverse_transform"):
        try:
            code = int(y_pred)
            label_str = encoder.inverse_transform([code])[0]
        except Exception:
            # maybe the model already outputs a string
            label_str = str(y_pred)
    else:
        label_str = str(y_pred)

    risk, actions = to_risk(target, label_str)
    return label_str, risk, actions


# ---------------------------
# Logging
# ---------------------------
LOG_PATH = "prediction_log.csv"


def append_log(target: str, label: str, risk: str):
    row = pd.DataFrame(
        {
            "timestamp": [datetime.utcnow().isoformat()],
            "target": [target],
            "prediction": [label],
            "risk_tier": [risk],
        }
    )
    if os.path.exists(LOG_PATH):
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)


# ---------------------------
# UI
# ---------------------------
st.title("AI-based Mental Health Detection & Support System")
st.caption("Developed for Thesis & Real-world Use | 2025")

tabs = st.tabs(["🔮 Prediction", "📊 Dashboard Analytics"])

with tabs[0]:
    target = st.selectbox("Select what you want to predict:", ["Anxiety", "Stress", "Depression"])
    try:
        model, encoder = load_model(target)
        st.success(f"{target} model loaded successfully!")
    except Exception:
        st.stop()  # loader already showed the exact problem

    st.markdown("**Research prototype — not medical advice. For urgent concerns, contact local crisis services.**")
    st.markdown("---")

    df_in = form_section(target)

    # predict button
    if st.button("🔍 Predict Mental Health Status"):
        try:
            label, risk, actions = predict_single(target, model, encoder, df_in)
            st.success(f"Predicted: **{label}**")
            st.info(f"Risk Level: **{risk}**")
            st.write("**Suggested Actions:** " + " • ".join(actions))
            append_log(target, label, risk)
        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc(), language="text")

with tabs[1]:
    st.header("Dashboard Analytics")
    st.caption("Aggregated from prediction_log.csv")

    if not os.path.exists(LOG_PATH):
        st.info("No predictions logged yet. Make a prediction first.")
        st.stop()

    try:
        log_df = pd.read_csv(LOG_PATH)
        if "timestamp" in log_df.columns:
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
            log_df["date"] = log_df["timestamp"].dt.date
    except Exception as e:
        st.error(f"Could not read {LOG_PATH}: {e}")
        st.stop()

    if len(log_df) == 0:
        st.info("Log is empty yet.")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        tgt = st.selectbox("Target", sorted(log_df["target"].unique()))
    dff = log_df[log_df["target"] == tgt].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Prediction Count")
        by_day = dff.groupby("date")["prediction"].count().reset_index()
        fig = px.line(by_day, x="date", y="prediction", markers=True, labels={"prediction": "# Predictions"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Tier Distribution")
        if "risk_tier" in dff.columns:
            pie = dff["risk_tier"].value_counts().reset_index()
            pie.columns = ["risk_tier", "count"]
            fig2 = px.pie(pie, names="risk_tier", values="count", hole=0.45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No risk_tier column found in the log.")
