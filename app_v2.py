# app_v2.py
# AI-based Mental Health Detection & Support System (Anxiety / Stress / Depression)
# v2: robust model/encoder loading, sklearn pickle compatibility, logging, and dashboard

import os
import io
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# -----------------------------
# 0) Fix for scikit-learn pickle "_RemainderColsList" error
#    (must run BEFORE any joblib.load)
# -----------------------------
try:
    from sklearn.compose import _column_transformer as _ct_patch
    if not hasattr(_ct_patch, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compat stub for older pickles with ColumnTransformer remainder list."""
            pass
        _ct_patch._RemainderColsList = _RemainderColsList  # monkey patch
except Exception:
    # If sklearn not available yet, we'll still try to load later; this is best-effort.
    pass

# -----------------------------
# 1) Page setup
# -----------------------------
st.set_page_config(
    page_title="AI-based Mental Health Detection & Support System",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI-based Mental Health Detection & Support System")
st.caption("Developed for Thesis & Real-world Use | 2025")

# -----------------------------
# 2) Files / Models mapping
# -----------------------------
MODEL_FILES = {
    "Anxiety": "final_anxiety_model.joblib",
    "Stress": "final_stress_model.joblib",
    "Depression": "final_depression_model.joblib",
}
ENCODER_FILES = {
    "Anxiety": "final_anxiety_encoder.joblib",
    "Stress": "final_stress_encoder.joblib",
    "Depression": "final_depression_encoder.joblib",
}
LOG_PATH = "prediction_log.csv"  # will be created in working dir

# -----------------------------
# 3) Risk plan (simple, non-clinical)
# -----------------------------
RISK_PLAN = {
    "Anxiety": {
        "0": ("Low", ["Maintain routine", "Sleep 7–9h", "Daily relaxation"]),
        "1": ("Mild", ["4-7-8 breathing", "10m journaling"]),
        "2": ("Moderate", ["Peer support", "Counseling sign-up"]),
        "3": ("Severe", ["Contact counselor", "Follow-up within 48h"]),
        # fallback
        "Minimal Anxiety": ("Low", ["Maintain routine"]),
        "Mild Anxiety": ("Mild", ["Breathing practice"]),
        "Moderate Anxiety": ("Moderate", ["Peer support / counseling"]),
        "Severe Anxiety": ("Severe", ["Contact counselor"]),
    },
    "Stress": {
        "0": ("Low", ["Pomodoro 25/5", "Walk 15m"]),
        "1": ("Mild", ["Time-blocking", "Say no to one task"]),
        "2": ("Moderate", ["Advisor meeting", "Brief CBT worksheet"]),
        "3": ("Severe", ["Escalate to services", "Wellbeing check-in"]),
        # common text labels
        "Low Stress": ("Low", ["Self-care routine"]),
        "Moderate Stress": ("Moderate", ["Planning + support"]),
        "High Perceived Stress": ("Severe", ["Contact services"]),
        "High": ("Severe", ["Contact services"]),
        "Moderate": ("Moderate", ["Planning + support"]),
    },
    "Depression": {
        "0": ("Low", ["Gratitude x3", "Social contact 10m"]),
        "1": ("Mild", ["1 small activity per day"]),
        "2": ("Moderate", ["Counseling referral", "Follow-up in 72h"]),
        "3": ("Severe", ["Immediate support", "Safety plan"]),
        # PHQ-like labels
        "No Depression": ("Low", ["Maintain routine"]),
        "Minimal Depression": ("Low", ["Sleep hygiene"]),
        "Mild Depression": ("Mild", ["Activation exercise"]),
        "Moderate Depression": ("Moderate", ["Counseling referral"]),
        "Moderately Severe Depression": ("Severe", ["Escalate & support"]),
        "Severe Depression": ("Severe", ["Immediate support"]),
    },
}

# -----------------------------
# 4) Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(target_name: str):
    """
    Robust loader:
    - returns (pipeline, encoder or None, info_text)
    - raises readable errors
    """
    model_file = MODEL_FILES.get(target_name)
    enc_file = ENCODER_FILES.get(target_name)

    if model_file is None:
        raise KeyError(f"Unknown target: {target_name}")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Encoder is optional (we’ll work without it if model classes_ are strings)
    encoder = None
    if enc_file and os.path.exists(enc_file):
        try:
            encoder = joblib.load(enc_file)
        except Exception as e:
            st.warning(f"Encoder load failed ({enc_file}): {e}")

    # Load model (after sklearn monkey-patch above)
    model = joblib.load(model_file)

    # Info string for the debug panel
    info = f"✅ Loaded {model_file}"
    if encoder is not None:
        info += f" & {enc_file}"
    else:
        info += " (no encoder)"

    return model, encoder, info


def list_repo_files():
    """Show local files & sizes to help debugging deployments."""
    items = []
    for name in sorted(os.listdir(".")):
        try:
            size = os.path.getsize(name)
            items.append((name, size))
        except Exception:
            items.append((name, None))
    return items


def classes_to_label(pred, model, encoder):
    """
    Convert predicted class -> human label.
    Priority:
      1) If encoder exists and was trained on integer codes, try inverse_transform.
      2) Else, if model.classes_ are strings, map index -> string.
      3) Else, return the raw value (stringify).
    """
    # If we have an encoder and prediction is integer-coded, try inverse_transform
    if encoder is not None:
        try:
            return encoder.inverse_transform(np.array([pred]))[0]
        except Exception:
            # If inverse_transform fails, fall back below
            pass

    # Many sklearn classifiers keep string labels directly in classes_
    try:
        if hasattr(model, "classes_"):
            classes = model.classes_
            # if pred is an index into classes (int) and classes are strings
            if isinstance(pred, (int, np.integer)) and pred >= 0 and pred < len(classes):
                cls = classes[int(pred)]
                # turn bytes -> str if ever needed
                return cls.decode() if isinstance(cls, bytes) else str(cls)
            # else, maybe the model already returned a string class
            return str(pred)
    except Exception:
        pass

    # final fallback
    return str(pred)


def to_risk(target_name: str, label: str):
    plan = RISK_PLAN.get(target_name, {})
    # exact match
    if label in plan:
        return plan[label]
    # numeric-string fallback
    key = str(label)
    return plan.get(key, ("Unknown", ["Consult professional for personalized support"]))


def append_log(row_dict):
    """Append a single-row dict to prediction_log.csv (safe, minimal)."""
    df = pd.DataFrame([row_dict])
    try:
        if os.path.exists(LOG_PATH):
            df.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Could not write log file: {e}")


def ensure_cols(df_in: pd.DataFrame, expected_cols):
    """Guarantee that all expected columns exist (fill with NaN if missing)."""
    df = df_in.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected_cols]


# -----------------------------
# 5) UI – Tabs
# -----------------------------
tab_pred, tab_dash = st.tabs(["🔮 Prediction", "📊 Dashboard Analytics"])

with tab_pred:
    # Selector
    target = st.selectbox("Select what you want to predict:", ["Anxiety", "Stress", "Depression"])

    # Debug / visibility panel
    with st.expander("🐞 Debug: files visible to the app (click to open)", expanded=False):
        files = list_repo_files()
        for name, size in files:
            size_txt = f"({size} bytes)" if size is not None else "(size ?)"
            st.markdown(f"- **{name}**  {size_txt}")

    # Load model (robust)
    try:
        model, encoder, info = load_model(target)
        st.success(f"Model ready: {info}")
    except Exception as e:
        st.error(f"Could not load {target} files. {type(e).__name__}: {e}")
        st.stop()

    # --- Build simple, friendly inputs (GAD-7 / PSS-10 / PHQ-8~ish) ---
    st.write("")
    if target == "Anxiety":
        st.subheader("🩺 Anxiety Screening (GAD-7 style)")
        items = {
            "GAD1": "Feeling nervous, anxious, or on edge (1=Not at all, 5=Nearly every day)",
            "GAD2": "Not having control over worrying (1..5)",
            "GAD3": "Worrying too much about different things (1..5)",
            "GAD4": "Trouble relaxing (1..5)",
            "GAD5": "Restlessness / hard to sit still (1..5)",
            "GAD6": "Easily annoyed or irritable (1..5)",
            "GAD7": "Feeling afraid as if something awful might happen (1..5)",
        }
        vals = {}
        cols = st.columns(2)
        for i, (k, prompt) in enumerate(items.items()):
            with cols[i % 2]:
                vals[k] = st.slider(prompt, 1, 5, 1)

        # optional demographics (the pipeline should happily ignore unknowns)
        age = st.number_input("Age (optional)", min_value=10, max_value=100, value=21, step=1)
        cgpa = st.number_input("Current CGPA (optional)", min_value=0.0, max_value=4.0, value=3.2, step=0.1)
        gender = st.selectbox("Gender (optional)", ["N/A", "Male", "Female", "Other"], index=0)
        dept = st.text_input("Department (optional)", value="N/A")
        univ = st.text_input("University (optional)", value="N/A")
        year = st.text_input("Academic_Year (optional)", value="N/A")
        waiver = st.text_input("waiver_or_scholarship (optional)", value="N/A")

        df_in = pd.DataFrame([{
            **vals,
            "Age": age,
            "Current_CGPA": cgpa,
            "Gender": None if gender == "N/A" else gender,
            "Department": None if dept == "N/A" else dept,
            "University": None if univ == "N/A" else univ,
            "Academic_Year": None if year == "N/A" else year,
            "waiver_or_scholarship": None if waiver == "N/A" else waiver,
        }])

    elif target == "Stress":
        st.subheader("🩺 Stress Screening (PSS-10 style)")
        items = {
            "PSS1": "Upset because of unexpected events (1..5)",
            "PSS2": "Unable to control important things in life (1..5)",
            "PSS3": "Felt nervous and stressed (1..5)",
            "PSS4": "Confident about handling problems (1..5)",
            "PSS5": "Things going your way (1..5)",
            "PSS6": "Could not cope with all things you had to do (1..5)",
            "PSS7": "Able to control irritations (1..5)",
            "PSS8": "Felt on top of things (1..5)",
            "PSS9": "Angry because things out of control (1..5)",
            "PSS10": "Difficulties piling up too high (1..5)",
        }
        vals = {}
        cols = st.columns(2)
        for i, (k, prompt) in enumerate(items.items()):
            with cols[i % 2]:
                vals[k] = st.slider(prompt, 1, 5, 1)

        age = st.number_input("Age (optional)", min_value=10, max_value=100, value=21, step=1, key="age_s")
        cgpa = st.number_input("Current CGPA (optional)", min_value=0.0, max_value=4.0, value=3.2, step=0.1, key="cgpa_s")
        gender = st.selectbox("Gender (optional)", ["N/A", "Male", "Female", "Other"], index=0, key="gender_s")
        dept = st.text_input("Department (optional)", value="N/A", key="dept_s")
        univ = st.text_input("University (optional)", value="N/A", key="univ_s")
        year = st.text_input("Academic_Year (optional)", value="N/A", key="year_s")
        waiver = st.text_input("waiver_or_scholarship (optional)", value="N/A", key="waiver_s")

        df_in = pd.DataFrame([{
            **vals,
            "Age": age,
            "Current_CGPA": cgpa,
            "Gender": None if gender == "N/A" else gender,
            "Department": None if dept == "N/A" else dept,
            "University": None if univ == "N/A" else univ,
            "Academic_Year": None if year == "N/A" else year,
            "waiver_or_scholarship": None if waiver == "N/A" else waiver,
        }])

    else:  # Depression
        st.subheader("🩺 Depression Screening (PHQ-style)")
        # keep it generic – the model only needs consistent feature names
        items = {
            "PHQ1": "Little interest or pleasure in doing things (1..5)",
            "PHQ2": "Feeling down, depressed, or hopeless (1..5)",
            "PHQ3": "Trouble sleeping / sleeping too much (1..5)",
            "PHQ4": "Feeling tired or having little energy (1..5)",
            "PHQ5": "Poor appetite or overeating (1..5)",
            "PHQ6": "Feeling bad about yourself (1..5)",
            "PHQ7": "Trouble concentrating (1..5)",
            "PHQ8": "Moving/speaking slowly or restless (1..5)",
        }
        vals = {}
        cols = st.columns(2)
        for i, (k, prompt) in enumerate(items.items()):
            with cols[i % 2]:
                vals[k] = st.slider(prompt, 1, 5, 1)

        age = st.number_input("Age (optional)", min_value=10, max_value=100, value=21, step=1, key="age_d")
        cgpa = st.number_input("Current CGPA (optional)", min_value=0.0, max_value=4.0, value=3.2, step=0.1, key="cgpa_d")
        gender = st.selectbox("Gender (optional)", ["N/A", "Male", "Female", "Other"], index=0, key="gender_d")
        dept = st.text_input("Department (optional)", value="N/A", key="dept_d")
        univ = st.text_input("University (optional)", value="N/A", key="univ_d")
        year = st.text_input("Academic_Year (optional)", value="N/A", key="year_d")
        waiver = st.text_input("waiver_or_scholarship (optional)", value="N/A", key="waiver_d")

        df_in = pd.DataFrame([{
            **vals,
            "Age": age,
            "Current_CGPA": cgpa,
            "Gender": None if gender == "N/A" else gender,
            "Department": None if dept == "N/A" else dept,
            "University": None if univ == "N/A" else univ,
            "Academic_Year": None if year == "N/A" else year,
            "waiver_or_scholarship": None if waiver == "N/A" else waiver,
        }])

    st.write("")
    if st.button("🔍 Predict Mental Health Status"):
        try:
            # The pipeline you saved knows its own expected columns thanks to ColumnTransformer.
            # We just feed a DataFrame; transformers will ignore unknowns and impute NaNs.
            pred_raw = model.predict(df_in)[0]

            label = classes_to_label(pred_raw, model, encoder)  # safe conversion
            risk_name, actions = to_risk(target, label)

            st.success(f"✅ Predicted: {label}")
            st.info(f"**Risk Level:** {risk_name}")
            st.write("**Suggested Actions:** " + " • ".join(actions))

            # ---- Log minimal info
            append_log({
                "timestamp": datetime.utcnow().isoformat(),
                "target": target,
                "prediction": label,
                "risk_tier": risk_name,
            })

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.caption("This research prototype is not medical advice. For urgent concerns, contact local crisis services or a qualified professional.")

# -----------------------------
# 6) Dashboard
# -----------------------------
with tab_dash:
    st.subheader("📊 Dashboard Analytics")
    if not os.path.exists(LOG_PATH):
        st.warning("No log file yet. Make a few predictions first.")
        st.stop()

    try:
        log_df = pd.read_csv(LOG_PATH)
    except Exception as e:
        st.error(f"Could not read log file: {e}")
        st.stop()

    if log_df.empty:
        st.warning("Log file is empty.")
        st.stop()

    # Parse dates
    try:
        log_df["date"] = pd.to_datetime(log_df["timestamp"]).dt.date
    except Exception:
        log_df["date"] = datetime.utcnow().date()

    # Top KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total predictions", len(log_df))
    with c2:
        st.metric("Targets covered", log_df["target"].nunique())
    with c3:
        st.metric("Last activity (UTC)", str(log_df["date"].max()))

    st.markdown("### Daily Predictions by Target")
    daily = (log_df.groupby(["date", "target"])
                   .size()
                   .reset_index(name="count"))
    if len(daily) > 0:
        import altair as alt
        chart = alt.Chart(daily).mark_line(point=True).encode(
            x="date:T", y="count:Q", color="target:N", tooltip=["date:T", "target:N", "count:Q"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to plot daily trends yet.")

    st.markdown("### Distribution by Risk Tier")
    risk_counts = log_df["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["risk_tier", "count"]
    if len(risk_counts) > 0:
        import altair as alt
        chart2 = alt.Chart(risk_counts).mark_bar().encode(
            x=alt.X("risk_tier:N", sort="-y"),
            y="count:Q",
            tooltip=["risk_tier:N", "count:Q"]
        )
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No risk data yet.")

