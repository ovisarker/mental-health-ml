# app.py — Mental Health Multiclass Classifier (Real-time Form UI)
# ----------------------------------------------------------------
# Tabs: PREDICT | EXPLAIN | ASSISTANT
# - No CSV uploads. Users fill a form; we build a single-row DataFrame and predict.
# - Uses your saved sklearn/imb pipeline (.joblib) and paired LabelEncoder (.joblib)
# - Works with your Early-Warning models; symptom questions are optional (for user feedback).
# ----------------------------------------------------------------

from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional (Explainability)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# --------------------------- Page & Consent ---------------------------
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Multiclass Classifier — Real-time")
st.caption(
    "Research prototype — not medical advice. Predictions are for educational use only. "
    "If you or someone you know is at risk, contact local emergency/crisis services immediately."
)
with st.expander("Consent & Privacy", expanded=False):
    st.write(
        "- By using this tool you consent to processing your inputs for research-quality analytics.\n"
        "- We do **not** store PII. We only log timestamp, target, predicted class, and risk tier.\n"
        "- This tool does **not** triage crisis situations."
    )

# --------------------------- Defaults ---------------------------
BASE = Path(__file__).parent  # repo folder

DEFAULTS = {
    "Anxiety": {
        "model":  str(BASE / "final_anxiety_model.joblib"),
        "encoder":str(BASE / "final_anxiety_encoder.joblib"),
        "meta":   ""  # optional JSON; leave empty if you didn't save it
    },
    "Stress": {
        "model":  str(BASE / "final_stress_model.joblib"),
        "encoder":str(BASE / "final_stress_encoder.joblib"),
        "meta":   ""
    },
    "Depression": {
        "model":  str(BASE / "final_depression_model.joblib"),
        "encoder":str(BASE / "final_depression_encoder.joblib"),
        "meta":   ""
    },
}

# --------------------------- Intervention plan ---------------------------
RISK_PLAN = {
    "Anxiety": {
        "tiers": {
            "0": ("Low",      ["Keep routine", "Sleep 7–9h", "30m activity"]),
            "1": ("Mild",     ["Breathing 4-7-8", "Journaling 10m"]),
            "2": ("Moderate", ["Peer support", "Counseling signup link"]),
            "3": ("Severe",   ["Contact counselor", "Follow-up within 48h"])
        }
    },
    "Stress": {
        "tiers": {
            "0": ("Low",      ["Pomodoro 25/5", "Walk 15m"]),
            "1": ("Mild",     ["Time-blocking", "Say no to one task"]),
            "2": ("Moderate", ["Advisor meeting", "Brief CBT worksheet"]),
            "3": ("Severe",   ["Escalate to student services", "Wellbeing check-in"])
        }
    },
    "Depression": {
        "tiers": {
            "0": ("Low",      ["Gratitude (3 items)", "Social contact 10m"]),
            "1": ("Mild",     ["Behavioral activation: 1 small task/day"]),
            "2": ("Moderate", ["Counseling referral", "Follow-up in 72h"]),
            "3": ("Severe",   ["Immediate support from services", "Safety plan review"])
        }
    }
}

def _risk_from_text(label_text: str) -> tuple[str, list[str]]:
    """Fallback mapping if model classes are strings (e.g., 'Moderate Anxiety')."""
    lt = label_text.lower()
    if "severe" in lt:   return ("Severe",   RISK_PLAN["Anxiety"]["tiers"]["3"][1])
    if "moderate" in lt: return ("Moderate", RISK_PLAN["Anxiety"]["tiers"]["2"][1])
    if "minimal" in lt:  return ("Mild",     RISK_PLAN["Anxiety"]["tiers"]["1"][1])
    if "mild" in lt:     return ("Mild",     RISK_PLAN["Anxiety"]["tiers"]["1"][1])
    return ("Low",        RISK_PLAN["Anxiety"]["tiers"]["0"][1])

def to_risk(target_name: str, predicted_label) -> tuple[str, list[str]]:
    conf = RISK_PLAN.get(target_name, {})
    tiers = conf.get("tiers", {})
    key = str(predicted_label)
    if key.isdigit() and key in tiers:
        return tiers[key]
    return _risk_from_text(str(predicted_label))

# --------------------------- Cached loaders ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: str):
    pipe = joblib.load(path)
    if "prep" not in pipe.named_steps or "clf" not in pipe.named_steps:
        raise ValueError("Pipeline must contain 'prep' and 'clf' steps.")
    return pipe

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    return joblib.load(path)

def read_metadata(path: str) -> dict:
    if not path or not os.path.exists(path): return {}
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return {}

# --------------------------- Column & logging helpers ---------------------------
def expected_raw_columns(pipe) -> list[str]:
    """Return raw input columns expected before One-Hot."""
    prep = pipe.named_steps["prep"]
    # assumes ('num', ...), ('cat', ...) in ColumnTransformer
    num_cols = list(prep.transformers_[0][2]) if len(prep.transformers_) > 0 else []
    cat_cols = list(prep.transformers_[1][2]) if len(prep.transformers_) > 1 else []
    return list(num_cols) + list(cat_cols)

def safe_inverse_transform(le, y_pred):
    try: return le.inverse_transform(y_pred)
    except Exception: return y_pred

def append_log_row(target: str, label_text: str, risk_tier: str, path="prediction_log.csv"):
    row = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "target": target,
        "prediction": label_text,
        "risk_tier": risk_tier,
    }])
    if os.path.exists(path): row.to_csv(path, mode="a", header=False, index=False)
    else: row.to_csv(path, index=False)

# --------------------------- Sidebar ---------------------------
st.sidebar.header("Configuration")
target = st.sidebar.selectbox("Choose target", ["Anxiety", "Stress", "Depression"], index=0)

model_path  = st.sidebar.text_input("Model pipeline (.joblib)",  DEFAULTS[target]["model"])
enc_path    = st.sidebar.text_input("Label encoder (.joblib)",    DEFAULTS[target]["encoder"])
meta_path   = st.sidebar.text_input("Metadata (.json, optional)", DEFAULTS[target]["meta"])

mode = st.sidebar.radio("Prediction mode", ["Quick Check (Early-Warning)", "Symptom Check (Assessment)"], index=0)

# --------------------------- Tabs ---------------------------
tab_pred, tab_explain, tab_assist = st.tabs(["🔮 Predict", "🔍 Explain", "🤝 Assistant"])

# ============== PREDICT TAB ==============
with tab_pred:
    st.subheader("Answer a few questions to get your result")

    # Load model/encoder once
    try:
        pipe = load_pipeline(model_path)
        le   = load_encoder(enc_path)
        exp_cols = expected_raw_columns(pipe)
    except Exception as e:
        st.error(f"Failed to load model/encoder: {e}")
        st.stop()

    # --- Common form fields (Early-Warning inputs)
    with st.form("realtime_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=15, max_value=80, value=22, step=1)
            cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=4.0, value=3.50, step=0.01, format="%.2f")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
            academic_year = st.selectbox("Academic Year", ["1st","2nd","3rd","4th"], index=2)
        with c2:
            university = st.text_input("University", value="DIU")
            department = st.text_input("Department", value="CSE")
            waiver = st.selectbox("Waiver or Scholarship", ["No","Yes"], index=0)

        # --- Optional symptom questions (for user feedback/assessment)
        gad_vals, pss_vals, dass_vals = {}, {}, {}
        if mode == "Symptom Check (Assessment)":
            st.markdown("**Symptom Questionnaire (optional)**")
            if target == "Anxiety":
                st.caption("GAD-7 style inputs (1–4). Choose what fits best.")
                for i in range(1,8):
                    gad_vals[f"GAD{i}"] = st.select_slider(f"GAD{i}", options=[1,2,3,4], value=2)
            elif target == "Stress":
                st.caption("PSS-10 style inputs (1–4).")
                for i in range(1,11):
                    pss_vals[f"PSS{i}"] = st.select_slider(f"PSS{i}", options=[1,2,3,4], value=2)
            elif target == "Depression":
                st.caption("DASS-D7 style inputs (1–4).")
                for i in range(1,8):
                    dass_vals[f"DASS_D{i}"] = st.select_slider(f"DASS_D{i}", options=[1,2,3,4], value=2)

        submitted = st.form_submit_button("Get Prediction")

    if submitted:
        try:
            # Build a single-row DataFrame **only** with columns the pipeline expects
            raw = {
                "Age": age,
                "Current_CGPA": cgpa,
                "Gender": gender,
                "University": university,
                "Department": department,
                "Academic_Year": academic_year,
                "waiver_or_scholarship": waiver
            }
            # If the model expects symptom items (e.g., replication models), include them if present in exp_cols
            for k, v in {**gad_vals, **pss_vals, **dass_vals}.items():
                if k in exp_cols: raw[k] = v

            # ensure all expected columns exist (fill missing with np.nan)
            row_dict = {c: raw.get(c, np.nan) for c in exp_cols}
            df_in = pd.DataFrame([row_dict])

            # Predict
            pred = pipe.predict(df_in)
            label = safe_inverse_transform(le, pred)[0]
            risk, actions = to_risk(target, label)

            st.success(f"**Prediction:** {label}")
            st.write(f"**Risk tier:** {risk}")
            st.write(f"**Suggested actions:** " + " • ".join(actions))

            # Show a compact table with the inputs and output
            show = df_in.copy()
            show["prediction"] = [label]
            show["risk_tier"] = [risk]
            st.dataframe(show, use_container_width=True)

            # Log (no PII)
            append_log_row(target, str(label), str(risk))

            # Save to session for Explain tab
            st.session_state["last_df_in"] = df_in
            st.session_state["last_label"] = label
            st.session_state["last_pipe"] = pipe

            # Optional: simple symptom score echo (for the user)
            if mode == "Symptom Check (Assessment)":
                if target == "Anxiety" and gad_vals:
                    gad_score = sum(gad_vals.values())
                    st.caption(f"GAD-like total (1–4 per item): **{gad_score}** (7 items)")
                if target == "Stress" and pss_vals:
                    pss_score = sum(pss_vals.values())
                    st.caption(f"PSS-like total (1–4 per item): **{pss_score}** (10 items)")
                if target == "Depression" and dass_vals:
                    dass_score = sum(dass_vals.values())
                    st.caption(f"DASS-D-like total (1–4 per item): **{dass_score}** (7 items)")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============== EXPLAIN TAB ==============
with tab_explain:
    st.subheader("Explain the latest prediction (tree models)")
    if not _HAS_SHAP:
        st.info("Install `shap` in requirements.txt to enable explanations.")
    else:
        if "last_df_in" not in st.session_state or "last_pipe" not in st.session_state:
            st.caption("Make a prediction first in the **Predict** tab.")
        else:
            df_in = st.session_state["last_df_in"]
            pipe  = st.session_state["last_pipe"]
            label = st.session_state.get("last_label", None)

            try:
                clf = pipe.named_steps["clf"]
                name = clf.__class__.__name__
                if not any(k in name.lower() for k in ["forest", "xgb", "lgbm", "catboost", "tree"]):
                    st.warning(f"Explainability is enabled for tree/boosting models. Detected: {name}")
                else:
                    prep = pipe.named_steps["prep"]
                    X_trans = prep.transform(df_in)  # transformed features
                    explainer = shap.Explainer(clf)
                    sv = explainer(X_trans)

                    if label is not None:
                        st.write("Predicted class:", label)

                    st.write("Top feature contributions for your last submission:")
                    shap.plots.bar(sv[0], max_display=12, show=False)
                    st.pyplot(bbox_inches="tight")

            except Exception as e:
                st.error(f"Explainability failed: {e}")

# ============== ASSISTANT TAB ==============
with tab_assist:
    st.subheader("Non-clinical Assistant")
    st.caption("Ask about breathing, sleep, time management, or support options. For emergencies, contact local services.")

    FAQ = {
        "breathing": "Try 4–7–8 breathing: inhale 4s, hold 7s, exhale 8s, for 4 rounds.",
        "sleep": "Aim for 7–9 hours. Keep consistent sleep/wake times; avoid screens 1 hour before bed.",
        "time management": "Use Pomodoro: 25m focus + 5m break. After 4 cycles, take a 20m break.",
        "support": "Consider speaking to your student counseling service or a trusted mentor.",
        "study stress": "Break tasks into chunks. 25-minute focus blocks, 5-minute breaks. Prioritize 1–2 high-impact tasks."
    }

    def faq_bot(query: str) -> str:
        q = (query or "").lower().strip()
        for k, v in FAQ.items():
            if k in q:
                return v
        return ("I can help with breathing, sleep, time management, support options, and study stress. "
                "Try asking about one of those.")

    user_q = st.text_input("Type a question (e.g., 'How to manage stress before exams?')")
    if st.button("Ask"):
        st.write(faq_bot(user_q))

# --------------------------- Footer ---------------------------
st.caption("This tool is not medical advice. For urgent concerns, contact local crisis services or a qualified professional.")
