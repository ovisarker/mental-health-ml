# app.py — Mental Health Multiclass Classifier (Final)
# ----------------------------------------------------
# Tabs: PREDICT | EXPLAIN | ASSISTANT
# Pipelines: sklearn/imb pipelines saved as .joblib with steps ['prep','sampler','clf']
# LabelEncoder paired as a .joblib
# ----------------------------------------------------

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

st.title("🧠 Mental Health Multiclass Classifier")
st.caption(
    "Research prototype — not medical advice. Predictions are for educational use only. "
    "If you or someone you know is at risk, contact local emergency/crisis services immediately."
)
with st.expander("Consent & Privacy", expanded=False):
    st.write(
        "- By using this tool you consent to processing your uploaded inputs for research-quality analytics.\n"
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
# Map predicted label -> risk tier + suggested actions
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
    """Fallback mapping if labels are strings like 'Mild Anxiety'."""
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
    # label is text like 'Moderate Anxiety' — map by keywords
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

# --------------------------- Column helpers ---------------------------
def expected_raw_columns(pipe) -> list[str]:
    """Return raw input columns expected before One-Hot."""
    prep = pipe.named_steps["prep"]
    # assumes ('num', ...), ('cat', ...) in ColumnTransformer
    num_cols = list(prep.transformers_[0][2]) if len(prep.transformers_) > 0 else []
    cat_cols = list(prep.transformers_[1][2]) if len(prep.transformers_) > 1 else []
    return list(num_cols) + list(cat_cols)

def check_missing_or_extra(upload_cols: list[str], expected_cols: list[str]):
    upload_set, expected_set = set(upload_cols), set(expected_cols)
    missing = list(expected_set - upload_set)
    extra   = list(upload_set - expected_set)
    return missing, extra

def safe_inverse_transform(le, y_pred):
    try: return le.inverse_transform(y_pred)
    except Exception: return y_pred

def append_log(df_pred: pd.DataFrame, target: str, path="prediction_log.csv"):
    cols = ["timestamp","target","prediction","risk_tier"]
    log_df = pd.DataFrame({
        "timestamp": [datetime.utcnow().isoformat()] * len(df_pred),
        "target": [target] * len(df_pred),
        "prediction": df_pred["prediction"],
        "risk_tier": df_pred["risk_tier"],
    })[cols]
    if os.path.exists(path):
        log_df.to_csv(path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(path, index=False)

# --------------------------- Sidebar ---------------------------
st.sidebar.header("Configuration")

target = st.sidebar.selectbox("Choose target", ["Anxiety", "Stress", "Depression"], index=0)

model_path  = st.sidebar.text_input("Model pipeline (.joblib)",  DEFAULTS[target]["model"])
enc_path    = st.sidebar.text_input("Label encoder (.joblib)",    DEFAULTS[target]["encoder"])
meta_path   = st.sidebar.text_input("Metadata (.json, optional)", DEFAULTS[target]["meta"])

show_template = st.sidebar.button("Show expected columns + download CSV template")

# Load on demand for template or prediction
pipe, le, meta = None, None, {}

if show_template:
    try:
        pipe = load_pipeline(model_path)
        cols = expected_raw_columns(pipe)
        st.info("Expected raw columns (CSV must contain these):")
        st.code(", ".join(cols), language="text")
        template = pd.DataFrame([{c: "" for c in cols}])
        st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                           file_name=f"template_{target.lower()}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Cannot inspect columns: {e}")

# --------------------------- Tabs ---------------------------
tab_pred, tab_explain, tab_assist = st.tabs(["🔮 Predict", "🔍 Explain", "🤝 Assistant"])

# ============== PREDICT TAB ==============
with tab_pred:
    st.subheader("Upload data and generate predictions")

    uploaded = st.file_uploader("Upload CSV with the expected raw columns", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("**Preview**")
            st.dataframe(df_in.head(20), use_container_width=True)

            if pipe is None:
                pipe = load_pipeline(model_path)
            if le is None:
                le = load_encoder(enc_path)
            if not meta:
                meta = read_metadata(meta_path)

            exp_cols = expected_raw_columns(pipe)
            missing, extra = check_missing_or_extra(df_in.columns.tolist(), exp_cols)
            if missing:
                st.error(f"Missing required columns ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}")
                st.stop()
            if extra:
                st.info(f"Upload has extra columns ({len(extra)}). They’ll be ignored by the pipeline.")

            preds = pipe.predict(df_in)
            labels = safe_inverse_transform(le, preds)

            out = df_in.copy()
            out["prediction"] = labels

            # Interventions
            risk_names, action_lists = [], []
            for lab in labels:
                risk, acts = to_risk(target, lab)
                risk_names.append(risk)
                action_lists.append(" • ".join(acts))
            out["risk_tier"] = risk_names
            out["suggested_actions"] = action_lists

            # Display
            st.success("✅ Predictions ready")
            st.dataframe(out[["prediction","risk_tier","suggested_actions"]].head(20),
                        use_container_width=True)

            # Log (no PII)
            append_log(out, target)

            # Download
            st.download_button("📥 Download predictions CSV",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name=f"predictions_{target.lower()}.csv",
                               mime="text/csv")

            # Save last input/preds to session for Explain tab
            st.session_state["df_in"] = df_in
            st.session_state["labels"] = labels
            st.session_state["pipe"] = pipe

            if meta:
                st.caption(f"Model: {meta.get('model','?')} | Classes: {meta.get('classes','?')}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============== EXPLAIN TAB ==============
with tab_explain:
    st.subheader("Explain a single prediction (tree models)")
    if not _HAS_SHAP:
        st.info("Install `shap` in requirements.txt to enable explanations.")
    else:
        if "df_in" not in st.session_state or "pipe" not in st.session_state:
            st.caption("Run a prediction first in the **Predict** tab.")
        else:
            df_in = st.session_state["df_in"]
            pipe  = st.session_state["pipe"]
            labels = st.session_state.get("labels", None)

            row_idx = st.number_input("Row index to explain", min_value=0, max_value=len(df_in)-1, value=0, step=1)
            if st.button("Explain this row"):
                try:
                    # Only reasonable for tree models (RF/LGBM/XGB/CatBoost)
                    clf = pipe.named_steps["clf"]
                    name = clf.__class__.__name__
                    if not any(k in name.lower() for k in ["forest", "xgb", "lgbm", "catboost", "tree"]):
                        st.warning(f"Explainability is only enabled for tree/boosting models. Detected: {name}")
                    else:
                        prep = pipe.named_steps["prep"]
                        X_trans = prep.transform(df_in)  # transformed features
                        # Build explainer directly on tree model
                        explainer = shap.Explainer(clf)
                        sv = explainer(X_trans[row_idx:row_idx+1])

                        if labels is not None:
                            st.write("Predicted class:", labels[row_idx])

                        st.write("Top feature contributions:")
                        # Plot bar (max 12)
                        shap.plots.bar(sv[0], max_display=12, show=False)
                        st.pyplot(bbox_inches="tight")
                except Exception as e:
                    st.error(f"Explainability failed: {e}")

# ============== ASSISTANT TAB ==============
with tab_assist:
    st.subheader("Non-clinical Assistant")
    st.caption("Ask about breathing, sleep, time management, or support options.")

    FAQ = {
        "breathing": "Try 4–7–8 breathing: inhale 4s, hold 7s, exhale 8s, for 4 rounds.",
        "sleep": "Aim for 7–9 hours. Keep consistent sleep/wake times; avoid screens 1 hour before bed.",
        "time management": "Use Pomodoro: 25m focus + 5m break. After 4 cycles, take a 20m break.",
        "support": "Consider speaking to your student counseling service or a trusted mentor.",
    }

    def faq_bot(query: str) -> str:
        q = (query or "").lower().strip()
        for k, v in FAQ.items():
            if k in q:
                return v
        return ("I can help with breathing, sleep, time management, and support options. "
                "Try asking about one of those.")

    user_q = st.text_input("Type a question (e.g., 'How to manage stress before exams?')")
    if st.button("Ask"):
        st.write(faq_bot(user_q))

# --------------------------- Footer ---------------------------
st.caption("This tool is not medical advice. For urgent concerns, contact local crisis services or a qualified professional.")
