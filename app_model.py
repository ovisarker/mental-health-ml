################################################################################
# Minimal ML-Powered Mental Health Assessment (Thesis Version)
# - Screening + Dashboard
# - Anxiety, Stress, Depression
# - English + Bangla
# - Uses trained ML models (.joblib) for prediction
#
# Developed by Team Dual Core (© 2025)
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import joblib
from datetime import datetime

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Assessment (ML)",
    page_icon="🧠",
    layout="wide",
)

LOG_PATH = "log.csv"

# ------------------------------------------------------------------------------
# SAFE CSV LOADER
# ------------------------------------------------------------------------------
def load_safe_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # reset corrupted file
        try:
            os.remove(path)
        except Exception:
            pass
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# LANGUAGE
# ------------------------------------------------------------------------------
LANG = st.sidebar.selectbox("Language", ["English", "বাংলা (Bangla)"])

TEXT = {
    "English": {
        "title": "AI-based Mental Health Assessment (ML)",
        "screen": "🧩 Screening",
        "dash": "📊 Dashboard",
        "choose_target": "Select assessment",
        "screening_form": "Screening Form",
        "instructions": "Rate each statement from 1 (lowest) to 5 (highest).",
        "scale": "Scale Meaning",
        "predict": "🔍 Predict Mental Health Status",
        "risk_level": "Risk Level",
        "suggested": "Suggested Actions (not a diagnosis)",
        "no_logs": "No screening records found.",
        "dash_title": "Analytics Dashboard",
        "dash_recent": "Recent Results",
        "dash_risk": "Risk Distribution",
        "dash_trend": "Trend Over Time",
        "model_missing": "Required model file not found for this assessment.",
    },
    "বাংলা (Bangla)": {
        "title": "এআই ভিত্তিক মানসিক স্বাস্থ্য মূল্যায়ন (এমএল)",
        "screen": "🧩 স্ক্রিনিং",
        "dash": "📊 ড্যাশবোর্ড",
        "choose_target": "মূল্যায়ন নির্বাচন করুন",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "প্রতিটি প্রশ্নের জন্য ১ (সর্বনিম্ন) থেকে ৫ (সর্বোচ্চ) নির্বাচন করুন।",
        "scale": "স্কেল মানে",
        "predict": "🔍 মানসিক স্বাস্থ্যের ফলাফল দেখুন",
        "risk_level": "ঝুঁকির স্তর",
        "suggested": "প্রস্তাবিত পদক্ষেপ (ডায়াগনোসিস নয়)",
        "no_logs": "কোনও স্ক্রিনিং ডেটা পাওয়া যায়নি।",
        "dash_title": "অ্যানালিটিক্স ড্যাশবোর্ড",
        "dash_recent": "সাম্প্রতিক ফলাফল",
        "dash_risk": "ঝুঁকির বণ্টন",
        "dash_trend": "সময়ের সাথে স্ক্রিনিং প্রবণতা",
        "model_missing": "এই মূল্যায়নের জন্য প্রয়োজনীয় মডেল ফাইল পাওয়া যায়নি।",
    },
}[LANG]

# ------------------------------------------------------------------------------
# QUESTIONS (EN + BN)
# ------------------------------------------------------------------------------
QUESTIONS_EN = {
    "Anxiety": [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it is hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen",
    ],
    "Stress": [
        "Upset because of unexpected events",
        "Unable to control important things in life",
        "Felt nervous and stressed",
        "Confident about handling problems",
        "Things going your way",
        "Could not cope with all things you had to do",
        "Able to control irritations in your life",
        "Felt on top of things",
        "Angry because things were out of control",
        "Felt difficulties piling up too high",
    ],
    "Depression": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble sleeping or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or like a failure",
        "Trouble concentrating on things",
        "Moving / speaking slowly or restlessness",
        "Thoughts of self-harm or death",
    ],
}

QUESTIONS_BN = {
    "Anxiety": [
        "নার্ভাস, উদ্বিগ্ন বা অস্থির অনুভব করা",
        "দুশ্চিন্তা থামাতে বা নিয়ন্ত্রণ করতে না পারা",
        "বিভিন্ন বিষয় নিয়ে অতিরিক্ত দুশ্চিন্তা করা",
        "মনকে শান্ত করতে কষ্ট হওয়া",
        "এতটাই অস্থির যে বসে থাকতে কষ্ট হয়",
        "সহজেই বিরক্ত বা রাগান্বিত হয়ে যাওয়া",
        "মনে হওয়া যেন কিছু খারাপ ঘটতে যাচ্ছে",
    ],
    "Stress": [
        "অপ্রত্যাশিত ঘটনার কারণে খুব কষ্ট পাওয়া",
        "গুরুত্বপূর্ণ বিষয়গুলো নিয়ন্ত্রণ করতে না পারার অনুভূতি",
        "নার্ভাস ও চাপগ্রস্ত অনুভব করা",
        "সমস্যা সামলাতে আত্মবিশ্বাসী হওয়া",
        "সব কিছু ইচ্ছেমতো হওয়া",
        "সব কাজ সামলাতে না পারার অনুভূতি",
        "বিরক্তিকর বিষয়গুলো নিয়ন্ত্রণ করতে পারা",
        "অনুভব করা যে আপনি সব কিছুর উপরে আছেন",
        "বিষয়গুলো নিয়ন্ত্রণের বাইরে চলে গেলে রাগ হওয়া",
        "অনুভব করা যে সমস্যাগুলো খুব দ্রুত জমে যাচ্ছে",
    ],
    "Depression": [
        "কাজকর্মে আগ্রহ বা আনন্দ কমে যাওয়া",
        "মন খারাপ, বিষণ্ন বা আশাহীন লাগা",
        "ঘুমের সমস্যা বা অতিরিক্ত ঘুমানো",
        "অল্পতেই ক্লান্ত বা শক্তিহীন লাগা",
        "খাবারের আগ্রহ কমে যাওয়া বা বেশি খাওয়া",
        "নিজেকে ব্যর্থ বা খুব খারাপ মনে হওয়া",
        "কোনো কাজে মনোযোগ দিতে কষ্ট হওয়া",
        "ধীরে চলাফেরা/কথা বলা বা অস্থিরতা",
        "নিজেকে আঘাত করা বা মৃত্যুর চিন্তা",
    ],
}

# ------------------------------------------------------------------------------
# SCALE MEANING (1–5)
# ------------------------------------------------------------------------------
SCALE_EN = {
    "Anxiety": [
        "Not at all",
        "Several days",
        "Half the days",
        "Nearly every day",
        "Almost always",
    ],
    "Depression": [
        "Not at all",
        "Several days",
        "Half the days",
        "Nearly every day",
        "Almost always",
    ],
    "Stress": [
        "Never",
        "Almost never",
        "Sometimes",
        "Fairly often",
        "Very often",
    ],
}

SCALE_BN = {
    "Anxiety": [
        "একদমই না",
        "কিছুদিন",
        "অর্ধেক দিন",
        "প্রায় প্রতিদিন",
        "প্রায় সব সময়",
    ],
    "Depression": [
        "একদমই না",
        "কিছুদিন",
        "অর্ধেক দিন",
        "প্রায় প্রতিদিন",
        "প্রায় সব সময়",
    ],
    "Stress": [
        "কখনোই না",
        "খুব কম",
        "মাঝে মাঝে",
        "প্রায়ই",
        "প্রায় সব সময়",
    ],
}

# ------------------------------------------------------------------------------
# MODEL LOADING (YOUR FILENAMES)
# ------------------------------------------------------------------------------
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

@st.cache_resource
def load_models():
    models = {}
    encoders = {}
    for target, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[target] = joblib.load(path)
        else:
            models[target] = None

        enc_path = ENCODER_FILES.get(target)
        if enc_path and os.path.exists(enc_path):
            encoders[target] = joblib.load(enc_path)
        else:
            encoders[target] = None
    return models, encoders

MODELS, ENCODERS = load_models()

# ------------------------------------------------------------------------------
# ML PREDICTION
# ------------------------------------------------------------------------------
def ml_predict(values, target):
    """
    Run ML model for given target.
    values: list of 1–5 slider scores.
    Returns: (label_str, risk_str, raw_pred, confidence_or_None)
    """
    model = MODELS.get(target)
    encoder = ENCODERS.get(target)

    if model is None:
        raise RuntimeError("Model file not found or failed to load.")

    # Use feature_names_in_ if available for correct column names
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and len(feature_names) == len(values):
        X = pd.DataFrame([values], columns=feature_names)
    else:
        X = pd.DataFrame([values])

    pred = model.predict(X)[0]

    # Decode label if encoder exists
    if encoder is not None:
        try:
            label = encoder.inverse_transform([pred])[0]
        except Exception:
            label = str(pred)
    else:
        label = str(pred)

    # Optional probability / confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            confidence = float(np.max(proba))
        except Exception:
            confidence = None

    # Risk tier mapping from label text
    label_lower = label.lower()
    if "minimal" in label_lower:
        risk = "Low"
    elif "mild" in label_lower:
        risk = "Moderate"
    elif "moderate" in label_lower:
        risk = "High"
    elif "severe" in label_lower:
        risk = "Critical"
    else:
        risk = "Unknown"

    return label, risk, pred, confidence

# ------------------------------------------------------------------------------
# PROFESSIONAL SUGGESTED ACTIONS
# ------------------------------------------------------------------------------
def professional_suggestions(target: str, risk: str) -> str:
    if risk == "Low":
        return (
            "Current symptoms appear in a lower range. Maintaining regular sleep, balanced nutrition, "
            "physical activity and supportive social contact is recommended. Monitoring mood and stress "
            "over time can help detect changes early."
        )
    if risk == "Moderate":
        return (
            "Symptoms are clinically relevant and may intermittently affect concentration, energy or motivation. "
            "Structured daily routines, stress-management strategies (for example, brief relaxation or breathing "
            "exercises) and talking with trusted people or a counselor can be helpful. If difficulties persist for "
            "several weeks, a professional mental health assessment is advisable."
        )
    if risk == "High":
        return (
            "Symptoms are in a higher range and likely impact day-to-day functioning. Reducing avoidable overload, "
            "seeking support from a qualified counselor, psychologist or physician and discussing work/study "
            "adjustments would be clinically appropriate. Early intervention can prevent further deterioration."
        )
    if risk == "Critical":
        return (
            "Symptoms are severe and may significantly interfere with safety, functioning or quality of life. "
            "A prompt consultation with a mental health professional or physician is strongly recommended. "
            "If there are thoughts of self-harm or you feel unable to stay safe, emergency services or crisis "
            "hotlines should be contacted immediately."
        )
    return (
        "Symptom level could not be clearly categorized. If you are unsure about your mental health or "
        "distress is affecting your daily life, consider discussing this result with a mental health professional."
    )

# ------------------------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------------------------
page = st.sidebar.radio("Navigation", [TEXT["screen"], TEXT["dash"]])

# ------------------------------------------------------------------------------
# PAGE: SCREENING
# ------------------------------------------------------------------------------
if page == TEXT["screen"]:
    st.title(TEXT["title"])

    target = st.selectbox(TEXT["choose_target"], ["Anxiety", "Stress", "Depression"])

    st.subheader(f"{target} — {TEXT['screening_form']}")
    st.write(TEXT["instructions"])

    col_q, col_scale = st.columns([3, 1.4])

    # Scale box
    with col_scale:
        st.markdown(f"**{TEXT['scale']} (1–5)**")
        scale_labels = (SCALE_EN if LANG == "English" else SCALE_BN)[target]
        for i, label in enumerate(scale_labels, start=1):
            st.write(f"{i} — {label}")

    # Questions
    responses = []
    with col_q:
        qs = QUESTIONS_EN[target] if LANG == "English" else QUESTIONS_BN[target]
        for i, q_text in enumerate(qs):
            st.write(f"**Q{i+1}. {q_text}**")
            responses.append(
                st.slider(
                    f"Q{i+1}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    label_visibility="collapsed",
                )
            )

    # Predict using ML model
    if st.button(TEXT["predict"]):
        if MODELS.get(target) is None:
            st.error(TEXT["model_missing"])
        else:
            try:
                label, risk, raw_pred, confidence = ml_predict(responses, target)

                st.success(f"🎯 Predicted: {label}")
                st.info(f"🩺 {TEXT['risk_level']}: **{risk}**")

                if confidence is not None:
                    st.write(f"Model confidence (approx.): **{confidence:.2f}**")

                st.write("### " + TEXT["suggested"])
                st.write(professional_suggestions(target, risk))

                # Save to log
                df = load_safe_csv(LOG_PATH)
                new_row = pd.DataFrame(
                    [
                        {
                            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "target": target,
                            "predicted_label": label,
                            "risk": risk,
                            "raw_prediction": raw_pred,
                        }
                    ]
                )
                if df.empty:
                    new_row.to_csv(LOG_PATH, index=False)
                else:
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(LOG_PATH, index=False)
                st.success("Result stored in local history.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------------------------
elif page == TEXT["dash"]:
    st.title(TEXT["dash_title"])

    df = load_safe_csv(LOG_PATH)
    if df.empty:
        st.warning(TEXT["no_logs"])
    else:
        st.subheader(TEXT["dash_recent"])
        st.dataframe(df.tail(20), use_container_width=True)

        # Risk distribution
        if "risk" in df.columns:
            st.subheader(TEXT["dash_risk"])
            risk_counts = df["risk"].value_counts().reset_index()
            risk_counts.columns = ["risk", "count"]
            chart = alt.Chart(risk_counts).mark_bar().encode(
                x="risk:N", y="count:Q", color="risk:N"
            )
            st.altair_chart(chart, use_container_width=True)

        # Trend: number of screenings per day
        if "datetime" in df.columns:
            st.subheader(TEXT["dash_trend"])
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            trend = (
                df.dropna(subset=["datetime"])
                .groupby(df["datetime"].dt.date)
                .size()
                .reset_index(name="screenings")
            )
            if not trend.empty:
                chart = alt.Chart(trend).mark_line(point=True).encode(
                    x="datetime:T", y="screenings:Q"
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not enough valid dates to show a trend.")

        # Download
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "mh_log_ml.csv",
            "text/csv",
        )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    """
<div style='text-align:center;margin-top:40px;opacity:0.7;'>
AI Mental Health Assessment System (ML)<br>
Developed by <b>Team Dual Core</b><br>
© 2025 All Rights Reserved
</div>
""",
    unsafe_allow_html=True,
)
