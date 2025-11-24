import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
from datetime import datetime

# -----------------------------------------------------------------------------
# ⚙️ Page configuration and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI‑based Mental Health Detection System",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
<style>
body {background-color:#0E1117;color:#FAFAFA;}
h1,h2,h3,h4,h5{color:#E0E0E0;}
.stButton>button{background:#111;color:white;font-weight:600;border-radius:10px;}
.stButton>button:hover{background:#2E8B57;color:white;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 🧠 Utility functions
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model(target: str):
    """Attempt to load a pre‑trained model and its encoder.

    Parameters
    ----------
    target : str
        One of "Anxiety", "Stress" or "Depression". Determines which model
        and encoder to load.

    Returns
    -------
    tuple
        A tuple of (model, encoder). Either may be None if loading fails.
    """
    models = {
        "Anxiety": "best_model_Anxiety_Label_Logistic_Regression.joblib",
        "Stress": "best_model_Stress_Label_Logistic_Regression.joblib",
        "Depression": "best_model_Depression_Label_CatBoost.joblib",
    }
    encoders = {
        "Anxiety": "final_anxiety_encoder.joblib",
        "Stress": "final_stress_encoder.joblib",
        "Depression": "final_depression_encoder.joblib",
    }

    m_path = models.get(target)
    e_path = encoders.get(target)

    model = None
    encoder = None
    # Load model if file exists
    if m_path and os.path.exists(m_path):
        try:
            model = joblib.load(m_path)
        except Exception:
            # If loading fails, leave model as None
            model = None

    # Load encoder if file exists and non‑empty
    if e_path and os.path.exists(e_path):
        try:
            enc = joblib.load(e_path)
            if hasattr(enc, "classes_") and len(enc.classes_) > 0:
                encoder = enc
        except Exception:
            encoder = None

    return model, encoder


def numeric_to_label(value: float, target: str) -> str:
    """Map a numeric score to a descriptive label when no encoder is available.

    A simple modulus mapping is used for model predictions. For fallback
    heuristics based on averaged questionnaire scores, the caller should
    implement its own logic.

    Parameters
    ----------
    value : float
        The numeric prediction value (e.g. integer class). When using this
        function as a fallback, the value will be coerced to int.
    target : str
        Either "Anxiety", "Stress", or "Depression" to determine which
        categories to return.

    Returns
    -------
    str
        A descriptive severity label (e.g. "Mild Stress").
    """
    idx = int(value) % 4
    if target == "Anxiety":
        return ["Minimal Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"][idx]
    if target == "Stress":
        return ["Minimal Stress", "Mild Stress", "Moderate Stress", "Severe Stress"][idx]
    # Depression by default
    return [
        "Minimal Depression",
        "Mild Depression",
        "Moderate Depression",
        "Severe Depression",
    ][idx]


def fallback_label_from_responses(responses: list[float], target: str) -> str:
    """Compute a severity label purely from the survey responses.

    This heuristic uses the mean of the provided responses to assign a label
    without relying on any machine‑learning model. The thresholds were chosen
    heuristically: averages below 1.5 indicate minimal severity, averages
    between 1.5 and 2.5 indicate mild severity, between 2.5 and 3.5 moderate,
    and above 3.5 severe.

    Parameters
    ----------
    responses : list of float
        The numerical slider responses from the questionnaire.
    target : str
        The target condition (Anxiety, Stress or Depression).

    Returns
    -------
    str
        A descriptive severity label based on the average response.
    """
    avg = float(np.mean(responses))
    if avg < 1.5:
        level = "Minimal"
    elif avg < 2.5:
        level = "Mild"
    elif avg < 3.5:
        level = "Moderate"
    else:
        level = "Severe"

    return f"{level} {target}"


def risk_tier_map(label: str) -> str:
    """Map severity label keywords to a risk tier.

    Parameters
    ----------
    label : str
        A descriptive severity label containing one of the keywords
        "Minimal", "Mild", "Moderate", or "Severe" (case insensitive).

    Returns
    -------
    str
        A risk tier of "Low", "Moderate", "High" or "Critical". If none of
        the expected keywords are found, "Unknown" is returned.
    """
    mapping = {
        "minimal": "Low",
        "mild": "Moderate",
        "moderate": "High",
        "severe": "Critical",
    }
    lowered = label.lower()
    for key, tier in mapping.items():
        if key in lowered:
            return tier
    return "Unknown"


def save_prediction_log(row: dict) -> None:
    """Append a prediction record to a CSV log file.

    Parameters
    ----------
    row : dict
        A dictionary containing the fields "datetime", "target",
        "predicted_label" and "risk_tier".
    """
    df = pd.DataFrame([row])
    log_path = "prediction_log.csv"
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure the input DataFrame has all expected features for the model.

    If the model exposes a `feature_names_in_` attribute (common for
    scikit‑learn estimators), this function will add any missing columns to
    the input DataFrame, filling them with zeros, and reorder all columns
    accordingly. This prevents "X has 0 features" errors when the trained
    pipeline expects additional features beyond the survey responses.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the survey responses as numeric columns.
    model : any
        A trained scikit‑learn model (or pipeline) exposing the
        `feature_names_in_` attribute. If the model has no such attribute, the
        input DataFrame is returned unchanged.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same number of rows as the input and columns
        matching the expected feature order of the model.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        return df[expected]
    return df


def safe_predict(model, encoder, responses: list[float], target: str) -> str:
    """Perform a model prediction with fallbacks.

    This helper attempts to predict a severity label using the provided model.
    It handles common issues such as missing features, unseen labels and
    runtime exceptions by falling back to a heuristic based on the responses.

    Parameters
    ----------
    model : any or None
        A trained classifier. If None, the fallback is used immediately.
    encoder : any or None
        A label encoder for converting numeric predictions to strings. If
        provided, it is used to decode model outputs. Otherwise the numeric
        label is mapped via `numeric_to_label()`.
    responses : list of float
        The raw survey responses.
    target : str
        The target condition (Anxiety, Stress or Depression).

    Returns
    -------
    str
        The predicted descriptive label (e.g. "Mild Stress").
    """
    # If no model is loaded, immediately use fallback
    if model is None:
        return fallback_label_from_responses(responses, target)

    # Prepare a DataFrame from responses
    df = pd.DataFrame([responses])
    # Attempt to align features if the model expects more features
    try:
        df_aligned = align_features(df, model)
        # Attempt prediction
        pred_val = model.predict(df_aligned)[0]
        # Decode using the encoder if available
        if encoder is not None:
            try:
                return encoder.inverse_transform([pred_val])[0]
            except Exception:
                # If decoding fails due to unseen label, use numeric fallback
                return numeric_to_label(pred_val, target)
        # No encoder: map numeric prediction directly
        return numeric_to_label(pred_val, target)
    except Exception:
        # For any error, fall back to heuristic label
        return fallback_label_from_responses(responses, target)


# -----------------------------------------------------------------------------
# 🧭 Sidebar and navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a page", ["🧩 Prediction", "📊 Dashboard"])


# -----------------------------------------------------------------------------
# 🧩 Prediction Page
# -----------------------------------------------------------------------------
if page == "🧩 Prediction":
    st.title("🧠 AI‑based Mental Health Detection & Support System")
    st.caption("Developed for Thesis & Real‑world Use | 2025")

    # Choose condition to predict
    target = st.selectbox("Select what you want to predict", ["Anxiety", "Stress", "Depression"])
    model, encoder = load_model(target)

    if model is None:
        st.warning(
            f"⚠️ Pre‑trained model for {target} could not be loaded. "
            "Predictions will use a simple scoring heuristic instead."
        )
    else:
        st.success(f"✅ {target} model loaded successfully!")

    # Display form header
    st.markdown(f"### 🧾 {target} Screening Form")
    st.info("Rate each statement from 1 (Not at all) to 5 (Nearly every day).")

    # Define question sets
    question_sets = {
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
            "Could not cope with all the things you had to do",
            "Able to control irritations in your life",
            "Felt on top of things",
            "Angry because things were out of control",
            "Felt difficulties piling up too high",
        ],
        "Depression": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating on things",
            "Moving/speaking slowly or restlessness",
            "Thoughts of self‑harm or death",
        ],
    }

    # Render sliders for each question
    responses = []
    for idx, question in enumerate(question_sets[target]):
        # Provide a unique key for each slider to avoid reusing widgets on rerun
        slider_key = f"{target}_q{idx}"
        val = st.slider(
            label=question,
            min_value=1,
            max_value=5,
            value=3,
            key=slider_key,
        )
        responses.append(val)

    # Perform prediction when button is clicked
    if st.button("🔍 Predict Mental Health Status"):
        predicted_label = safe_predict(model, encoder, responses, target)
        risk = risk_tier_map(predicted_label)
        st.success(f"🎯 Predicted: **{predicted_label}**")
        st.info(f"🩺 Risk Level: **{risk}**")

        # Suggested actions based on risk
        actions = {
            "Low": "Maintain a healthy routine • Sleep 7–9h • Practice daily relaxation",
            "Moderate": "Engage in regular exercise • Keep a journal • Follow a balanced diet",
            "High": "Seek counseling • Reduce workload • Practice mindfulness",
            "Critical": "Consult a mental health professional immediately • Rely on your support network",
        }.get(risk, "Monitor your mental health regularly.")
        st.markdown(f"**Suggested Actions:** {actions}")

        # Save the prediction to the log
        save_prediction_log(
            {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": target,
                "predicted_label": predicted_label,
                "risk_tier": risk,
            }
        )


# -----------------------------------------------------------------------------
# 📊 Dashboard Page
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Mental Health Analytics Dashboard")
    log_path = "prediction_log.csv"

    if not os.path.exists(log_path):
        st.warning("No predictions have been made yet. Please perform a prediction first.")
    else:
        df = pd.read_csv(log_path)
        st.dataframe(df.tail(10), use_container_width=True)

        st.subheader("📈 Risk Distribution Overview")
        tiers = df["risk_tier"].value_counts(normalize=True).mul(100)
        for tier, color in zip(
            ["Low", "Moderate", "High", "Critical"],
            ["#00FF88", "#FFFF00", "#FFA500", "#FF4444"],
        ):
            val = float(tiers.get(tier, 0))
            st.markdown(
                f"<div style='color:{color};font-weight:600'>{tier}: {val:.1f}%</div>",
                unsafe_allow_html=True,
            )
            st.progress(int(val))

        # Time‑series trend
        df["datetime"] = pd.to_datetime(df["datetime"])
        trend = df.groupby(df["datetime"].dt.date).size().reset_index(name="Predictions")
        st.altair_chart(
            alt.Chart(trend).mark_line(point=True, color="#00FFAA").encode(
                x="datetime:T",
                y="Predictions:Q",
            ),
            use_container_width=True,
        )

        # Risk tier distribution
        dist = df["risk_tier"].value_counts().reset_index()
        dist.columns = ["Risk Tier", "Count"]
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x=alt.X("Risk Tier:N", sort="-y"),
                y="Count:Q",
                color="Risk Tier:N",
            ),
            use_container_width=True,
        )

        # Download log file
        st.download_button(
            "⬇️ Download Prediction Log",
            data=df.to_csv(index=False),
            file_name="prediction_log.csv",
            mime="text/csv",
        )
