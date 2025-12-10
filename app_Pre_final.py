"""
app_final.py
=================

This Streamlit application provides a more comprehensive mental‐health assessment and
prediction tool for university students.  Unlike earlier iterations that simply
reported raw questionnaire scores, this version integrates trained machine
learning models to determine which mental‐health concern (Anxiety, Stress or
Depression) is most prominent for an individual.  It also estimates the
severity of each condition based on established clinical cut–off scores and
offers tailored self‑care suggestions.  The app collects demographic and
lifestyle information, though these features are not currently used by the
models, and logs user sessions for research purposes.  The models were trained
offline and saved as joblib files.

To run the app locally use:

    streamlit run app_final.py

"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent

MODELS: Dict[str, str] = {
    "Anxiety": "best_model_Anxiety_Label_Logistic_Regression.joblib",
    "Stress": "best_model_Stress_Label_Logistic_Regression.joblib",
    "Depression": "best_model_Depression_Label_CatBoost.joblib",
}

ENCODERS: Dict[str, str] = {
    "Anxiety": "final_anxiety_encoder.joblib",
    "Stress": "final_stress_encoder.joblib",
    "Depression": "final_depression_encoder.joblib",
}

# Questionnaire definitions for each mental health issue.  These questions
# correspond to the GAD-7 (Anxiety), PSS (Stress) and PHQ-9 (Depression)
QUESTIONNAIRES: Dict[str, List[str]] = {
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
        "In the last month, how often have you been upset because of something unexpected?",
        "In the last month, how often have you felt that you were unable to control the important things in your life?",
        "In the last month, how often have you felt nervous and stressed?",
        "In the last month, how often have you felt confident about your ability to handle your personal problems?",
        "In the last month, how often have you felt that things were going your way?",
        "In the last month, how often have you found that you could not cope with all the things that you had to do?",
        "In the last month, how often have you been able to control irritations in your life?",
    ],
    "Depression": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
    ],
}

SCORE_MEANINGS = {
    1: "Not at all",
    2: "Several days",
    3: "Half the days",
    4: "Nearly every day",
    5: "Almost always",
}

# Severity thresholds for each scale
SEVERITY_THRESHOLDS = {
    "Anxiety": [(0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"), (15, 21, "Severe")],
    "Stress": [(0, 13, "Low"), (14, 26, "Moderate"), (27, 40, "High")],
    "Depression": [(0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"), (15, 19, "Moderately severe"), (20, 27, "Severe")],
}

# Suggestions for each mental health issue and severity level
SUGGESTIONS = {
    "Anxiety": {
        "Minimal": "You appear to have minimal anxiety symptoms. Maintaining a balanced lifestyle with regular exercise, adequate sleep and mindfulness practices can help keep anxiety at bay.",
        "Mild": "Mild anxiety can often be managed with relaxation techniques such as deep breathing, meditation or yoga. Consider speaking to a counselor if symptoms persist.",
        "Moderate": "Moderate anxiety may benefit from cognitive behavioral techniques or counseling. Discuss your feelings with a trusted friend or mental health professional.",
        "Severe": "Severe anxiety warrants professional intervention. Seek support from a licensed therapist or psychiatrist to discuss treatment options including therapy or medication.",
    },
    "Stress": {
        "Low": "You appear to be managing stress well. Continue using healthy coping strategies such as regular exercise, time management and social support.",
        "Moderate": "Moderate stress can be reduced through stress‑management techniques like progressive muscle relaxation, prioritizing tasks and practicing self‑care.",
        "High": "High stress levels can impact physical and mental well‑being. Consider reaching out to counseling services, practicing mindfulness and re‑evaluating your workload to reduce stress.",
    },
    "Depression": {
        "Minimal": "Your depression score is minimal. Maintain a healthy routine and stay connected with friends and activities you enjoy.",
        "Mild": "Mild depression can be alleviated through exercise, talking with supportive individuals and engaging in pleasant activities. Monitor your mood and seek help if symptoms worsen.",
        "Moderate": "Moderate depression may require structured therapy such as cognitive behavioral therapy. Reach out to mental health services or your healthcare provider for guidance.",
        "Moderately severe": "Your symptoms suggest moderately severe depression. It is important to speak with a mental health professional who can offer therapy and discuss treatment options.",
        "Severe": "Severe depression requires prompt professional attention. Please consult a psychiatrist or mental health specialist to explore therapy and possible medications. If you feel unsafe, contact crisis support immediately.",
    },
}


def load_models() -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load ML models and corresponding encoders from disk.

    Returns:
        A tuple (models, encoders) where each is a dictionary mapping
        problem name to the loaded object.
    """
    models = {}
    encoders = {}
    for issue, model_file in MODELS.items():
        model_path = MODEL_DIR / model_file
        if model_path.exists():
            models[issue] = joblib.load(model_path)
        else:
            st.warning(f"Model file for {issue} not found at {model_path}.")
    for issue, enc_file in ENCODERS.items():
        enc_path = MODEL_DIR / enc_file
        if enc_path.exists():
            encoders[issue] = joblib.load(enc_path)
        else:
            st.warning(f"Encoder file for {issue} not found at {enc_path}.")
    return models, encoders


def compute_severity(issue: str, total_score: int) -> str:
    """Determine severity label based on total questionnaire score.

    Args:
        issue: One of "Anxiety", "Stress" or "Depression".
        total_score: Summed score from questionnaire responses.

    Returns:
        Severity string.
    """
    thresholds = SEVERITY_THRESHOLDS.get(issue, [])
    for (low, high, label) in thresholds:
        if low <= total_score <= high:
            return label
    return "Unknown"


def predict_issue(models: Dict[str, object], encoders: Dict[str, object],
                  responses: Dict[str, List[int]]) -> Tuple[str, Dict[str, float]]:
    """Use ML models to predict the most prominent mental health issue.

    Each model outputs the probability of the positive class (presence of that
    condition).  We compare probabilities across issues and select the one with
    the highest value.

    Args:
        models: Dict mapping issue to trained model.
        encoders: Dict mapping issue to preprocessing transformer.
        responses: Dict mapping issue to list of questionnaire scores (1–5).

    Returns:
        A tuple of (predicted_issue, probabilities) where probabilities maps
        each issue to its predicted probability.
    """
    probs: Dict[str, float] = {}
    for issue, model in models.items():
        if issue not in responses:
            continue
        # Raw responses are 1–5; convert to numpy array and reshape for encoder
        features = np.array(responses[issue]).reshape(1, -1)
        # Apply encoder if available
        if encoders.get(issue):
            features = encoders[issue].transform(features)
        # Some models (e.g. CatBoost) have predict_proba; fallback to predict
        try:
            prob = model.predict_proba(features)[0, 1]
        except Exception:
            pred = model.predict(features)[0]
            prob = float(pred)
        probs[issue] = float(prob)
    # Determine the issue with the highest probability
    predicted_issue = max(probs, key=probs.get) if probs else "Unknown"
    return predicted_issue, probs


def main() -> None:
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI‑Based Mental Health Assessment",
        layout="wide",
        page_icon="🧠",
    )
    st.title("AI‑Based Mental Health Assessment")
    st.markdown(
        """
        This tool uses a combination of validated questionnaires and machine‑learning models
        to help you understand your mental health.  Your responses remain
        confidential.  The results provided here are for informational purposes only and
        do not replace professional diagnosis or treatment.  If you feel
        suicidal, unsafe, or in crisis, contact emergency services or a
        trusted mental health professional immediately.
        """
    )

    # Load models and encoders
    models, encoders = load_models()

    # Collect demographic and lifestyle information
    with st.expander("Personal Information (optional)"):
        name = st.text_input("Name (optional)")
        age = st.selectbox(
            "Age Group",
            ["<18", "18–24", "25–34", "35–44", "45–54", "55+"],
            index=1,
        )
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        program = st.text_input("Programme/Major (optional)")
        year_of_study = st.selectbox(
            "Year of Study", ["1st", "2nd", "3rd", "4th", "Masters", "PhD", "Other"]
        )
        living_situation = st.selectbox(
            "Living Situation", ["Home", "Dormitory", "With roommates", "Other"]
        )
        relationship = st.selectbox(
            "Relationship Status", ["Single", "In a relationship", "Married", "Other"]
        )
        # Additional lifestyle questions can be added here

    # Collect questionnaire responses
    st.subheader("Questionnaires")
    st.markdown(
        "Please rate how often you have experienced each situation over the past two weeks (1 = Not at all; 5 = Almost always)."
    )
    responses: Dict[str, List[int]] = {}
    total_scores: Dict[str, int] = {}
    for issue, questions in QUESTIONNAIRES.items():
        st.markdown(f"### {issue} Questions")
        issue_scores: List[int] = []
        for q in questions:
            score = st.slider(q, min_value=1, max_value=5, value=1)
            issue_scores.append(score)
        responses[issue] = issue_scores
        total_scores[issue] = int(sum(issue_scores))
        st.markdown(
            f"Total {issue} Score: **{total_scores[issue]}** (Severity: **{compute_severity(issue, total_scores[issue])}**)"
        )

    if st.button("Predict Mental Health Status", type="primary"):
        # Perform prediction
        pred_issue, probabilities = predict_issue(models, encoders, responses)
        # Display results
        st.subheader("Prediction Results")
        st.markdown(f"**Most prominent issue:** {pred_issue}")
        st.markdown("**Predicted probabilities:**")
        prob_df = pd.DataFrame(
            [{"Issue": k, "Probability": f"{v:.2%}"} for k, v in probabilities.items()]
        )
        st.table(prob_df)
        # Show severity and suggestion for each issue
        for issue in ["Anxiety", "Stress", "Depression"]:
            severity = compute_severity(issue, total_scores.get(issue, 0))
            suggestion = SUGGESTIONS.get(issue, {}).get(severity, "")
            st.markdown(f"#### {issue}")
            st.markdown(f"Severity: **{severity}**")
            if suggestion:
                st.markdown(f"*Suggestion:* {suggestion}")

        # Log the session for researchers (optional)
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "name": name,
            "age_group": age,
            "gender": gender,
            "program": program,
            "year_of_study": year_of_study,
            "living_situation": living_situation,
            "relationship_status": relationship,
            "responses": responses,
            "total_scores": total_scores,
            "predicted_issue": pred_issue,
            "probabilities": probabilities,
        }
        try:
            LOG_DIR = MODEL_DIR / "logs"
            LOG_DIR.mkdir(exist_ok=True)
            log_file = LOG_DIR / f"log_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.info(f"Could not save log file: {e}")

    # Footer
    st.markdown("""---\nDeveloped by **Team Dual Core** as part of a research project.  This tool is for educational purposes and should not be considered medical advice.\n""")


if __name__ == "__main__":
    main()
