"""
Streamlit app for a machine‑learning driven mental‑health assessment.

This app collects responses to standardised mental‑health screening
questionnaires (GAD‑7 for anxiety, PSS‑10 for stress, and PHQ‑9 for
depression) along with a few demographic and lifestyle indicators.  It
uses pre‑trained machine‑learning models to classify the most
prominent mental‑health issue for the user (anxiety, stress or
depression).  It then evaluates the severity level of each issue
based on the total questionnaire score and provides appropriate
suggestions.

Key features:
  • Unified assessment – the user answers questions for anxiety,
    stress and depression in one session.  After submission, the
    app predicts which issue is most prominent using trained
    classifiers and shows severity for each issue.
  • Machine‑learning models – logistic regression and CatBoost
    classifiers stored in joblib files are loaded at runtime and
    predict probabilities for each issue.
  • Custom severity assessment – severity levels are calculated
    according to accepted clinical thresholds for each questionnaire.
  • Friendly UI – built with Streamlit, the app guides the user
    through each set of questions and presents results and
    self‑help suggestions.
  • Footer – acknowledges the project team (Team Dual Core).

Note: this tool does not provide a formal diagnosis.  It is an
educational resource to help users understand their wellbeing.  If
you have urgent concerns or think you might be in crisis, please
contact a qualified mental‑health professional or emergency services.
"""

import joblib
import numpy as np
import streamlit as st
from pathlib import Path


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Paths to the trained models and encoders.  These files should exist
# in the same directory as this script.  The models are binary
# classifiers (1=issue present, 0=not present) trained on previously
# collected screening data.  Encoders are loaded but not used for
# feature transformation; instead they remain available in case the
# underlying models expect categorical encodings of labels.
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


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_models(models_dict: dict[str, str]) -> dict[str, object]:
    """Load each machine‑learning model from disk.

    Parameters
    ----------
    models_dict : dict
        Mapping of issue names to filenames.

    Returns
    -------
    dict
        Mapping of issue names to loaded model objects.
    """
    models: dict[str, object] = {}
    for issue, filename in models_dict.items():
        path = Path(filename)
        if path.exists():
            models[issue] = joblib.load(path)
        else:
            st.warning(f"Model file for {issue} not found: {filename}")
    return models


def load_encoders(encoder_dict: dict[str, str]) -> dict[str, object]:
    """Load label encoders from disk.

    These encoders map numeric predictions back to human‑readable labels.
    They are not used to transform feature inputs.  If an encoder file
    is missing, a warning is displayed but prediction still proceeds.

    Parameters
    ----------
    encoder_dict : dict
        Mapping of issue names to filenames.

    Returns
    -------
    dict
        Mapping of issue names to loaded encoders.
    """
    encoders: dict[str, object] = {}
    for issue, filename in encoder_dict.items():
        path = Path(filename)
        if path.exists():
            encoders[issue] = joblib.load(path)
        else:
            st.warning(f"Encoder file for {issue} not found: {filename}")
    return encoders


def calculate_severity(issue: str, score: int) -> str:
    """Return a descriptive severity level given an issue and its score.

    Parameters
    ----------
    issue : {'Anxiety', 'Stress', 'Depression'}
        The mental‑health issue being scored.
    score : int
        The total questionnaire score.

    Returns
    -------
    str
        A human‑readable severity rating.
    """
    if issue == "Anxiety":
        if score <= 4:
            return "Minimal or no anxiety"
        elif score <= 9:
            return "Mild anxiety"
        elif score <= 14:
            return "Moderate anxiety"
        else:
            return "Severe anxiety"
    elif issue == "Stress":
        # PSS scores range from 0–40 (10 items x 0–4).  Suggested bands:
        if score <= 13:
            return "Low stress"
        elif score <= 26:
            return "Moderate stress"
        else:
            return "High stress"
    elif issue == "Depression":
        if score <= 4:
            return "Minimal depression"
        elif score <= 9:
            return "Mild depression"
        elif score <= 14:
            return "Moderate depression"
        elif score <= 19:
            return "Moderately severe depression"
        else:
            return "Severe depression"
    return "Unknown"


def get_suggestions(issue: str, severity: str) -> str:
    """Return general self‑help suggestions based on issue and severity.

    This function supplies generic recommendations.  For serious
    symptoms (e.g. severe depression), encourage the user to seek
    professional help immediately.

    Parameters
    ----------
    issue : str
        Mental‑health issue name.
    severity : str
        Severity description.

    Returns
    -------
    str
        Suggested actions or resources.
    """
    # Simple mapping; could be expanded with more nuance.
    suggestions = {
        "Anxiety": {
            "Minimal or no anxiety": "Maintain healthy routines and mindfulness.",
            "Mild anxiety": "Consider stress‑management techniques like deep breathing and regular exercise.",
            "Moderate anxiety": "Practice cognitive‑behavioural strategies, relaxation techniques, and talk to someone you trust.",
            "Severe anxiety": "Consult a mental‑health professional. Explore therapy or medication options."
        },
        "Stress": {
            "Low stress": "Keep up your current stress management practices and good self‑care.",
            "Moderate stress": "Prioritise tasks, set boundaries, and practice relaxation or meditation.",
            "High stress": "Seek support from friends, family or a counsellor. Develop a stress reduction plan."
        },
        "Depression": {
            "Minimal depression": "Stay active, socialise, and maintain good sleep habits.",
            "Mild depression": "Engage in enjoyable activities and consider talking to a counsellor.",
            "Moderate depression": "Reach out to a professional for therapy, and lean on your support network.",
            "Moderately severe depression": "Seek professional treatment; therapy and possibly medication may be needed.",
            "Severe depression": "Contact a mental‑health professional urgently. If you feel unsafe, contact emergency services." 
        }
    }
    return suggestions.get(issue, {}).get(severity, "No specific suggestions available.")


def predict_issue(
    models: dict[str, object],
    responses: dict[str, list[int]]
) -> tuple[str, dict[str, float]]:
    """Predict the most prominent mental‑health issue.

    Each model outputs a probability that the user has the corresponding
    issue.  The function returns the issue with the highest probability
    and a dictionary of probabilities per issue.

    Parameters
    ----------
    models : dict
        Mapping of issue names to sklearn/catboost models.
    responses : dict
        Mapping of issue names to lists of numeric questionnaire
        responses.

    Returns
    -------
    tuple
        (predicted issue, probability dict)
    """
    probabilities: dict[str, float] = {}
    for issue, model in models.items():
        answers = responses.get(issue, [])
        if not answers:
            continue
        # Convert to 2D array for model input
        features = np.array(answers).reshape(1, -1)
        try:
            # Some models (e.g. CatBoost) implement predict_proba
            prob = model.predict_proba(features)[0, 1]
        except Exception:
            # Fall back to decision_function or predict for models
            # without predict_proba.  The result is scaled to 0–1.
            try:
                raw = model.decision_function(features)
                # Min‑max scale; assume raw outputs range roughly between -inf and +inf
                prob = 1 / (1 + np.exp(-raw))
            except Exception:
                prob = float(model.predict(features)[0])
        probabilities[issue] = float(prob)
    # Select issue with maximum probability
    if probabilities:
        predicted_issue = max(probabilities, key=probabilities.get)
    else:
        predicted_issue = "None"
    return predicted_issue, probabilities


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Mental Health Assessment System",
        page_icon="🧠",
        layout="centered",
    )
    st.title("Mental Health Assessment System")
    st.markdown(
        "This application helps you self‑assess for anxiety, stress, and depression "
        "using standard screening questionnaires.  Your responses are processed "
        "by machine‑learning models to estimate which issue is most prominent "
        "and provide severity levels and suggestions.\n\n"
        "**Disclaimer:** This tool is not a clinical diagnosis.  Always consult a "
        "qualified healthcare provider for professional advice."
    )

    # Demographic and lifestyle inputs
    st.header("Basic Information")
    age_group = st.selectbox(
        "What is your age group?",
        ["<18", "18–24", "25–34", "35–44", "45–54", "55+"],
        index=1,
    )
    sleep_quality = st.selectbox(
        "How would you describe your sleep quality?",
        ["Excellent", "Good", "Average", "Poor"],
        index=2,
    )
    physical_activity = st.selectbox(
        "How often do you engage in physical activity (per week)?",
        ["Rarely", "1–2 times", "3–4 times", "5+ times"],
        index=1,
    )

    # Collect responses for each issue
    responses: dict[str, list[int]] = {}
    total_scores: dict[str, int] = {}
    issues_questions = {
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
            "In the last month, how often have you felt confident about your ability to handle your personal problems? (reverse scored)",
            "In the last month, how often have you felt that things were going your way? (reverse scored)",
            "In the last month, how often have you found that you could not cope with all the things that you had to do?",
            "In the last month, how often have you been able to control irritations in your life? (reverse scored)",
            "In the last month, how often have you felt that you were on top of things? (reverse scored)",
            "In the last month, how often have you been angered because of things that were outside your control?",
            "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
        ],
        "Depression": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
            "Trouble concentrating on things, such as reading the newspaper or watching television",
            "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
            "Thoughts that you would be better off dead, or of hurting yourself in some way",
        ],
    }

    # Likert scale options for questionnaires (0–3 for GAD/PHQ; 0–4 for PSS)
    options_gad_phq = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3,
    }
    options_pss = {
        "Never": 0,
        "Almost never": 1,
        "Sometimes": 2,
        "Fairly often": 3,
        "Very often": 4,
    }

    # Form for Anxiety questions
    st.header("Anxiety (GAD‑7)")
    anxiety_answers: list[int] = []
    for q in issues_questions["Anxiety"]:
        response = st.selectbox(q, list(options_gad_phq.keys()), key=f"anx_{q}")
        anxiety_answers.append(options_gad_phq[response])
    responses["Anxiety"] = anxiety_answers
    total_scores["Anxiety"] = sum(anxiety_answers)

    # Form for Stress questions (PSS‑10)
    st.header("Stress (PSS‑10)")
    stress_answers: list[int] = []
    for q in issues_questions["Stress"]:
        response = st.selectbox(q, list(options_pss.keys()), key=f"str_{q}")
        score = options_pss[response]
        # Reverse score items 4, 5, 7, 8 (0<->4, 1<->3) by mapping 0→4, 1→3, 2→2, 3→1, 4→0
        if "reverse scored" in q:
            score = 4 - score
        stress_answers.append(score)
    responses["Stress"] = stress_answers
    total_scores["Stress"] = sum(stress_answers)

    # Form for Depression questions (PHQ‑9)
    st.header("Depression (PHQ‑9)")
    depression_answers: list[int] = []
    for q in issues_questions["Depression"]:
        response = st.selectbox(q, list(options_gad_phq.keys()), key=f"dep_{q}")
        depression_answers.append(options_gad_phq[response])
    responses["Depression"] = depression_answers
    total_scores["Depression"] = sum(depression_answers)

    # Once the user is ready, run prediction
    if st.button("Assess My Mental Health", type="primary"):
        st.subheader("Results")
        # Load models and encoders
        models = load_models(MODEL_FILES)
        encoders = load_encoders(ENCODER_FILES)
        # Predict the most prominent issue
        pred_issue, probabilities = predict_issue(models, responses)
        # Display probabilities
        st.markdown("### Predicted Prominent Issue")
        if pred_issue == "None":
            st.write("Unable to determine the most prominent issue from your responses.")
        else:
            st.write(f"**{pred_issue}** with probability {probabilities[pred_issue]:.2f}")
        # Show probability for each issue
        st.markdown("### Probability by Issue")
        for issue, prob in probabilities.items():
            st.write(f"{issue}: {prob:.2f}")
        # Display severity and suggestions for each issue
        st.markdown("### Severity and Recommendations")
        for issue in ["Anxiety", "Stress", "Depression"]:
            score = total_scores.get(issue, 0)
            severity = calculate_severity(issue, score)
            suggestion = get_suggestions(issue, severity)
            st.write(f"**{issue}** – Total Score: {score}, Severity: {severity}")
            st.write(f"Recommendation: {suggestion}")
    # Footer
    st.markdown("---")
    st.markdown(
        "Developed by **Team Dual Core** | This application is for educational purposes only."
    )


if __name__ == "__main__":
    main()
