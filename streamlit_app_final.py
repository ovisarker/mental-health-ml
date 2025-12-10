"""
streamlit_app_final.py
----------------------

This Streamlit application provides a unified interface for assessing mental health
conditions (Anxiety, Stress, and Depression) based on validated psychometric
questionnaires (PSS‑10, GAD‑7, PHQ‑9).  The app collects demographic and
academic information from a student, presents the questionnaire items in
friendly language, and uses pre‑trained machine‑learning models to predict
binary risk levels for each condition.  Predictions are displayed with color
coding, logged to a CSV file, and downloadable by the user.

Assumptions:
  - The trained models are stored in the same directory as this script and
    loaded through the predict_all helper (see predict_all.py).
  - Input scales: PSS items (0–4), GAD and PHQ items (0–3).
  - This app is an educational tool and not a substitute for professional
    diagnosis.

Usage:
  Run via `streamlit run streamlit_app_final.py` after ensuring
  predict_all.py and the model files exist in the same directory.
"""

import streamlit as st
import pandas as pd
import datetime
import os

from predict_all import predict_all  # Import unified prediction helper


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="AI Mental Health Assessment",
        page_icon="🧠",
        layout="wide",
    )

    # Title and description
    st.title("🧠 AI‑Powered Mental Health Assessment System")
    st.markdown(
        """
    This research tool uses **Machine Learning models** trained on validated
    psychometric scales (**PSS‑10**, **GAD‑7**, **PHQ‑9**) to predict
    three key mental‑health conditions among students:

      • 😰 **Anxiety**
      • 😓 **Stress**
      • 😞 **Depression**

    > ⚠️ **Disclaimer:** This application is for educational and research
    purposes. It is **not a clinical diagnostic tool**. For concerns about
    mental health, please consult a licensed professional.
    """,
        unsafe_allow_html=True,
    )

    # Sidebar: collect basic student information
    st.sidebar.header("📌 Student Information")
    age = st.sidebar.selectbox("Age range", ["18‑22", "23‑27", "28‑32", "33+"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    university = st.sidebar.text_input("University")
    department = st.sidebar.text_input("Department")
    academic_year = st.sidebar.selectbox(
        "Academic Year / Semester",
        ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    )
    cgpa = st.sidebar.selectbox(
        "Current CGPA",
        ["<2.50", "2.50‑3.00", "3.00‑3.50", "3.50‑4.00"],
    )
    scholarship = st.sidebar.selectbox(
        "Scholarship / Waiver", ["No", "Yes ‑ Partial", "Yes ‑ Full"],
    )

    # Tabs for questionnaire sections
    tab_pss, tab_gad, tab_phq = st.tabs([
        "🟨 PSS‑10 (Stress)",
        "🟦 GAD‑7 (Anxiety)",
        "🟥 PHQ‑9 (Depression)",
    ])

    # Define questions for each questionnaire
    pss_questions = [
        "How often did you feel upset due to academic issues?",
        "How often did you feel unable to control important academic matters?",
        "How often did academic pressure make you feel nervous or stressed?",
        "How often did you feel unable to cope with academic tasks (assignments, quizzes, exams)?",
        "How often did you feel confident in handling university‑related problems? (Reverse scored)",
        "How often did you feel that things were going your way academically? (Reverse scored)",
        "How often were you able to control irritations caused by academic issues? (Reverse scored)",
        "How often did you feel your academic performance was satisfactory? (Reverse scored)",
        "How often did you feel anger due to poor academic outcomes beyond your control?",
        "How often did academic difficulties pile up so high that you could not overcome them?",
    ]

    gad_questions = [
        "Feeling nervous, anxious, or on edge because of academic pressure?",
        "Not being able to stop or control worrying about academic issues?",
        "Worrying too much about different university‑related things?",
        "Trouble relaxing due to academic stress?",
        "Being so restless that it's hard to sit still when thinking about studies?",
        "Becoming easily annoyed or irritable because of academic workload?",
        "Feeling afraid as if something awful might happen academically?",
    ]

    phq_questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself or that you are a failure?",
        "Trouble concentrating on reading, studies, or watching something?",
        "Moving or speaking so slowly that others noticed — or the opposite (restless/fidgety)?",
        "Thoughts that you would be better off dead, or of hurting yourself in some way?",
    ]

    # Dictionaries to hold slider responses
    pss = {}
    gad = {}
    phq = {}

    # Collect PSS responses
    with tab_pss:
        st.subheader("Perceived Stress Scale (PSS‑10)")
        st.caption(
            "Scale: 0 = Never, 1 = Almost never, 2 = Sometimes, 3 = Fairly often, 4 = Very often"
        )
        for i, question in enumerate(pss_questions, start=1):
            pss[f"PSS{i}"] = st.slider(question, min_value=0, max_value=4, value=0)

    # Collect GAD responses
    with tab_gad:
        st.subheader("Generalized Anxiety Disorder (GAD‑7)")
        st.caption(
            "Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day"
        )
        for i, question in enumerate(gad_questions, start=1):
            gad[f"GAD{i}"] = st.slider(question, min_value=0, max_value=3, value=0)

    # Collect PHQ responses
    with tab_phq:
        st.subheader("Patient Health Questionnaire (PHQ‑9)")
        st.caption(
            "Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day"
        )
        for i, question in enumerate(phq_questions, start=1):
            phq[f"PHQ{i}"] = st.slider(question, min_value=0, max_value=3, value=0)

    # Divider
    st.markdown("---")

    # Button for prediction
    if st.button("🔍 Run AI Prediction"):
        # Construct input record
        user_input = {
            "Age": age,
            "Gender": gender,
            "University": university,
            "Department": department,
            "Academic_Year": academic_year,
            "Current_CGPA": cgpa,
            "waiver_or_scholarship": scholarship,
        }
        user_input.update(pss)
        user_input.update(gad)
        user_input.update(phq)

        try:
            result = predict_all(user_input)
        except FileNotFoundError as fnf_error:
            st.error(
                f"Model file not found: {fnf_error}.\n"
                "Ensure that the trained model files are present in the same directory as this script."
            )
            return
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        # Display predictions with color boxes
        def colored_box(text: str, background_color: str) -> None:
            st.markdown(
                f"""
                <div style="background-color:{background_color}; padding:15px; border-radius:10px; margin:5px 0; font-size:18px;">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("📊 Prediction Results")
        # Anxiety
        if "High" in result["Anxiety"]:
            colored_box(f"😰 Anxiety: <b>{result['Anxiety']}</b>", "#ffcccc")
        else:
            colored_box(f"😌 Anxiety: <b>{result['Anxiety']}</b>", "#d4ffd4")
        # Stress
        if "High" in result["Stress"]:
            colored_box(f"😓 Stress: <b>{result['Stress']}</b>", "#ffe0b3")
        else:
            colored_box(f"😌 Stress: <b>{result['Stress']}</b>", "#d4ffd4")
        # Depression
        if "Present" in result["Depression"]:
            colored_box(f"😞 Depression: <b>{result['Depression']}</b>", "#ffd6d6")
        else:
            colored_box(f"🙂 Depression: <b>{result['Depression']}</b>", "#d4ffd4")

        # Log prediction
        log_entry = user_input.copy()
        log_entry.update(result)
        log_entry["Timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        log_df = pd.DataFrame([log_entry])

        log_file = "prediction_logs.csv"
        if not os.path.exists(log_file):
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode="a", index=False, header=False)

        st.success("✅ Prediction saved to log file.")
        st.download_button(
            label="⬇ Download This Prediction (CSV)",
            data=log_df.to_csv(index=False),
            file_name="mh_prediction_result.csv",
            mime="text/csv",
        )

        st.info(
            "These predictions are based on machine‑learning patterns from student data.\n"
            "For any serious concerns, please consult a mental health professional."
        )


if __name__ == "__main__":
    main()
