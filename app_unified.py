"""
Streamlit Mental Health Assessment App (Full Code)
-------------------------------------------------
This Streamlit application predicts whether a student is experiencing anxiety,
stress, or depression based on questionnaire responses and demographic inputs.
It leverages machine learning models trained in the accompanying
`unified_mental_health_pipeline.py` module. The app also provides an
overall mental-health status (dominant issue) and basic explainability
information by showing which numeric questionnaire items contribute most
to each prediction.

To run this app:
 1. Ensure `Processed.csv` and `unified_mental_health_pipeline.py` are in the
    same directory as this script.
 2. From a terminal, run `streamlit run full_streamlit_app.py`.

The app presents the PHQ-9, GAD-7, and PSS-10 questionnaires with
student-friendly labels and collects demographic and academic data.
It then calls `predict_for_student()` from the pipeline module to
generate predictions.

Note: This tool provides early risk assessment and is not a clinical
diagnosis.
"""

import streamlit as st
import pandas as pd

from unified_mental_health_pipeline import (
    predict_for_student,
    anx_clf_num,
    str_clf_num,
    dep_clf_num,
    x_numeric,
)


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="ML Mental Health Assessment", layout="wide")

    st.title("🧠 ML-Based Student Mental Health Assessment")
    st.write(
        """
This tool predicts **Anxiety**, **Stress**, and **Depression** using Machine
Learning. It combines questionnaire scores with demographic information to
provide a risk assessment and then derives an overall mental-health
status (dominant issue). It also displays basic explainability by
showing which numeric questionnaire items most influence each prediction.
        """
    )
    st.markdown("---")

    st.subheader("📋 Student Profile & Questionnaire Input")

    with st.form("mh_form"):
        # Demographic and academic inputs
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👤 Demographic & Academic Info")
            age = st.number_input("Age", min_value=16, max_value=40, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            university = st.text_input("University")
            department = st.text_input("Department")
            year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
            cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=4.0, value=3.0)
            scholarship = st.selectbox("Scholarship/Waiver", ["Yes", "No"])

            st.markdown("#### Question Scale")
            st.info(
                "0 = Never • 1 = Almost Never • 2 = Sometimes • 3 = Fairly Often • 4 = Very Often"
            )

        # Questionnaire inputs
        with col2:
            st.markdown("### 🟦 PSS-10 (Stress – Academic Context)")
            pss_questions = [
                "How often did you feel upset due to academic issues?",
                "How often did you feel unable to control important academic matters?",
                "How often did academic pressure make you feel nervous or stressed?",
                "How often did you feel unable to cope with assignments/exams?",
                "How often did you feel confident in handling university problems? (Reverse)",
                "How often did you feel things were going well academically? (Reverse)",
                "How often were you able to control irritation from academic issues? (Reverse)",
                "How often did you feel your academic performance was satisfactory? (Reverse)",
                "How often did you feel anger due to poor academic outcomes?",
                "How often did academic difficulties pile up beyond control?",
            ]
            PSS = [st.number_input(f"PSS{i+1}: {q}", min_value=0, max_value=4, value=1) for i, q in enumerate(pss_questions)]

            st.markdown("### 🟩 GAD-7 (Anxiety – Academic Context)")
            gad_questions = [
                "How often did you feel nervous or on edge due to academic pressure?",
                "How often were you unable to stop worrying about academic issues?",
                "How often did academic pressure stop you from relaxing?",
                "How often were you easily annoyed or irritated due to academics?",
                "How often did you worry too much about academic matters?",
                "How often did restlessness make it hard to sit still due to stress?",
                "How often did you feel afraid that something bad might happen academically?",
            ]
            GAD = [st.number_input(f"GAD{i+1}: {q}", min_value=0, max_value=4, value=1) for i, q in enumerate(gad_questions)]

            st.markdown("### 🟥 PHQ-9 (Depression Symptoms)")
            phq_questions = [
                "Little interest or pleasure in activities?",
                "Feeling down, depressed, or hopeless?",
                "Trouble sleeping (too much or too little)?",
                "Feeling tired or low energy?",
                "Poor appetite or overeating?",
                "Feeling bad about yourself or like a failure?",
                "Trouble concentrating (study, books, TV)?",
                "Moving/speaking slower or faster than usual?",
                "Thoughts of harming yourself or being better off dead?",
            ]
            PHQ = [st.number_input(f"PHQ{i+1}: {q}", min_value=0, max_value=4, value=1) for i, q in enumerate(phq_questions)]

        submitted = st.form_submit_button("🔍 Run Assessment")

    # Run predictions when form submitted
    if submitted:
        st.markdown("---")
        st.subheader("🔎 Predictions")

        # Build input dictionary matching pipeline feature names
        student_data = {
            "Age": age,
            "Gender": gender,
            "University": university,
            "Department": department,
            "Academic_Year": year,
            "Current_CGPA": cgpa,
            "waiver_or_scholarship": scholarship,
        }
        # Add PSS, GAD, PHQ responses
        for i, val in enumerate(PSS):
            student_data[f"PSS{i+1}"] = val
        for i, val in enumerate(GAD):
            student_data[f"GAD{i+1}"] = val
        for i, val in enumerate(PHQ):
            student_data[f"PHQ{i+1}"] = val

        # Predict
        anx_pred, str_pred, dep_pred, main_status = predict_for_student(student_data)

        # Display results
        colA, colB, colC = st.columns(3)
        colA.metric("Anxiety", "Present" if anx_pred == 1 else "Absent")
        colB.metric("Stress", "Present" if str_pred == 1 else "Absent")
        colC.metric("Depression", "Present" if dep_pred == 1 else "Absent")

        st.markdown("### 🧠 Overall Mental Health Status")
        st.success(main_status)

        st.markdown("---")
        st.subheader("📘 Explainable AI: Top Numeric Features")

        # Helper function to get top-k features based on coefficient magnitude
        def get_top_features(model, columns, k=8):
            coefs = model.coef_[0]
            df_features = pd.DataFrame({"Feature": columns, "Coefficient": coefs})
            df_features["Abs"] = df_features["Coefficient"].abs()
            return df_features.sort_values(by="Abs", ascending=False).head(k)

        # Extract top features for each condition
        top_anx = get_top_features(anx_clf_num, x_numeric.columns)
        top_str = get_top_features(str_clf_num, x_numeric.columns)
        top_dep = get_top_features(dep_clf_num, x_numeric.columns)

        # Display top features tables
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("### 🔵 Anxiety – Key Features")
            st.table(top_anx[["Feature", "Coefficient"]])
        with c2:
            st.write("### 🟡 Stress – Key Features")
            st.table(top_str[["Feature", "Coefficient"]])
        with c3:
            st.write("### 🔴 Depression – Key Features")
            st.table(top_dep[["Feature", "Coefficient"]])

        st.markdown("---")
        st.info(
            "⚠️ This tool is for screening and research purposes only. "
            "It is not a clinical diagnosis and should not be used as such."
        )


if __name__ == "__main__":
    main()
