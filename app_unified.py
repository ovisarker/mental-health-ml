# app.py
import streamlit as st
import pandas as pd

# Import unified ML pipeline
from unified_mental_health_pipeline import (
    predict_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="ML Mental Health Assessment", layout="wide")

st.title("🧠 ML-Based Student Mental Health Assessment System")
st.write("""
This tool predicts **Anxiety**, **Stress**, and **Depression** using  
Machine Learning (Logistic Regression + Preprocessing + Explainability).

It also determines the **Overall Mental Health Status** and  
shows **Explainable AI** insights (top influential features).
""")
st.markdown("---")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.subheader("📋 Student Profile & Questionnaire Input")

with st.form("mh_form"):

    col1, col2 = st.columns(2)

    # ============================
    # Demographic Inputs
    # ============================
    with col1:
        st.markdown("### 👤 Demographic & Academic Info")
        age = st.number_input("Age", 16, 40, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        university = st.text_input("University")
        department = st.text_input("Department")
        year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
        cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
        scholarship = st.selectbox("Scholarship/Waiver", ["Yes", "No"])

        st.markdown("#### Scale for All Questions")
        st.info("0 = Never • 1 = Almost Never • 2 = Sometimes • 3 = Fairly Often • 4 = Very Often")

    # ============================
    # Questionnaire Inputs
    # ============================
    with col2:

        # ---------- PSS-10 (Stress) ----------
        st.markdown("### 🟦 PSS-10 (Stress – Academic Context)")
        PSS_questions = [
            "How often did you feel upset due to academic issues?",
            "How often did you feel unable to control important academic matters?",
            "How often did academic pressure make you feel nervous or stressed?",
            "How often did you feel unable to cope with assignments/exams?",
            "How often did you feel confident in handling university problems? (Reverse)",
            "How often did you feel things were going well academically? (Reverse)",
            "How often were you able to control irritation from academic issues? (Reverse)",
            "How often did you feel your academic performance was satisfactory? (Reverse)",
            "How often did you feel anger due to poor academic outcomes?",
            "How often did academic difficulties pile up beyond control?"
        ]
        PSS = []
        for i,q in enumerate(PSS_questions):
            PSS.append(st.number_input(f"PSS{i+1}: {q}", 0, 4, 1))

        # ---------- GAD-7 (Anxiety) ----------
        st.markdown("### 🟩 GAD-7 (Anxiety – Academic Context)")
        GAD_questions = [
            "How often did you feel nervous or on edge due to academic pressure?",
            "How often were you unable to stop worrying about academic issues?",
            "How often did academic pressure stop you from relaxing?",
            "How often were you easily annoyed or irritated due to academics?",
            "How often did you worry too much about academic matters?",
            "How often did restlessness make it hard to sit still due to stress?",
            "How often did you feel afraid that something bad might happen academically?"
        ]
        GAD = []
        for i,q in enumerate(GAD_questions):
            GAD.append(st.number_input(f"GAD{i+1}: {q}", 0, 4, 1))

        # ---------- PHQ-9 (Depression) ----------
        st.markdown("### 🟥 PHQ-9 (Depression Symptoms)")
        PHQ_questions = [
            "Little interest or pleasure in activities?",
            "Feeling down, depressed, or hopeless?",
            "Trouble sleeping (too much or too little)?",
            "Feeling tired or low energy?",
            "Poor appetite or overeating?",
            "Feeling bad about yourself or like a failure?",
            "Trouble concentrating (study, books, TV)?",
            "Moving/speaking slower or faster than usual?",
            "Thoughts of harming yourself or being better off dead?"
        ]
        PHQ = []
        for i,q in enumerate(PHQ_questions):
            PHQ.append(st.number_input(f"PHQ{i+1}: {q}", 0, 4, 1))

    submitted = st.form_submit_button("🔍 Run ML Assessment")

# ---------------------------------------------------------
# RUN ML
# ---------------------------------------------------------
if submitted:
    st.markdown("---")
    st.subheader("🔎 Machine Learning Predictions")

    # Build input dict for ML model
    data = {
        "Age": age, "Gender": gender, "University": university,
        "Department": department, "Academic_Year": year,
        "Current_CGPA": cgpa, "waiver_or_scholarship": scholarship
    }

    for i in range(10):
        data[f"PSS{i+1}"] = PSS[i]

    for i in range(7):
        data[f"GAD{i+1}"] = GAD[i]

    for i in range(9):
        data[f"PHQ{i+1}"] = PHQ[i]

    # Run ML Pipeline
    anx, stress, dep, main_issue = predict_for_student(data)

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Anxiety", "Present" if anx == 1 else "Absent")
    with colB:
        st.metric("Stress", "Present" if stress == 1 else "Absent")
    with colC:
        st.metric("Depression", "Present" if dep == 1 else "Absent")

    st.markdown("### 🧠 Overall Mental Health Status")
    st.success(f"**{main_issue}**")

    st.markdown("---")

    # -----------------------------------------------------
    # XAI SECTION
    # -----------------------------------------------------
    st.subheader("📘 Explainable AI (Top Influential Features)")

    def top_features(model, cols, k=8):
        coefs = model.coef_[0]
        df_feat = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
        df_feat["Abs"] = df_feat["Coefficient"].abs()
        return df_feat.sort_values("Abs", ascending=False).head(k)

    top_anx = top_features(anx_clf_num, x_numeric.columns)
    top_str = top_features(str_clf_num, x_numeric.columns)
    top_dep = top_features(dep_clf_num, x_numeric.columns)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### 🔵 Anxiety – Key Features")
        st.dataframe(top_anx[["Feature", "Coefficient"]])

    with c2:
        st.write("### 🟡 Stress – Key Features")
        st.dataframe(top_str[["Feature", "Coefficient"]])

    with c3:
        st.write("### 🔴 Depression – Key Features")
        st.dataframe(top_dep[["Feature", "Coefficient"]])

    st.markdown("---")
    st.info("""
    ⚠️ This tool is for **screening & research** purposes only.  
    It identifies **risk patterns**, not a clinical diagnosis.
    """)
