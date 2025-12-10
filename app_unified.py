# app.py
import streamlit as st
import pandas as pd

# 🔹 Import from your unified ML pipeline file
from unified_mental_health_pipeline import (
    predict_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num
)

# ---------------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML-based Student Mental Health Assessment",
    layout="wide"
)

st.title("🧠 Machine Learning-Based Student Mental Health Assessment")
st.write("""
This system uses **Machine Learning** to predict:

- Anxiety (present / absent)  
- Stress (present / absent)  
- Depression (present / absent)  

Then it combines them into an **Overall Mental Health Status**  
and shows basic **Explainable AI (XAI)** insights.
""")
st.markdown("---")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.subheader("📋 Student Information & Questionnaire")

with st.form("input_form"):

    col1, col2 = st.columns(2)

    # -------- Demographic / Academic --------
    with col1:
        st.markdown("### 👤 Demographic & Academic Info")
        age = st.number_input("Age", 16, 40, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        university = st.text_input("University", "")
        department = st.text_input("Department", "")
        year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
        cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
        scholarship = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

        st.markdown("#### Scale for all questions")
        st.markdown("""
        **0 = Never**  
        **1 = Almost Never**  
        **2 = Sometimes**  
        **3 = Fairly Often**  
        **4 = Very Often**
        """)

    with col2:
        # -------- PSS-10 (Stress) --------
        st.markdown("### 🟦 PSS-10 (Perceived Stress – Academic)")

        PSS_questions = [
            "How often did you feel upset due to academic issues?",
            "How often did you feel unable to control important academic matters?",
            "How often did academic pressure make you feel nervous or stressed?",
            "How often did you feel unable to cope with academic tasks (assignments, quizzes, exams)?",
            "How often did you feel confident in handling university-related problems? (Reverse scored)",
            "How often did you feel that things were going your way academically? (Reverse scored)",
            "How often were you able to control irritations caused by academic issues? (Reverse scored)",
            "How often did you feel your academic performance was satisfactory? (Reverse scored)",
            "How often did you feel anger due to poor academic outcomes beyond your control?",
            "How often did academic difficulties pile up so high that you could not overcome them?"
        ]
        PSS = []
        for i, q in enumerate(PSS_questions):
            PSS.append(
                st.number_input(f"PSS{i+1}: {q}", min_value=0, max_value=4, value=1)
            )

        # -------- GAD-7 (Anxiety) --------
        st.markdown("### 🟩 GAD-7 (Anxiety – Academic Context)")

        GAD_questions = [
            "How often did you feel nervous or on edge due to academic pressure?",
            "How often were you unable to stop worrying about academic issues?",
            "How often did academic pressure stop you from relaxing?",
            "How often were you easily annoyed or irritated due to academics?",
            "How often did you worry too much about academic matters?",
            "How often did restlessness make it hard to sit still due to academic stress?",
            "How often did you feel afraid as if something bad might happen academically?"
        ]
        GAD = []
        for i, q in enumerate(GAD_questions):
            GAD.append(
                st.number_input(f"GAD{i+1}: {q}", min_value=0, max_value=4, value=1)
            )

        # -------- PHQ-9 (Depression) --------
        st.markdown("### 🟥 PHQ-9 (Depression Symptoms)")

        PHQ_questions = [
            "Little interest or pleasure in activities?",
            "Feeling down, depressed, or hopeless?",
            "Trouble sleeping (too much or too little)?",
            "Feeling tired or low energy?",
            "Poor appetite or overeating?",
            "Feeling bad about yourself or feeling like a failure?",
            "Trouble concentrating (books, study, TV)?",
            "Moving or speaking slower/faster than usual?",
            "Thoughts of harming yourself or being better off dead?"
        ]
        PHQ = []
        for i, q in enumerate(PHQ_questions):
            PHQ.append(
                st.number_input(f"PHQ{i+1}: {q}", min_value=0, max_value=4, value=1)
            )

    submitted = st.form_submit_button("🔍 Run ML Assessment")

# ---------------------------------------------------------
# RUN ML MODELS WHEN SUBMIT
# ---------------------------------------------------------
if submitted:
    st.markdown("---")
    st.subheader("🔎 ML Prediction Results")

    # Build data dict exactly matching training columns
    data = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship,
    }

    # Add PSS
    for i in range(10):
        data[f"PSS{i+1}"] = PSS[i]

    # Add GAD
    for i in range(7):
        data[f"GAD{i+1}"] = GAD[i]

    # Add PHQ
    for i in range(9):
        data[f"PHQ{i+1}"] = PHQ[i]

    # Call unified ML pipeline
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
    # BASIC XAI – TOP NUMERIC FEATURES
    # -----------------------------------------------------
    st.subheader("📘 Explainable AI (XAI): Key Influential Features")

    st.write("""
    Below are the most influential numeric features (questionnaire items, etc.)
    for each condition, based on **Logistic Regression coefficients**.
    Positive coefficient = increases risk, negative = reduces risk.
    """)

    # Helper to extract top-k features from LR model
    def get_top_features(model, cols, top_k=8):
        coefs = model.coef_[0]
        df_feat = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
        df_feat["|Coefficient|"] = df_feat["Coefficient"].abs()
        return df_feat.sort_values("|Coefficient|", ascending=False).head(top_k)

    top_anx = get_top_features(anx_clf_num, x_numeric.columns)
    top_str = get_top_features(str_clf_num, x_numeric.columns)
    top_dep = get_top_features(dep_clf_num, x_numeric.columns)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### 🔵 Anxiety – Top Features")
        st.dataframe(top_anx[["Feature", "Coefficient"]])

    with c2:
        st.write("### 🟡 Stress – Top Features")
        st.dataframe(top_str[["Feature", "Coefficient"]])

    with c3:
        st.write("### 🔴 Depression – Top Features")
        st.dataframe(top_dep[["Feature", "Coefficient"]])

    st.markdown("---")
    st.info(
        "⚠️ This tool is for **screening & research** only and does **not** provide a "
        "clinical diagnosis. It uses ML models trained on student data to identify early risk patterns."
    )
