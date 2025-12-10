# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Unified ML Pipeline import (your merged ML code)
from unified_mental_health_pipeline import predict_for_student, determine_main_issue
from unified_mental_health_pipeline import (
    anxiety_model, stress_model, depression_model,
    x_numeric, train_lr_numeric, show_top_features
)

st.set_page_config(page_title="Mental Health Assessment (ML Based)", layout="wide")

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
st.title("🧠 Machine Learning-Based Mental Health Assessment System")
st.write("""
This system predicts **Anxiety**, **Stress**, and **Depression** using Machine Learning 
(Logistic Regression + Preprocessing Pipeline).  
It also determines the **Overall Mental Health Status** and shows **Explainable AI insights**.
""")

st.markdown("---")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------

st.subheader("📋 Student Information & Questionnaire Input")

with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 16, 40, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        university = st.text_input("University")
        department = st.text_input("Department")
        year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
        cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
        scholarship = st.selectbox("Scholarship/Waiver", ["Yes", "No"])

    with col2:
        st.markdown("### 🟦 PSS (Stress) Responses")
        PSS = [st.number_input(f"PSS{i+1}", 0, 4, 1) for i in range(10)]

        st.markdown("### 🟩 GAD (Anxiety) Responses")
        GAD = [st.number_input(f"GAD{i+1}", 0, 3, 1) for i in range(7)]

        st.markdown("### 🟥 PHQ (Depression) Responses")
        PHQ = [st.number_input(f"PHQ{i+1}", 0, 3, 1) for i in range(9)]

    submitted = st.form_submit_button("🔍 Run ML Assessment")

# ---------------------------------------------------------
# RUN ML MODELS
# ---------------------------------------------------------

if submitted:
    st.markdown("---")
    st.subheader("🔎 ML Prediction Results")

    # create dictionary for ML input
    data = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship
    }

    # add PSS
    for i in range(10):
        data[f"PSS{i+1}"] = PSS[i]

    # add GAD
    for i in range(7):
        data[f"GAD{i+1}"] = GAD[i]

    # add PHQ
    for i in range(9):
        data[f"PHQ{i+1}"] = PHQ[i]

    # run unified ML pipeline
    anx, str_, dep, main_issue = predict_for_student(data)

    # show prediction
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Anxiety Prediction", "Present" if anx == 1 else "Absent")

    with colB:
        st.metric("Stress Prediction", "Present" if str_ == 1 else "Absent")

    with colC:
        st.metric("Depression Prediction", "Present" if dep == 1 else "Absent")

    st.markdown("### 🧠 Overall Mental Health Status:")
    st.success(f"**{main_issue}**")

    st.markdown("---")

    # ---------------------------------------------------------
    # BASIC XAI (Explainability)
    # ---------------------------------------------------------
    st.subheader("📘 Explainable AI (XAI): Key Influential Features")

    st.write("""
    These features (from Logistic Regression coefficients) indicate which questionnaire items 
    and numeric factors contribute most to **Anxiety**, **Stress**, and **Depression** predictions.
    """)

    import pandas as pd
    from unified_mental_health_pipeline import (
        anx_clf_num, str_clf_num, dep_clf_num
    )

    # Top Features Extraction
    def get_top_features(model, cols, top_k=8):
        coefs = model.coef_[0]
        df_feat = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
        df_feat["Abs"] = df_feat["Coefficient"].abs()
        return df_feat.sort_values("Abs", ascending=False).head(top_k)

    top_anx = get_top_features(anx_clf_num, x_numeric.columns)
    top_str = get_top_features(str_clf_num, x_numeric.columns)
    top_dep = get_top_features(dep_clf_num, x_numeric.columns)

    colX, colY, colZ = st.columns(3)

    with colX:
        st.write("### 🔵 Anxiety Top Features")
        st.dataframe(top_anx[["Feature", "Coefficient"]])

    with colY:
        st.write("### 🟡 Stress Top Features")
        st.dataframe(top_str[["Feature", "Coefficient"]])

    with colZ:
        st.write("### 🔴 Depression Top Features")
        st.dataframe(top_dep[["Feature", "Coefficient"]])

    st.markdown("---")
    st.info("This ML-based assessment is not a clinical diagnosis. It identifies early risk patterns to help understand mental-health conditions.")
