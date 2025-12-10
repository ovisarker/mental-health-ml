# =====================================
# app.py — Professional Streamlit App
# Unified Mental-Health Prediction System
# =====================================

import streamlit as st
import pandas as pd
import datetime
from predict_all import predict_all  # Import from your prediction script
import os

st.set_page_config(
    page_title="AI Mental Health Assessment",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------
# Header Section
# ------------------------------
st.title("🧠 AI-Powered Mental Health Assessment System")
st.markdown("""
This tool predicts **Anxiety, Stress, and Depression** levels using Machine Learning models trained on
validated psychometric scales (PSS-10, GAD-7, PHQ-9).  
""")

st.info("Fill in your information and questionnaire responses to receive AI-assisted mental health screening.")

# ------------------------------
# Sidebar – User Info
# ------------------------------
st.sidebar.header("📌 User Information")

age = st.sidebar.selectbox("Age", ["18-22", "23-27", "28-32", "33+"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
university = st.sidebar.text_input("University Name")
department = st.sidebar.text_input("Department")
academic_year = st.sidebar.selectbox("Academic Year",
                                     ["1st Year", "2nd Year", "3rd Year", "4th Year"])
cgpa = st.sidebar.selectbox("Current CGPA", ["<2.50", "2.50-3.00", "3.00-3.50", "3.50-4.00"])
scholarship = st.sidebar.selectbox("Scholarship/Waiver",
                                   ["No", "Yes - Partial", "Yes - Full"])

# ------------------------------
# Questionnaire Inputs (Tabs)
# ------------------------------
tab1, tab2, tab3 = st.tabs(["🟨 PSS-10 (Stress)", "🟦 GAD-7 (Anxiety)", "🟥 PHQ-9 (Depression)"])

# ------------------------------
# PSS-10 Inputs
# ------------------------------
with tab1:
    st.subheader("Perceived Stress Scale (PSS-10)")
    pss = {}
    for i in range(1, 11):
        pss[f"PSS{i}"] = st.slider(f"PSS{i}", 0, 4, 0)

# ------------------------------
# GAD-7 Inputs
# ------------------------------
with tab2:
    st.subheader("Generalized Anxiety Disorder (GAD-7)")
    gad = {}
    for i in range(1, 8):
        gad[f"GAD{i}"] = st.slider(f"GAD{i}", 0, 3, 0)

# ------------------------------
# PHQ-9 Inputs
# ------------------------------
with tab3:
    st.subheader("Patient Health Questionnaire (PHQ-9)")
    phq = {}
    for i in range(1, 10):
        phq[f"PHQ{i}"] = st.slider(f"PHQ{i}", 0, 3, 0)

# ------------------------------
# Predict Button
# ------------------------------
st.markdown("---")
if st.button("🔍 Run AI Prediction"):
    
    # Build user dictionary in correct format
    user_input = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": academic_year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship
    }

    # Add PSS, GAD, PHQ data
    user_input.update(pss)
    user_input.update(gad)
    user_input.update(phq)

    # Run prediction
    result = predict_all(user_input)

    # Color box style
    def colored_box(text, color):
        st.markdown(f"""
        <div style="background-color:{color};padding:15px;border-radius:10px;margin:5px 0;font-size:18px;">
            {text}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 AI Predictions")

    if "High" in result["Anxiety"]:
        colored_box(f"😰 Anxiety: **{result['Anxiety']}**", "#ffcccc")
    else:
        colored_box(f"😌 Anxiety: **{result['Anxiety']}**", "#d4ffd4")

    if "High" in result["Stress"]:
        colored_box(f"😓 Stress: **{result['Stress']}**", "#ffebcc")
    else:
        colored_box(f"😌 Stress: **{result['Stress']}**", "#d4ffd4")

    if "Present" in result["Depression"]:
        colored_box(f"😞 Depression: **{result['Depression']}**", "#ffd6d6")
    else:
        colored_box(f"🙂 Depression: **{result['Depression']}**", "#d4ffd4")

    # ------------------------------
    # Save Log (CSV)
    # ------------------------------
    log_entry = user_input.copy()
    log_entry.update(result)
    log_entry["Timestamp"] = str(datetime.datetime.now())

    log_df = pd.DataFrame([log_entry])

    if not os.path.exists("prediction_logs.csv"):
        log_df.to_csv("prediction_logs.csv", index=False)
    else:
        log_df.to_csv("prediction_logs.csv", index=False, mode="a", header=False)

    st.success("Prediction saved to log file successfully!")

    st.download_button(
        label="⬇ Download Prediction Result",
        data=log_df.to_csv(index=False),
        file_name="mh_prediction_result.csv",
        mime="text/csv"
    )
