# =====================================
# app.py — Professional Streamlit App
# Unified Mental-Health Prediction System
# =====================================

import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

from predict_all import predict_all, anxiety_model, stress_model, depression_model

st.set_page_config(
    page_title="AI Mental Health Assessment",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 AI-Powered Mental Health Assessment System")
st.markdown("""
This tool predicts **Anxiety, Stress, and Depression** using Machine Learning models  
trained on validated psychometric scales (**PSS-10, GAD-7, PHQ-9**).
""")
st.info("Fill in your information and questionnaire responses to receive AI-assisted screening. This is **not** a clinical diagnosis tool.")

# ------------------------------
# Sidebar – User Info
# ------------------------------
st.sidebar.header("📌 Basic Information")

age = st.sidebar.selectbox("Age", ["18-22", "23-27", "28-32", "33+"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
university = st.sidebar.text_input("University Name")
department = st.sidebar.text_input("Department")
academic_year = st.sidebar.selectbox(
    "Academic Year",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)
cgpa = st.sidebar.selectbox(
    "Current CGPA",
    ["<2.50", "2.50-3.00", "3.00-3.50", "3.50-4.00"]
)
scholarship = st.sidebar.selectbox(
    "Scholarship/Waiver",
    ["No", "Yes - Partial", "Yes - Full"]
)

# ------------------------------
# Questionnaire Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["🟨 PSS-10 (Stress)", "🟦 GAD-7 (Anxiety)", "🟥 PHQ-9 (Depression)"])

# --- PSS-10 (0–4) ---
pss_questions = [
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

with tab1:
    st.subheader("Perceived Stress Scale (PSS-10)")
    st.caption("Scale: 0 = Never, 1 = Almost Never, 2 = Sometimes, 3 = Fairly Often, 4 = Very Often")
    pss = {}
    for i, q in enumerate(pss_questions, start=1):
        pss[f"PSS{i}"] = st.slider(q, 0, 4, 0)

# --- GAD-7 (0–3) ---
gad_questions = [
    "Feeling nervous, anxious, or on edge because of academic pressure?",
    "Not being able to stop or control worrying about academic issues?",
    "Worrying too much about different university-related things?",
    "Trouble relaxing due to academic stress?",
    "Being so restless that it's hard to sit still when thinking about studies?",
    "Becoming easily annoyed or irritable because of academic workload?",
    "Feeling afraid as if something awful might happen academically?"
]

with tab2:
    st.subheader("Generalized Anxiety Disorder (GAD-7)")
    st.caption("Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")
    gad = {}
    for i, q in enumerate(gad_questions, start=1):
        gad[f"GAD{i}"] = st.slider(q, 0, 3, 0)

# --- PHQ-9 (0–3) ---
phq_questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself or that you are a failure?",
    "Trouble concentrating on reading, studies, or watching something?",
    "Moving or speaking so slowly that others noticed — or the opposite, being fidgety or restless?",
    "Thoughts that you would be better off dead, or of hurting yourself in some way?"
]

with tab3:
    st.subheader("Patient Health Questionnaire (PHQ-9)")
    st.caption("Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")
    phq = {}
    for i, q in enumerate(phq_questions, start=1):
        phq[f"PHQ{i}"] = st.slider(q, 0, 3, 0)

st.markdown("---")

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔍 Run AI Prediction"):

    # Build user dict in correct format for model
    user_input = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": academic_year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship
    }
    user_input.update(pss)
    user_input.update(gad)
    user_input.update(phq)

    # Run unified prediction
    result = predict_all(user_input)

    # Color helper
    def colored_box(text, color):
        st.markdown(
            f"""
            <div style="background-color:{color};padding:15px;border-radius:10px;
                        margin:5px 0;font-size:18px;">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📊 AI Prediction Results")

    if "High" in result["Anxiety"]:
        colored_box(f"😰 Anxiety: <b>{result['Anxiety']}</b>", "#ffcccc")
    else:
        colored_box(f"😌 Anxiety: <b>{result['Anxiety']}</b>", "#d4ffd4")

    if "High" in result["Stress"]:
        colored_box(f"😓 Stress: <b>{result['Stress']}</b>", "#ffebcc")
    else:
        colored_box(f"😌 Stress: <b>{result['Stress']}</b>", "#d4ffd4")

    if "Present" in result["Depression"]:
        colored_box(f"😞 Depression: <b>{result['Depression']}</b>", "#ffd6d6")
    else:
        colored_box(f"🙂 Depression: <b>{result['Depression']}</b>", "#d4ffd4")

    # ------------------------------
    # Save Log
    # ------------------------------
    log_entry = user_input.copy()
    log_entry.update(result)
    log_entry["Timestamp"] = str(datetime.datetime.now())

    log_df = pd.DataFrame([log_entry])

    if not os.path.exists("prediction_logs.csv"):
        log_df.to_csv("prediction_logs.csv", index=False)
    else:
        log_df.to_csv("prediction_logs.csv", index=False, mode="a", header=False)

    st.success("✅ Prediction saved to log file successfully.")

    st.download_button(
        label="⬇ Download This Result (CSV)",
        data=log_df.to_csv(index=False),
        file_name="mh_prediction_result.csv",
        mime="text/csv"
    )

    # ==========================================
    # 🔍 Basic SHAP Explainability (Depression)
    # ==========================================
    st.markdown("---")
    st.markdown("## 🧩 Explainability – Which factors affect Depression prediction?")

    try:
        # Load background data from Processed.csv
        bg_df = pd.read_csv("Processed.csv")
        bg_df.columns = bg_df.columns.str.strip()

        # Drop leakage/label cols same as training
        drop_cols = [
            "Stress Value", "Stress Label",
            "Anxiety Value", "Anxiety Label",
            "Depression Value", "Depression Label",
            "GAD_Total", "PSS_Total",
            "Depression_Binary", "Anxiety_Binary", "Stress_Binary"
        ]
        drop_cols = list(set(drop_cols).intersection(set(bg_df.columns)))
        X_bg = bg_df.drop(columns=drop_cols)

        # Use a small sample for speed
        X_bg_sample = X_bg.sample(min(200, len(X_bg)), random_state=42)

        # Get preprocessor & classifier from pipeline
        preprocessor = depression_model.named_steps["preprocess"]
        clf_dep = depression_model.named_steps["classifier"]

        # Transform background data
        X_bg_trans = preprocessor.transform(X_bg_sample)

        # Get feature names after preprocessing
        cat_cols = preprocessor.transformers_[0][2]
        num_cols = preprocessor.transformers_[1][2]
        ohe = preprocessor.transformers_[0][1]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        all_feature_names = np.concatenate([cat_feature_names, num_cols])

        # Build SHAP Explainer (tree-based)
        explainer_dep = shap.TreeExplainer(clf_dep)
        shap_values = explainer_dep.shap_values(X_bg_trans)

        st.caption("Global SHAP summary: features with higher absolute SHAP values have stronger influence on Depression risk.")

        fig, ax = plt.subplots(figsize=(8, 5))
        # For binary classifier, use class 1 SHAP values
        shap.summary_plot(
            shap_values[1] if isinstance(shap_values, list) else shap_values,
            X_bg_trans,
            feature_names=all_feature_names,
            show=False
        )
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")
        st.caption("Make sure 'Processed.csv' and the trained depression model are available.")
