# ==============================
# app.py  –  Unified Mental Health ML App
# ==============================

import streamlit as st
import pandas as pd
import datetime
import os

from predict_all import predict_all   # our unified prediction helper

# ---------------------------
# Basic page config
# ---------------------------
st.set_page_config(
    page_title="AI Mental Health Assessment",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Page header
# ---------------------------
st.title("🧠 AI-Powered Mental Health Assessment System")

st.markdown(
    """
This web app uses **Machine Learning models** trained on **PSS-10, GAD-7, PHQ-9** questionnaire data  
to predict three key mental-health conditions among students:

- 😰 **Anxiety**  
- 😓 **Stress**  
- 😞 **Depression**

> ⚠️ This is an experimental research tool, **not a medical diagnosis**.
"""
)

# ---------------------------
# Sidebar – Demographic / academic info
# ---------------------------
st.sidebar.header("📌 Student Information")

age = st.sidebar.selectbox("Age range", ["18-22", "23-27", "28-32", "33+"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

university = st.sidebar.text_input("University")
department = st.sidebar.text_input("Department")

academic_year = st.sidebar.selectbox(
    "Academic Year / Semester",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)

cgpa = st.sidebar.selectbox(
    "Current CGPA",
    ["<2.50", "2.50-3.00", "3.00-3.50", "3.50-4.00"]
)

scholarship = st.sidebar.selectbox(
    "Scholarship / Waiver",
    ["No", "Yes - Partial", "Yes - Full"]
)

# ---------------------------
# Tabs for PSS / GAD / PHQ
# ---------------------------
tab_pss, tab_gad, tab_phq = st.tabs(
    ["🟨 PSS-10 (Stress)", "🟦 GAD-7 (Anxiety)", "🟥 PHQ-9 (Depression)"]
)

# ---------- PSS-10 (0–4) ----------
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

with tab_pss:
    st.subheader("Perceived Stress Scale (PSS-10)")
    st.caption("Scale: 0 = Never, 1 = Almost never, 2 = Sometimes, 3 = Fairly often, 4 = Very often")

    pss = {}
    for i, q in enumerate(pss_questions, start=1):
        pss[f"PSS{i}"] = st.slider(q, min_value=0, max_value=4, value=0)

# ---------- GAD-7 (0–3) ----------
gad_questions = [
    "Feeling nervous, anxious, or on edge because of academic pressure?",
    "Not being able to stop or control worrying about academic issues?",
    "Worrying too much about different university-related things?",
    "Trouble relaxing due to academic stress?",
    "Being so restless that it's hard to sit still when thinking about studies?",
    "Becoming easily annoyed or irritable because of academic workload?",
    "Feeling afraid as if something awful might happen academically?"
]

with tab_gad:
    st.subheader("Generalized Anxiety Disorder (GAD-7)")
    st.caption("Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")

    gad = {}
    for i, q in enumerate(gad_questions, start=1):
        gad[f"GAD{i}"] = st.slider(q, min_value=0, max_value=3, value=0)

# ---------- PHQ-9 (0–3) ----------
phq_questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself or that you are a failure?",
    "Trouble concentrating on reading, studies, or watching something?",
    "Moving or speaking so slowly that others noticed – or the opposite (restless, fidgety)?",
    "Thoughts that you would be better off dead, or of hurting yourself in some way?"
]

with tab_phq:
    st.subheader("Patient Health Questionnaire (PHQ-9)")
    st.caption("Scale: 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")

    phq = {}
    for i, q in enumerate(phq_questions, start=1):
        phq[f"PHQ{i}"] = st.slider(q, min_value=0, max_value=3, value=0)

st.markdown("---")

# ---------------------------
# Predict button
# ---------------------------
if st.button("🔍 Run AI Prediction"):

    # Build input dictionary in EXACT same format as training features
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

    # Run model prediction (3 conditions)
    try:
        result = predict_all(user_input)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # --------- Pretty result display ---------
    st.subheader("📊 Prediction Results")

    def colored_box(text, color):
        st.markdown(
            f"""
            <div style="background-color:{color};
                        padding:15px;
                        border-radius:10px;
                        margin:5px 0;
                        font-size:18px;">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

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

    # ---------------------------
    # Save log to CSV
    # ---------------------------
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

    # Download this single prediction
    st.download_button(
        label="⬇ Download This Prediction (CSV)",
        data=log_df.to_csv(index=False),
        file_name="mh_prediction_result.csv",
        mime="text/csv"
    )

    st.info(
        "These predictions are based on machine-learning patterns from student data. "
        "For any serious concern, please consult a mental health professional."
    )
