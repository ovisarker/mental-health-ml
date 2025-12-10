import streamlit as st
import pandas as pd

# Import ML functions from your unified pipeline
from unified_mental_health_pipeline import (
    predict_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num,
)

# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(page_title="ML Student Mental Health Assessment", layout="wide")


# ----------------------------------------------------------
# RISK LEVEL CALCULATION FUNCTIONS
# ----------------------------------------------------------
def compute_risk_levels(PSS, GAD, PHQ):
    pss_total = sum(PSS)     # 0–40
    gad_total = sum(GAD)     # 0–28 (your dataset uses 0–4 per item)
    phq_total = sum(PHQ)     # 0–36 (your dataset uses 0–4 per item)

    # Stress (PSS-10)
    if pss_total <= 13:
        stress_level = "Low"
    elif pss_total <= 26:
        stress_level = "Moderate"
    else:
        stress_level = "High"

    # Anxiety (GAD-7 scaled)
    if gad_total <= 6:
        anxiety_level = "Minimal"
    elif gad_total <= 12:
        anxiety_level = "Mild"
    elif gad_total <= 19:
        anxiety_level = "Moderate"
    else:
        anxiety_level = "Severe"

    # Depression (PHQ-9 scaled)
    if phq_total <= 6:
        depression_level = "Minimal"
    elif phq_total <= 13:
        depression_level = "Mild"
    elif phq_total <= 19:
        depression_level = "Moderate"
    elif phq_total <= 25:
        depression_level = "Moderately Severe"
    else:
        depression_level = "Severe"

    return (stress_level, anxiety_level, depression_level,
            pss_total, gad_total, phq_total)


# ----------------------------------------------------------
# SUGGESTIONS BASED ON RISK LEVEL
# ----------------------------------------------------------
def get_suggestions(stress_level, anxiety_level, depression_level):
    suggestions = []

    if stress_level in ["Moderate", "High"]:
        suggestions.append("• Break academic tasks into smaller parts to reduce overload.")
        suggestions.append("• Practice deep breathing during high-pressure moments.")

    if anxiety_level in ["Moderate", "Severe"]:
        suggestions.append("• Limit caffeine and maintain a regular sleep schedule.")
        suggestions.append("• Try grounding exercises when worrying becomes intense.")

    if depression_level in ["Moderate", "Moderately Severe", "Severe"]:
        suggestions.append("• Try maintaining a daily routine with light physical activity.")
        suggestions.append("• Reach out to a trusted person or counselor when overwhelmed.")

    if not suggestions:
        suggestions.append("• No major risks detected. Maintain healthy lifestyle habits.")

    return suggestions


# ----------------------------------------------------------
# BUILD STUDENT DATA DICTIONARY FOR ML
# ----------------------------------------------------------
def build_student_dict(age, gender, university, department, year, cgpa, scholarship,
                       PSS, GAD, PHQ):
    data = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship,
    }

    for i in range(10):
        data[f"PSS{i+1}"] = PSS[i]
    for i in range(7):
        data[f"GAD{i+1}"] = GAD[i]
    for i in range(9):
        data[f"PHQ{i+1}"] = PHQ[i]

    return data


# ----------------------------------------------------------
# XAI Feature Importance Helper
# ----------------------------------------------------------
def get_top_features(model, cols, top_k=8):
    coefs = model.coef_[0]
    df = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
    df["Abs"] = df["Coefficient"].abs()
    return df.sort_values("Abs", ascending=False).head(top_k)


# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------
def main():
    st.title("🧠 ML-Based Student Mental Health Assessment System")
    st.write(
        "This system predicts **Anxiety, Stress, Depression**, assigns **risk levels**, "
        "identifies the **dominant issue**, and provides **suggestions & explainability**."
    )
    st.info("⚠️ This is a research tool, not a medical diagnostic system.")

    st.markdown("---")

    # ------------------------------------------------------
    # INPUT SECTION
    # ------------------------------------------------------
    st.subheader("📋 Student Information & Questionnaires")

    with st.form("assessment_form"):

        # Basic student info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎓 Student Details")
            age = st.number_input("Age", 16, 40, 20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            university = st.text_input("University")
            department = st.text_input("Department")
            year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
            cgpa = st.number_input("CGPA", 0.0, 4.0, 3.0)
            scholarship = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

        # Stress section (PSS-10)
        st.markdown("### 🟦 Stress Assessment (PSS-10)")
        st.caption("Scale: 0 = Never • 1 = Almost Never • 2 = Sometimes • 3 = Fairly Often • 4 = Very Often")
        PSS_q = [
            "Upset due to academic issues",
            "Unable to control important academic things",
            "Felt nervous/stressed due to academics",
            "Couldn't cope with assignments/exams",
            "Felt confident handling problems (Reverse)",
            "Things going your way academically (Reverse)",
            "Controlled irritation from academics (Reverse)",
            "Felt academic performance was satisfactory (Reverse)",
            "Felt anger from poor academic outcome",
            "Academic difficulties piled up beyond control",
        ]
        PSS = [st.slider(f"PSS{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PSS_q)]

        # Anxiety section (GAD-7)
        st.markdown("### 🟩 Anxiety Assessment (GAD-7)")
        GAD_q = [
            "Nervous or on edge",
            "Unable to stop worrying",
            "Trouble relaxing",
            "Easily annoyed/irritated",
            "Worrying too much",
            "Restlessness",
            "Feeling something bad might happen",
        ]
        GAD = [st.slider(f"GAD{i+1}: {q}", 0, 4, 1) for i, q in enumerate(GAD_q)]

        # Depression section (PHQ-9)
        st.markdown("### 🟥 Depression Assessment (PHQ-9)")
        PHQ_q = [
            "Little interest or pleasure",
            "Feeling down/hopeless",
            "Sleeping issues",
            "Feeling tired or low energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Moving/speaking slowly/fast",
            "Thoughts of self-harm (⚠ Serious)",
        ]
        PHQ = [st.slider(f"PHQ{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PHQ_q)]

        submitted = st.form_submit_button("🔍 Run Assessment")

    # ------------------------------------------------------
    # RUN ML + SHOW RESULTS
    # ------------------------------------------------------
    if submitted:

        # Compute risk level from score-based scales
        stress_level, anxiety_level, depression_level, pss_total, gad_total, phq_total = compute_risk_levels(
            PSS, GAD, PHQ
        )

        # Build dict and run ML prediction
        student_data = build_student_dict(
            age, gender, university, department, year, cgpa, scholarship,
            PSS, GAD, PHQ
        )

        anx_pred, str_pred, dep_pred, main_status = predict_for_student(student_data)

        # XAI tables
        top_anx = get_top_features(anx_clf_num, x_numeric.columns)
        top_str = get_top_features(str_clf_num, x_numeric.columns)
        top_dep = get_top_features(dep_clf_num, x_numeric.columns)

        # ------------------------------------------------------
        # SIMPLE RESULT VIEW
        # ------------------------------------------------------
        st.markdown("## ✅ Results Summary")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Anxiety (ML)", "Present" if anx_pred == 1 else "Absent")
        with c2:
            st.metric("Stress (ML)", "Present" if str_pred == 1 else "Absent")
        with c3:
            st.metric("Depression (ML)", "Present" if dep_pred == 1 else "Absent")

        st.markdown(f"### 🧠 Dominant Mental-Health Issue: **{main_status}**")

        # ------------------------------------------------------
        # RISK LEVELS
        # ------------------------------------------------------
        st.markdown("## 🎯 Risk Levels (Score-based)")
        st.write(f"**Stress:** {stress_level} (Score {pss_total}/40)")
        st.write(f"**Anxiety:** {anxiety_level} (Score {gad_total}/28)")
        st.write(f"**Depression:** {depression_level} (Score {phq_total}/36)")

        # ------------------------------------------------------
        # SUGGESTIONS
        # ------------------------------------------------------
        st.markdown("## 💡 Suggestions")
        for s in get_suggestions(stress_level, anxiety_level, depression_level):
            st.write(s)

        # ------------------------------------------------------
        # EMERGENCY HELP
        # ------------------------------------------------------
        st.markdown("## 🚨 Emergency Support")
        if PHQ[8] >= 3:  # suicide risk
            st.error("⚠️ Severe depression pattern detected with self-harm indication. Seek help immediately.")
        st.write("🇧🇩 Bangladesh Hotline: **09612-119911 (Kaan Pete Roi)**")
        st.write("🇺🇸 USA Emergency: **988 (Suicide & Crisis Lifeline)**")

        # ------------------------------------------------------
        # XAI TABLES
        # ------------------------------------------------------
        st.markdown("## 🔬 Explainable AI (Top Influential Features)")

        colA, colB, colC = st.columns(3)
        with colA:
            st.write("### Anxiety – Top Features")
            st.dataframe(top_anx)

        with colB:
            st.write("### Stress – Top Features")
            st.dataframe(top_str)

        with colC:
            st.write("### Depression – Top Features")
            st.dataframe(top_dep)


if __name__ == "__main__":
    main()
