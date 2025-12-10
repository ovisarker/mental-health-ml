import streamlit as st
import pandas as pd

# Import from your unified ML pipeline
from unified_mental_health_pipeline import (
    predict_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num,
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML-Based Student Mental Health Assessment",
    layout="wide"
)


# ---------------------------------------------------------
# SMALL UTILS
# ---------------------------------------------------------
def get_top_features(model, cols, top_k=8):
    """Return top_k features by absolute coefficient from a numeric LR model."""
    coefs = model.coef_[0]
    df_feat = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
    df_feat["Abs"] = df_feat["Coefficient"].abs()
    return df_feat.sort_values("Abs", ascending=False).head(top_k)


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


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.title("🧠 ML-Based Student Mental Health Assessment System")
    st.write(
        """
This system uses **Machine Learning** to predict:
- Anxiety (Present / Absent)  
- Stress (Present / Absent)  
- Depression (Present / Absent)  

Then it gives an **Overall Mental-Health Status** and basic **Explainable AI** insights.
        """
    )
    st.info(
        "⚠️ This tool is for **screening & research** only. It does **not** provide "
        "a clinical diagnosis or treatment."
    )
    st.markdown("---")

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------
    st.subheader("📋 Student Profile & Questionnaire")

    with st.form("mh_form"):
        col1, col2 = st.columns(2)

        # DEMOGRAPHIC / ACADEMIC
        with col1:
            st.markdown("### 👤 Demographic & Academic Info")
            age = st.number_input("Age", 16, 40, 20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            university = st.text_input("University", "")
            department = st.text_input("Department", "")
            year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
            cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
            scholarship = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

            st.markdown("#### Scale for All Questions")
            st.info(
                "0 = Never • 1 = Almost Never • 2 = Sometimes • "
                "3 = Fairly Often • 4 = Very Often"
            )

        # QUESTIONNAIRES
        with col2:
            # PSS-10
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
                "How often did academic difficulties pile up beyond control?",
            ]
            PSS = []
            for i, q in enumerate(PSS_questions):
                PSS.append(st.number_input(f"PSS{i+1}: {q}", 0, 4, 1))

            # GAD-7
            st.markdown("### 🟩 GAD-7 (Anxiety – Academic Context)")
            GAD_questions = [
                "How often did you feel nervous or on edge due to academic pressure?",
                "How often were you unable to stop worrying about academic issues?",
                "How often did academic pressure stop you from relaxing?",
                "How often were you easily annoyed or irritated due to academics?",
                "How often did you worry too much about academic matters?",
                "How often did restlessness make it hard to sit still due to stress?",
                "How often did you feel afraid that something bad might happen academically?",
            ]
            GAD = []
            for i, q in enumerate(GAD_questions):
                GAD.append(st.number_input(f"GAD{i+1}: {q}", 0, 4, 1))

            # PHQ-9
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
                "Thoughts of harming yourself or being better off dead?",
            ]
            PHQ = []
            for i, q in enumerate(PHQ_questions):
                PHQ.append(st.number_input(f"PHQ{i+1}: {q}", 0, 4, 1))

        submitted = st.form_submit_button("🔍 Run ML Assessment")

    # -----------------------------------------------------
    # VALIDATION & ML PREDICTION
    # -----------------------------------------------------
    anx_pred = str_pred = dep_pred = None
    main_status = None
    top_anx = top_str = top_dep = None

    if submitted:
        # Basic sanity warnings
        total_pss = sum(PSS)
        total_gad = sum(GAD)
        total_phq = sum(PHQ)

        if total_pss == 40 or total_gad == 28 or total_phq == 36:
            st.warning(
                "⚠️ All answers are at the maximum (4). This is an extreme pattern "
                "and may not represent typical student responses."
            )

        if university.strip() == "" or department.strip() == "":
            st.warning("⚠️ University and Department are empty. Please fill them if possible.")

        # Build student dict and run ML
        student_data = build_student_dict(
            age, gender, university, department, year, cgpa, scholarship,
            PSS, GAD, PHQ
        )

        try:
            anx_pred, str_pred, dep_pred, main_status = predict_for_student(student_data)
        except Exception as e:
            st.error(f"Error during ML prediction: {e}")
            return

        # Prepare XAI top features (numeric LR models)
        try:
            top_anx = get_top_features(anx_clf_num, x_numeric.columns)
            top_str = get_top_features(str_clf_num, x_numeric.columns)
            top_dep = get_top_features(dep_clf_num, x_numeric.columns)
        except Exception as e:
            st.warning(f"XAI feature importance could not be computed: {e}")

    # -----------------------------------------------------
    # TABS: SIMPLE VIEW / ADVANCED ML VIEW / ABOUT
    # -----------------------------------------------------
    tab_simple, tab_advanced, tab_about = st.tabs(
        ["✅ Simple View", "🔬 Advanced ML View", "ℹ️ About System"]
    )

    # SIMPLE VIEW
    with tab_simple:
        st.subheader("✅ Simple Result View")

        if not submitted:
            st.info("Submit the form to see predictions.")
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Anxiety", "Present" if anx_pred == 1 else "Absent")
            with colB:
                st.metric("Stress", "Present" if str_pred == 1 else "Absent")
            with colC:
                st.metric("Depression", "Present" if dep_pred == 1 else "Absent")

            st.markdown("### 🧠 Overall Mental-Health Status")
            if main_status:
                st.success(f"**{main_status}**")
            else:
                st.info("Overall status not available.")

            # Short natural-language summary
            st.markdown("### 📝 Summary (Human-friendly Explanation)")
            summary_lines = []
            if anx_pred == 1:
                summary_lines.append("- Signs of **anxiety** risk are present.")
            else:
                summary_lines.append("- No strong pattern of **anxiety** risk detected.")

            if str_pred == 1:
                summary_lines.append("- Signs of **stress** risk are present.")
            else:
                summary_lines.append("- No strong pattern of **stress** risk detected.")

            if dep_pred == 1:
                summary_lines.append("- Signs of **depression** risk are present.")
            else:
                summary_lines.append("- No strong pattern of **depression** risk detected.")

            if main_status:
                summary_lines.append(f"- Overall, the model identifies: **{main_status}**.")

            st.write("\n".join(summary_lines))

    # ADVANCED ML VIEW
    with tab_advanced:
        st.subheader("🔬 Advanced ML View (For Internal / Viva)")

        if not submitted:
            st.info("Run the assessment first to see ML details.")
        else:
            st.markdown("#### 1️⃣ Raw Predictions (0/1 flags)")
            st.write(
                f"- Anxiety flag: `{anx_pred}`  (1 = Present, 0 = Absent)\n"
                f"- Stress flag: `{str_pred}`\n"
                f"- Depression flag: `{dep_pred}`"
            )

            st.markdown("#### 2️⃣ Explainable AI – Top Numeric Features")
            st.write(
                "These tables show the most influential **numeric** features based on "
                "logistic regression coefficients from numeric-only models."
            )

            if top_anx is not None and top_str is not None and top_dep is not None:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("##### 🔵 Anxiety – Top Features")
                    st.dataframe(top_anx[["Feature", "Coefficient"]])
                with c2:
                    st.write("##### 🟡 Stress – Top Features")
                    st.dataframe(top_str[["Feature", "Coefficient"]])
                with c3:
                    st.write("##### 🔴 Depression – Top Features")
                    st.dataframe(top_dep[["Feature", "Coefficient"]])
            else:
                st.info("XAI tables not available.")

            st.markdown("#### 3️⃣ Notes on Model Design (You can say this in viva)")
            st.markdown(
                """
- We use **Logistic Regression** with `class_weight='balanced'` to handle class imbalance.  
- **Preprocessing:** Categorical features → OneHotEncoder, numeric features → StandardScaler.  
- Dataset split: **80% train, 20% test** (or similar, depending on version).  
- We trained **three independent binary classifiers** for Anxiety, Stress, and Depression.  
- Then we derived an **Overall Mental-Health Status** (dominant condition) from the three outputs.  
- For explainability, we trained **numeric-only LR models** and used coefficients as feature importance.
                """
            )

    # ABOUT TAB
    with tab_about:
        st.subheader("ℹ️ About This System")
        st.write(
            """
This application is part of a **thesis research project** on student mental health:

- **Goal:** Early detection of anxiety, stress, and depression risk patterns among university students.  
- **Data:** Demographic + academic info + validated questionnaires (PSS-10, GAD-7, PHQ-9).  
- **Methods:** Supervised machine learning (Logistic Regression) with proper preprocessing pipelines.  
- **Output:** Binary risk predictions + overall dominant issue + basic model explainability.

It is **not a clinical tool**, but a **screening and research prototype** to support mental-health awareness
and future intervention design.
            """
        )


if __name__ == "__main__":
    main()
