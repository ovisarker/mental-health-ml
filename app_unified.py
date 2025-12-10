import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1) DATA LOAD + MODEL TRAIN
# =========================
@st.cache_resource
def train_models():
    # ---- Load dataset ----
    df = pd.read_csv("Updated Processed Diu.csv")

    # ---- Create labels ----
    # Depression_Binary from "Depression Label"
    if "Depression_Binary" not in df.columns:
        df["Depression_Binary"] = df["Depression Label"].apply(
            lambda x: 0 if str(x).strip().lower() == "no depression" else 1
        )

    # Anxiety_Binary from GAD1–GAD7
    gad_cols = [c for c in df.columns if c.startswith("GAD")]
    df["GAD_Total"] = df[gad_cols].sum(axis=1)
    df["Anxiety_Binary"] = df["GAD_Total"].apply(lambda v: 0 if v <= 4 else 1)

    # Stress_Binary from PSS1–PSS10
    pss_cols = [c for c in df.columns if c.startswith("PSS")]
    df["PSS_Total"] = df[pss_cols].sum(axis=1)
    df["Stress_Binary"] = df["PSS_Total"].apply(lambda v: 0 if v <= 13 else 1)

    # ---- Drop target-related columns from features ----
    drop_cols = [
        "Depression Label",
        "Depression Value",
        "Depression_Binary",
        "Anxiety_Binary",
        "Stress_Binary",
        "GAD_Total",
        "PSS_Total",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y_anx = df["Anxiety_Binary"]
    y_str = df["Stress_Binary"]
    y_dep = df["Depression_Binary"]

    # ---- Categorical / Numeric detection ----
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # ---- Preprocessor ----
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    # ---- Base classifier (same for all 3) ----
    def make_clf():
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear"
        )

    # One common split for all models
    X_train, X_test, y_anx_train, y_anx_test = train_test_split(
        X, y_anx, test_size=0.2, random_state=42, stratify=y_anx
    )
    _, _, y_str_train, y_str_test = train_test_split(
        X, y_str, test_size=0.2, random_state=42, stratify=y_str
    )
    _, _, y_dep_train, y_dep_test = train_test_split(
        X, y_dep, test_size=0.2, random_state=42, stratify=y_dep
    )

    # ---- Build 3 pipelines ----
    anxiety_model = Pipeline(
        steps=[("preprocess", preprocessor), ("clf", make_clf())]
    )
    stress_model = Pipeline(
        steps=[("preprocess", preprocessor), ("clf", make_clf())]
    )
    depression_model = Pipeline(
        steps=[("preprocess", preprocessor), ("clf", make_clf())]
    )

    # ---- Train models ----
    anxiety_model.fit(X_train, y_anx_train)
    stress_model.fit(X_train, y_str_train)
    depression_model.fit(X_train, y_dep_train)

    # ---- Quick metrics (console print) ----
    for name, model, yt in [
        ("Anxiety", anxiety_model, y_anx_test),
        ("Stress", stress_model, y_str_test),
        ("Depression", depression_model, y_dep_test),
    ]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(yt, y_pred)
        print(f"{name} Accuracy: {acc:.3f}")
        print(classification_report(yt, y_pred, digits=3))

    return anxiety_model, stress_model, depression_model, X.columns.tolist()


# =========================
# 2) QUESTION TEXT + MAPPING
# =========================

PSS_QUESTIONS = {
    "PSS1": "In a semester, how often have you felt upset due to something that happened in your academic affairs?",
    "PSS2": "In a semester, how often you felt as if you were unable to control important things in your academic affairs?",
    "PSS3": "In a semester, how often you felt nervous and stressed because of academic pressure?",
    "PSS4": "In a semester, how often you felt as if you could not cope with all the mandatory academic activities?",
    "PSS5": "In a semester, how often you felt confident about your ability to handle your academic / university problems?",
    "PSS6": "In a semester, how often you felt as if things in your academic life is going your way?",
    "PSS7": "In a semester, how often are you able to control irritations in your academic / university affairs?",
    "PSS8": "In a semester, how often you felt as if your academic performance was on top?",
    "PSS9": "In a semester, how often you got angered due to bad performance or low grades that are beyond your control?",
    "PSS10": "In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them?",
}

GAD_QUESTIONS = {
    "GAD1": "In a semester, how often you felt nervous, anxious or on edge due to academic pressure?",
    "GAD2": "In a semester, how often have you been unable to stop worrying about your academic affairs?",
    "GAD3": "In a semester, how often have you had trouble relaxing due to academic pressure?",
    "GAD4": "In a semester, how often have you been easily annoyed or irritated because of academic pressure?",
    "GAD5": "In a semester, how often have you worried too much about academic affairs?",
    "GAD6": "In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?",
    "GAD7": "In a semester, how often have you felt afraid, as if something awful might happen?",
}

PHQ_QUESTIONS = {
    "PHQ1": "In a semester, how often have you had little interest or pleasure in doing things?",
    "PHQ2": "In a semester, how often have you been feeling down, depressed or hopeless?",
    "PHQ3": "In a semester, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "PHQ4": "In a semester, how often have you been feeling tired or having little energy?",
    "PHQ5": "In a semester, how often have you had poor appetite or overeating?",
    "PHQ6": "In a semester, how often have you been feeling bad about yourself or that you are a failure?",
    "PHQ7": "In a semester, how often have you been having trouble concentrating on things (e.g., reading books)?",
    "PHQ8": "In a semester, how often have you moved or spoken so slowly that other people noticed, or been very restless?",
    "PHQ9": "In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself?",
}

# PSS scale (0–4)
PSS_OPTIONS = {
    "Never": 0,
    "Almost never": 1,
    "Sometimes": 2,
    "Fairly often": 3,
    "Very often": 4,
}

# GAD / PHQ scale (0–3)
GAD_PHQ_OPTIONS = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


# =========================
# 3) STREAMLIT UI
# =========================
def main():
    st.title("Student Mental Health Prediction System (Anxiety, Stress, Depression)")

    st.markdown(
        """
This app takes **demographic info + PSS-10 + GAD-7 + PHQ-9** responses and
uses **machine learning** models (trained on real student data) to classify:

- Anxiety: Present / Not present  
- Stress: Present / Not present  
- Depression: Present / Not present  
"""
    )

    # Train or load models
    with st.spinner("Training ML models from dataset... (only once)"):
        anxiety_model, stress_model, depression_model, feature_cols = train_models()

    st.success("Models ready ✅")

    # ------------- Demographic Inputs -------------
    st.header("1. Demographic & Academic Information")

    age = st.selectbox(
        "Age range",
        ["18-20", "21-23", "24-26", "27+"],
    )
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    university = st.text_input("University", "Daffodil International University")
    department = st.text_input("Department", "CSE")
    academic_year = st.selectbox(
        "Academic Year",
        ["1st", "2nd", "3rd", "4th", "Other"],
    )
    current_cgpa = st.selectbox(
        "Current CGPA range",
        ["<2.50", "2.50-2.99", "3.00-3.49", "3.50-4.00"],
    )
    waiver = st.selectbox(
        "Did you receive a waiver or scholarship?",
        ["No", "Yes (partial)", "Yes (full)"],
    )

    # ------------- PSS-10 -------------
    st.header("2. Perceived Stress Scale (PSS-10) – Academic Context")
    pss_answers = {}
    for code, q in PSS_QUESTIONS.items():
        ans = st.radio(q, list(PSS_OPTIONS.keys()), key=code)
        pss_answers[code] = PSS_OPTIONS[ans]

    # ------------- GAD-7 -------------
    st.header("3. Generalized Anxiety (GAD-7) – Academic Context")
    gad_answers = {}
    for code, q in GAD_QUESTIONS.items():
        ans = st.radio(q, list(GAD_PHQ_OPTIONS.keys()), key=code)
        gad_answers[code] = GAD_PHQ_OPTIONS[ans]

    # ------------- PHQ-9 -------------
    st.header("4. Depression (PHQ-9) – Academic Context")
    phq_answers = {}
    for code, q in PHQ_QUESTIONS.items():
        ans = st.radio(q, list(GAD_PHQ_OPTIONS.keys()), key=code)
        phq_answers[code] = GAD_PHQ_OPTIONS[ans]

    if st.button("🔍 Predict Mental Health Status"):
        # Build a single-row DataFrame with same column names as training data
        # ধরে নিচ্ছি dataset এ column নামগুলি এগুলোর মত:
        input_dict = {
            "Age": age,
            "Gender": gender,
            "University": university,
            "Department": department,
            "Academic_Year": academic_year,
            "Current_CGPA": current_cgpa,
            "waiver_or_scholarship": waiver,
        }

        # Add PSS/GAD/PHQ numeric answers
        input_dict.update(pss_answers)
        input_dict.update(gad_answers)
        input_dict.update(phq_answers)

        # Make sure all model feature columns exist in this row
        # যদি dataset এ আরও কিছু extra feature থাকে, সেগুলো default NaN হবে
        full_input = {col: input_dict.get(col, np.nan) for col in feature_cols}
        user_df = pd.DataFrame([full_input])

        # Predict using 3 separate models
        anx_pred = anxiety_model.predict(user_df)[0]
        str_pred = stress_model.predict(user_df)[0]
        dep_pred = depression_model.predict(user_df)[0]

        anx_proba = anxiety_model.predict_proba(user_df)[0][1]
        str_proba = stress_model.predict_proba(user_df)[0][1]
        dep_proba = depression_model.predict_proba(user_df)[0][1]

        st.subheader("✅ Prediction Results (ML Classification)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
**Anxiety**  
Prediction: `{"Present" if anx_pred == 1 else "Not Present"}`  
Risk score (ML probability): **{anx_proba:.2f}**
"""
            )

        with col2:
            st.markdown(
                f"""
**Stress**  
Prediction: `{"Present" if str_pred == 1 else "Not Present"}`  
Risk score (ML probability): **{str_proba:.2f}**
"""
            )

        with col3:
            st.markdown(
                f"""
**Depression**  
Prediction: `{"Present" if dep_pred == 1 else "Not Present"}`  
Risk score (ML probability): **{dep_proba:.2f}**
"""
            )

        st.info(
            "Note: These results are generated by **machine learning models** "
            "trained on real student data (PSS-10, GAD-7, PHQ-9 + demographics) – "
            "this is not just manual score summation."
        )


if __name__ == "__main__":
    main()
