# app_Pre_final.py
# Student Mental Health Assessment using Machine Learning
# Uses Logistic Regression models trained on Processed.csv inside the app.

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------
# Page config + basic theming
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Assessment (ML-based)",
    page_icon="🧠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    body {background-color:#0E1117;color:#FAFAFA;}
    h1,h2,h3,h4,h5,h6 {color:#E0E0E0;}
    .stButton>button {
        background: #ff4b4b;
        color: white;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover {
        background: #ff6b6b;
        color: white;
    }
    .risk-low {color:#00e676;font-weight:600;}
    .risk-moderate {color:#ffeb3b;font-weight:600;}
    .risk-high {color:#ff9800;font-weight:600;}
    .risk-critical {color:#ff5252;font-weight:600;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Utility: scoring from questionnaire totals (no ML)
# ---------------------------------------------------------------------
def score_stress(pss_scores):
    total = float(sum(pss_scores))
    if total <= 13:
        label = "Low Perceived Stress"
    elif total <= 26:
        label = "Moderate Stress"
    else:
        label = "High Perceived Stress"
    return int(total), label


def score_anxiety(gad_scores):
    total = float(sum(gad_scores))
    if total <= 4:
        label = "Minimal Anxiety"
    elif total <= 9:
        label = "Mild Anxiety"
    elif total <= 14:
        label = "Moderate Anxiety"
    else:
        label = "Severe Anxiety"
    return int(total), label


def score_depression(phq_scores):
    total = float(sum(phq_scores))
    if total <= 4:
        label = "No / Minimal Depression"
    elif total <= 9:
        label = "Mild Depression"
    elif total <= 14:
        label = "Moderate Depression"
    elif total <= 19:
        label = "Moderately Severe Depression"
    else:
        label = "Severe Depression"
    return int(total), label


def risk_tier_from_label(label: str) -> str:
    """Convert severity label to a generic risk tier."""
    text = label.lower()
    if "severe" in text:
        return "Critical"
    if "high" in text:
        return "High"
    if "moderate" in text or "mild" in text:
        return "Moderate"
    return "Low"


# For ranking which issue is most serious (used with ML outputs)
STRESS_RANK = {
    "Low Perceived Stress": 0,
    "Moderate Stress": 1,
    "High Perceived Stress": 2,
}
ANX_RANK = {
    "Minimal Anxiety": 0,
    "Mild Anxiety": 1,
    "Moderate Anxiety": 2,
    "Severe Anxiety": 3,
}
DEP_RANK = {
    "No Depression": 0,
    "Minimal Depression": 1,
    "Mild Depression": 2,
    "Moderate Depression": 3,
    "Moderately Severe Depression": 4,
    "Severe Depression": 5,
    "No / Minimal Depression": 1,  # from manual scoring
}


def dominant_issue_from_labels(a_label: str, s_label: str, d_label: str):
    scores = {
        "Anxiety": ANX_RANK.get(a_label, 0),
        "Stress": STRESS_RANK.get(s_label, 0),
        "Depression": DEP_RANK.get(d_label, 0),
    }
    best_issue = max(scores, key=scores.get)
    if all(v == 0 for v in scores.values()):
        return "None", scores
    return best_issue, scores


# ---------------------------------------------------------------------
# ML: load data + train models (once, cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_data_and_train_models():
    """
    Loads Processed.csv and trains three Logistic Regression models (one for
    Stress, Anxiety, and Depression labels).

    ML is really used here:
    - Inputs: demographics + PSS/GAD/PHQ items (33 features)
    - Targets: Stress Label, Anxiety Label, Depression Label
    """
    csv_path = "Processed.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed.csv not found in app directory. "
            f"Please place your dataset at: {os.path.abspath(csv_path)}"
        )

    df = pd.read_csv(csv_path)

    # Features used for ML (33 total)
    feature_cols = (
        ["Age", "Gender", "University", "Department",
         "Academic_Year", "Current_CGPA", "waiver_or_scholarship"]
        + [f"PSS{i}" for i in range(1, 11)]
        + [f"GAD{i}" for i in range(1, 8)]
        + [f"PHQ{i}" for i in range(1, 10)]
    )

    # Categorical vs numeric
    cat_cols = [
        "Age", "Gender", "University", "Department",
        "Academic_Year", "Current_CGPA", "waiver_or_scholarship",
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    models = {}
    for issue, target_col in [
        ("Anxiety", "Anxiety Label"),
        ("Stress", "Stress Label"),
        ("Depression", "Depression Label"),
    ]:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=500, multi_class="auto")),
            ]
        )
        pipe.fit(df[feature_cols], df[target_col].astype(str))
        models[issue] = pipe

    return df, feature_cols, cat_cols, num_cols, models


# Try to load + train once
try:
    DATAFRAME, FEATURE_COLS, CAT_COLS, NUM_COLS, MODELS = load_data_and_train_models()
    DATA_READY = True
except Exception as e:
    DATA_READY = False
    DATA_ERROR = str(e)


def ml_predict_all(input_row: dict):
    """Run all three ML models on a single user input row."""
    sample_df = pd.DataFrame([input_row], columns=FEATURE_COLS)
    pred_labels = {}
    pred_probs = {}
    for issue, model in MODELS.items():
        proba = model.predict_proba(sample_df)[0]
        labels = model.classes_
        label = labels[np.argmax(proba)]
        pred_labels[issue] = str(label)
        pred_probs[issue] = dict(zip(labels, proba))
    return pred_labels, pred_probs


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🧩 Assessment", "📊 Dataset Overview"],
    index=0,
)

# ---------------------------------------------------------------------
# PAGE 1 – Assessment
# ---------------------------------------------------------------------
if page == "🧩 Assessment":
    st.title("🧠 Student Mental Health Assessment (Machine Learning-based)")
    st.caption(
        "This tool uses **standard questionnaires** (PSS-10, GAD-7, PHQ-9) "
        "and **Logistic Regression models** trained on real student data."
    )

    if not DATA_READY:
        st.error(
            "❌ Machine-learning backend is not available.\n\n"
            f"Reason: {DATA_ERROR}"
        )
        st.stop()

    # ---------- Options from dataset (ensures perfect match with training) ----
    age_opts = sorted(DATAFRAME["Age"].dropna().unique().tolist())
    gender_opts = sorted(DATAFRAME["Gender"].dropna().unique().tolist())
    uni_opts = sorted(DATAFRAME["University"].dropna().unique().tolist())
    dept_opts = sorted(DATAFRAME["Department"].dropna().unique().tolist())
    year_opts = sorted(DATAFRAME["Academic_Year"].dropna().unique().tolist())
    cgpa_opts = sorted(DATAFRAME["Current_CGPA"].dropna().unique().tolist())
    waiver_opts = sorted(DATAFRAME["waiver_or_scholarship"].dropna().unique().tolist())

    with st.form("assessment_form"):
        st.subheader("🎓 Basic Academic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.selectbox("Age group", age_opts)
            gender = st.selectbox("Gender", gender_opts)
        with col2:
            university = st.selectbox("University type", uni_opts)
            department = st.selectbox("Department", dept_opts)
        with col3:
            academic_year = st.selectbox("Current academic year", year_opts)
            cgpa = st.selectbox("Current CGPA range", cgpa_opts)
        waiver = st.selectbox("Waiver / Scholarship status", waiver_opts)

        st.markdown("---")
        st.subheader("😓 Perceived Stress (PSS-10)")
        st.caption("Scale 0–4: 0 = Never, 4 = Very Often")
        pss_scores = []
        for i in range(1, 11):
            pss_scores.append(
                st.slider(
                    f"PSS{i}",
                    min_value=0,
                    max_value=4,
                    value=2,
                    step=1,
                )
            )

        st.markdown("---")
        st.subheader("😰 Generalized Anxiety (GAD-7)")
        st.caption("Scale 0–3: 0 = Not at all, 3 = Nearly every day")
        gad_scores = []
        for i in range(1, 8):
            gad_scores.append(
                st.slider(
                    f"GAD{i}",
                    min_value=0,
                    max_value=3,
                    value=1,
                    step=1,
                )
            )

        st.markdown("---")
        st.subheader("😔 Depressive Symptoms (PHQ-9)")
        st.caption("Scale 0–3: 0 = Not at all, 3 = Nearly every day")
        phq_scores = []
        for i in range(1, 10):
            phq_scores.append(
                st.slider(
                    f"PHQ{i}",
                    min_value=0,
                    max_value=3,
                    value=1,
                    step=1,
                )
            )

        submitted = st.form_submit_button("Assess My Mental Health")

    if submitted:
        # -----------------------------------------------------------------
        # 1) Rule-based scoring (official questionnaire thresholds)
        # -----------------------------------------------------------------
        stress_total, stress_label_manual = score_stress(pss_scores)
        anx_total, anx_label_manual = score_anxiety(gad_scores)
        dep_total, dep_label_manual = score_depression(phq_scores)

        # -----------------------------------------------------------------
        # 2) Prepare row for ML & run models
        # -----------------------------------------------------------------
        input_row = {
            "Age": age,
            "Gender": gender,
            "University": university,
            "Department": department,
            "Academic_Year": academic_year,
            "Current_CGPA": cgpa,
            "waiver_or_scholarship": waiver,
        }
        for i, val in enumerate(pss_scores, start=1):
            input_row[f"PSS{i}"] = val
        for i, val in enumerate(gad_scores, start=1):
            input_row[f"GAD{i}"] = val
        for i, val in enumerate(phq_scores, start=1):
            input_row[f"PHQ{i}"] = val

        ml_labels, ml_probs = ml_predict_all(input_row)

        # Use ML labels (not the manual ones) to decide dominant issue
        dom_issue, dom_scores = dominant_issue_from_labels(
            ml_labels["Anxiety"], ml_labels["Stress"], ml_labels["Depression"]
        )

        # -----------------------------------------------------------------
        # 3) Display results
        # -----------------------------------------------------------------
        st.markdown("## 🧾 Results")

        # --- Main conclusion (uses ML models) ---
        if dom_issue == "None":
            st.info(
                "According to the **machine-learning model**, no single mental health "
                "issue is clearly dominant. However, please review the scores below."
            )
        else:
            st.success(
                f"According to the **machine-learning model trained on student data**, "
                f"your most prominent challenge appears to be **{dom_issue}**."
            )

        # --- Detailed per-issue summary (manual + ML label) ---
        colA, colB, colC = st.columns(3)

        with colA:
            risk = risk_tier_from_label(ml_labels["Stress"])
            css = {
                "Low": "risk-low",
                "Moderate": "risk-moderate",
                "High": "risk-high",
                "Critical": "risk-critical",
            }.get(risk, "")
            st.markdown("### 😓 Stress")
            st.write(f"**PSS-10 Total Score:** {stress_total}")
            st.write(f"**Questionnaire Category:** {stress_label_manual}")
            st.write(
                f"**ML-Predicted Label:** {ml_labels['Stress']} "
                f"( <span class='{css}'>Risk: {risk}</span> )",
                unsafe_allow_html=True,
            )

        with colB:
            risk = risk_tier_from_label(ml_labels["Anxiety"])
            css = {
                "Low": "risk-low",
                "Moderate": "risk-moderate",
                "High": "risk-high",
                "Critical": "risk-critical",
            }.get(risk, "")
            st.markdown("### 😰 Anxiety")
            st.write(f"**GAD-7 Total Score:** {anx_total}")
            st.write(f"**Questionnaire Category:** {anx_label_manual}")
            st.write(
                f"**ML-Predicted Label:** {ml_labels['Anxiety']} "
                f"( <span class='{css}'>Risk: {risk}</span> )",
                unsafe_allow_html=True,
            )

        with colC:
            risk = risk_tier_from_label(ml_labels["Depression"])
            css = {
                "Low": "risk-low",
                "Moderate": "risk-moderate",
                "High": "risk-high",
                "Critical": "risk-critical",
            }.get(risk, "")
            st.markdown("### 😔 Depression")
            st.write(f"**PHQ-9 Total Score:** {dep_total}")
            st.write(f"**Questionnaire Category:** {dep_label_manual}")
            st.write(
                f"**ML-Predicted Label:** {ml_labels['Depression']} "
                f"( <span class='{css}'>Risk: {risk}</span> )",
                unsafe_allow_html=True,
            )

        # --- Suggestions by dominant issue (simple rule-based text) ---
        st.markdown("---")
        st.markdown("### 🩺 Brief Self-care Suggestions (Not a diagnosis)")

        if dom_issue == "Stress":
            st.write(
                "- Review your **time management** and academic load.\n"
                "- Practice short **breathing exercises** or **walks** between study blocks.\n"
                "- Try to maintain regular **sleep** and reduce caffeine late at night."
            )
        elif dom_issue == "Anxiety":
            st.write(
                "- Use **structured routines** to reduce uncertainty.\n"
                "- Limit constant checking of marks / social media.\n"
                "- Consider talking with a **counsellor** if anxiety impacts daily tasks."
            )
        elif dom_issue == "Depression":
            st.write(
                "- Maintain basic routines: **sleep, food, hygiene**, even if motivation is low.\n"
                "- Stay connected with at least **one trusted person**.\n"
                "- If you ever feel unsafe or have self-harm thoughts, seek **immediate professional help**."
            )
        else:
            st.write(
                "Scores are relatively low. Still, maintaining **healthy habits** "
                "and monitoring your mood regularly is recommended."
            )

        st.caption(
            "⚠️ This tool is **not a clinical diagnosis**. Results are for academic "
            "and awareness purposes only."
        )

# ---------------------------------------------------------------------
# PAGE 2 – Dataset overview (for viva / teachers)
# ---------------------------------------------------------------------
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset & Model Overview")

    if not DATA_READY:
        st.error(
            "Dataset / ML models could not be loaded.\n\n"
            f"Reason: {DATA_ERROR}"
        )
        st.stop()

    st.subheader("Dataset Summary")
    st.write(f"Total records: **{DATAFRAME.shape[0]}**")
    st.write(f"Total columns: **{DATAFRAME.shape[1]}**")

    st.dataframe(DATAFRAME.head(), use_container_width=True)

    st.markdown("#### Target Label Distribution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Stress Label**")
        st.bar_chart(DATAFRAME["Stress Label"].value_counts())
    with col2:
        st.write("**Anxiety Label**")
        st.bar_chart(DATAFRAME["Anxiety Label"].value_counts())
    with col3:
        st.write("**Depression Label**")
        st.bar_chart(DATAFRAME["Depression Label"].value_counts())

    st.markdown("#### Features used by ML models")
    st.write(", ".join(FEATURE_COLS))

    st.info(
        "Each model is a **Logistic Regression** classifier inside a "
        "scikit-learn `Pipeline` with `ColumnTransformer` (One-Hot Encoding "
        "for categorical features + StandardScaler for numeric features)."
    )

# ---------------------------------------------------------------------
# Footer – team name
# ---------------------------------------------------------------------
st.markdown(
    """
---
**Developed by:** *Mental Health ML Research Team*  
Department of CSE – Student Mental Health Project (Phase-2)
""",
    unsafe_allow_html=True,
)
