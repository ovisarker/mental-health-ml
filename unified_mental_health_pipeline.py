"""
Unified Mental Health Pipeline
------------------------------
- Loads Processed.csv
- Builds binary labels for Anxiety, Stress, Depression
- Trains 3 Logistic Regression models (with preprocessing)
- Provides:
    - predict_for_student(student_dict)
    - risk_levels_for_student(student_dict)
    - determine_main_issue(anx, stress, dep)
    - x_numeric, anx_clf_num, str_clf_num, dep_clf_num (for XAI)

NOTE: Make sure Processed.csv is in the same folder.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------
# STEP 1 — Load dataset
# -------------------------------------------------------
DATA_PATH = "Processed.csv"  # change if needed

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------
# STEP 2 — Create binary labels
# -------------------------------------------------------

# 2.1 Depression_Binary
if "Depression Label" in df.columns:
    def _dep_binary(val):
        if isinstance(val, str) and val.strip().lower() == "no depression":
            return 0
        return 1
    df["Depression_Binary"] = df["Depression Label"].apply(_dep_binary)
else:
    # Fallback: use PHQ total
    phq_cols = [c for c in df.columns if c.upper().startswith("PHQ")]
    df["PHQ_Total"] = df[phq_cols].sum(axis=1)
    df["Depression_Binary"] = (df["PHQ_Total"] >= 7).astype(int)

# 2.2 Anxiety_Binary from GAD
gad_cols = [c for c in df.columns if c.upper().startswith("GAD")]
if gad_cols:
    df["GAD_Total"] = df[gad_cols].sum(axis=1)
    # cut-off based on 0–4 scale: <=4 no anxiety, >=5 anxiety
    df["Anxiety_Binary"] = (df["GAD_Total"] > 4).astype(int)
else:
    df["Anxiety_Binary"] = 0  # fallback

# 2.3 Stress_Binary from PSS
pss_cols = [c for c in df.columns if c.upper().startswith("PSS")]
if pss_cols:
    df["PSS_Total"] = df[pss_cols].sum(axis=1)
    # standard: <=13 low, >=14 stress present
    df["Stress_Binary"] = (df["PSS_Total"] >= 14).astype(int)
else:
    df["Stress_Binary"] = 0  # fallback

# -------------------------------------------------------
# STEP 3 — Features (X) and Targets (y)
# -------------------------------------------------------

drop_cols = [
    "Depression Label", "Depression Value",
    "Anxiety Label", "Anxiety Value",
    "Stress Label", "Stress Value",
    "Depression_Binary", "Anxiety_Binary", "Stress_Binary",
    "GAD_Total", "PSS_Total", "PHQ_Total"
]

drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y_anx = df["Anxiety_Binary"]
y_str = df["Stress_Binary"]
y_dep = df["Depression_Binary"]

# Identify types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# -------------------------------------------------------
# STEP 4 — Preprocessing pipeline (for final models)
# -------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# -------------------------------------------------------
# STEP 5 — Train 3 Logistic Regression models
# (train on full data for deployment simplicity)
# -------------------------------------------------------

def make_clf():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear"
    )

anxiety_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", make_clf())
])
stress_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", make_clf())
])
depression_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", make_clf())
])

# Fit models
anxiety_model.fit(X, y_anx)
stress_model.fit(X, y_str)
depression_model.fit(X, y_dep)

# -------------------------------------------------------
# STEP 6 — Numeric-only models for XAI
# -------------------------------------------------------

# Build numeric-only dataframe safely
x_numeric = X[numeric_cols].copy()

# convert everything to numeric, coerce errors
for col in x_numeric.columns:
    x_numeric[col] = pd.to_numeric(x_numeric[col], errors="coerce")

# fill missing
x_numeric = x_numeric.fillna(x_numeric.median(numeric_only=True))

anx_clf_num = make_clf()
str_clf_num = make_clf()
dep_clf_num = make_clf()

anx_clf_num.fit(x_numeric, y_anx)
str_clf_num.fit(x_numeric, y_str)
dep_clf_num.fit(x_numeric, y_dep)

# -------------------------------------------------------
# STEP 7 — Helper: determine main issue
# -------------------------------------------------------

def determine_main_issue(anx: int, stress: int, dep: int) -> str:
    """
    Determine dominant mental-health issue from 3 binary flags.
    Priority: Depression > Anxiety > Stress
    """
    if anx == 0 and stress == 0 and dep == 0:
        return "No major issue"

    # All present
    if anx == 1 and stress == 1 and dep == 1:
        return "Depression-dominant"

    # Two present
    if dep == 1 and anx == 1 and stress == 0:
        return "Depression-dominant"
    if dep == 1 and stress == 1 and anx == 0:
        return "Depression-dominant"
    if anx == 1 and stress == 1 and dep == 0:
        return "Anxiety-dominant"

    # Single present
    if dep == 1:
        return "Depression-dominant"
    if anx == 1:
        return "Anxiety-dominant"
    if stress == 1:
        return "Stress-dominant"

    return "No major issue"


# -------------------------------------------------------
# STEP 8 — Risk Levels (score-based, for UI only)
# -------------------------------------------------------

def risk_levels_for_student(student_dict: dict):
    """
    Compute textual risk levels (no scores) for Stress, Anxiety, Depression
    using PSS, GAD, PHQ items in the student_dict.
    """
    # Stress (PSS-10, 0–40)
    pss_vals = [student_dict.get(f"PSS{i}", 0) for i in range(1, 11)]
    pss_total = sum(pss_vals)

    if pss_total <= 13:
        stress_level = "Low"
    elif pss_total <= 26:
        stress_level = "Moderate"
    else:
        stress_level = "High"

    # Anxiety (GAD-7, 0–28 adjusted)
    gad_vals = [student_dict.get(f"GAD{i}", 0) for i in range(1, 8)]
    gad_total = sum(gad_vals)

    if gad_total <= 6:
        anx_level = "Minimal"
    elif gad_total <= 12:
        anx_level = "Mild"
    elif gad_total <= 19:
        anx_level = "Moderate"
    else:
        anx_level = "Severe"

    # Depression (PHQ-9, 0–36 adjusted)
    phq_vals = [student_dict.get(f"PHQ{i}", 0) for i in range(1, 10)]
    phq_total = sum(phq_vals)

    if phq_total <= 6:
        dep_level = "Minimal"
    elif phq_total <= 13:
        dep_level = "Mild"
    elif phq_total <= 19:
        dep_level = "Moderate"
    elif phq_total <= 25:
        dep_level = "Moderately Severe"
    else:
        dep_level = "Severe"

    return {
        "Stress": stress_level,
        "Anxiety": anx_level,
        "Depression": dep_level,
    }

# -------------------------------------------------------
# STEP 9 — Public function: predict_for_student
# -------------------------------------------------------

def predict_for_student(student_dict: dict):
    """
    Student dict must contain:
        Age, Gender, University, Department, Academic_Year,
        Current_CGPA, waiver_or_scholarship,
        PSS1..PSS10, GAD1..GAD7, PHQ1..PHQ9
    Extra keys are ignored; missing keys are filled with NaN.
    """
    # Build single-row DataFrame
    row = pd.DataFrame([student_dict])

    # Ensure all training columns present
    for col in X.columns:
        if col not in row.columns:
            row[col] = np.nan

    # Order columns
    row = row[X.columns]

    anx = int(anxiety_model.predict(row)[0])
    stress = int(stress_model.predict(row)[0])
    dep = int(depression_model.predict(row)[0])

    main_issue = determine_main_issue(anx, stress, dep)

    return anx, stress, dep, main_issue


if __name__ == "__main__":
    # Simple internal test (only if you run this file directly)
    sample = X.iloc[0].to_dict()
    a, s, d, m = predict_for_student(sample)
    print("Sample prediction -> Anxiety:", a, "Stress:", s, "Depression:", d, "Main:", m)
    print("Sample risk:", risk_levels_for_student(sample))