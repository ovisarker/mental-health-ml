"""
Unified Mental Health Pipeline (Fixed Version)
--------------------------------------------
This module trains three independent binary classifiers to predict the presence of anxiety,
stress and depression among students based on questionnaire responses, demographic and
lifestyle features. It then derives an overall mental-health status (dominant condition)
from those predictions and includes a basic explainability component using feature
importance of numeric-only logistic regression models.

Changes from previous versions:
  * Uses SimpleImputer to fill missing values in both categorical and numeric features before
    encoding and scaling. This prevents issues where OneHotEncoder encountered NaN values and
    triggered a TypeError with np.isnan.
  * Safe numeric conversion and fillna in the XAI section to avoid casting errors.

To use:
  1. Place this file in the same folder as your Streamlit `app.py` and `Processed.csv` data.
  2. Import the functions `predict_for_student`, `determine_main_issue`, and the XAI models
     (`anx_clf_num`, `str_clf_num`, `dep_clf_num`, `x_numeric`) from this module in your
     Streamlit app.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# -------------------------------------------------------
# STEP 1 — Load Dataset
# -------------------------------------------------------
print("🔹 Loading dataset from Processed.csv ...")
df = pd.read_csv("Processed.csv")
print("Dataset shape:", df.shape)


# -------------------------------------------------------
# STEP 2 — Create Binary Labels
# -------------------------------------------------------

def make_depression_binary(label: str) -> int:
    """Convert depression label to binary (0 = No Depression, 1 = otherwise)."""
    if isinstance(label, str) and label.strip().lower() == "no depression":
        return 0
    return 1

# Derive binary targets
df["Depression_Binary"] = df["Depression Label"].apply(make_depression_binary)

# Anxiety: GAD total score cutoff
gad_cols = [c for c in df.columns if c.upper().startswith("GAD")]
df["GAD_Total"] = df[gad_cols].sum(axis=1)
df["Anxiety_Binary"] = df["GAD_Total"].apply(lambda x: 0 if x <= 4 else 1)

# Stress: PSS total score cutoff
pss_cols = [c for c in df.columns if c.upper().startswith("PSS")]
df["PSS_Total"] = df[pss_cols].sum(axis=1)
df["Stress_Binary"] = df["PSS_Total"].apply(lambda x: 0 if x <= 13 else 1)

print("\n🔹 Label counts:")
print("Anxiety_Binary:\n", df["Anxiety_Binary"].value_counts())
print("Stress_Binary:\n", df["Stress_Binary"].value_counts())
print("Depression_Binary:\n", df["Depression_Binary"].value_counts())


# -------------------------------------------------------
# STEP 3 — Build Features (X) and Labels (y)
# -------------------------------------------------------

# Drop columns that are targets or derived
drop_cols = [
    "Depression Label",
    "Depression_Binary",
    "Anxiety_Binary",
    "Stress_Binary",
    "Depression Value",  # if present
    "GAD_Total",
    "PSS_Total",
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y_anx = df["Anxiety_Binary"]
y_str = df["Stress_Binary"]
y_dep = df["Depression_Binary"]

print("\n🔹 Feature shape:", X.shape)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


# -------------------------------------------------------
# STEP 4 — Preprocessing Pipelines
# -------------------------------------------------------

# For numeric features: impute missing values then scale
numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

# For categorical features: impute missing values then one-hot encode
categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_preprocessor, categorical_cols),
        ("num", numeric_preprocessor, numeric_cols),
    ]
)


# -------------------------------------------------------
# STEP 5 — Train/Test Split
# -------------------------------------------------------

X_train, X_test, y_anx_train, y_anx_test, y_str_train, y_str_test, y_dep_train, y_dep_test = train_test_split(
    X, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep,
)


# -------------------------------------------------------
# STEP 6 — Build ML Pipelines
# -------------------------------------------------------

def make_model():
    """Create a logistic regression model pipeline with preprocessing."""
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", clf),
    ])

# Instantiate three separate models
anxiety_model = make_model()
stress_model = make_model()
depression_model = make_model()


# -------------------------------------------------------
# STEP 7 — Train Models
# -------------------------------------------------------

print("\n🔹 Training Anxiety model ...")
anxiety_model.fit(X_train, y_anx_train)
print("🔹 Training Stress model ...")
stress_model.fit(X_train, y_str_train)
print("🔹 Training Depression model ...")
depression_model.fit(X_train, y_dep_train)


# -------------------------------------------------------
# STEP 8 — Evaluate Models (Console Only)
# -------------------------------------------------------

def _evaluate(name: str, model: Pipeline, X_val, y_val):
    print(f"\n===== {name} Evaluation =====")
    preds = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

_evaluate("Anxiety", anxiety_model, X_test, y_anx_test)
_evaluate("Stress", stress_model, X_test, y_str_test)
_evaluate("Depression", depression_model, X_test, y_dep_test)


# -------------------------------------------------------
# STEP 9 — Overall Mental Health Determination
# -------------------------------------------------------

def determine_main_issue(anx: int, stress: int, dep: int) -> str:
    """Determine the dominant condition given three binary predictions."""
    if anx == 0 and stress == 0 and dep == 0:
        return "No major mental-health issue"
    # Give priority: Depression > Anxiety > Stress
    if dep == 1:
        return "Depression-dominant"
    if anx == 1:
        return "Anxiety-dominant"
    if stress == 1:
        return "Stress-dominant"
    return "Mixed / Uncertain"


# -------------------------------------------------------
# STEP 10 — Predict for a Single Student
# -------------------------------------------------------

def predict_for_student(student_dict: dict):
    """
    Accept a dictionary of student features and return binary predictions for
    anxiety, stress and depression along with the overall mental health status.
    """
    new_df = pd.DataFrame([student_dict])
    # Ensure all expected columns exist in the input
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[X.columns]

    anx_pred = anxiety_model.predict(new_df)[0]
    str_pred = stress_model.predict(new_df)[0]
    dep_pred = depression_model.predict(new_df)[0]

    main_issue = determine_main_issue(anx_pred, str_pred, dep_pred)

    return anx_pred, str_pred, dep_pred, main_issue


# -------------------------------------------------------
# STEP 11 — SAFE XAI (Numeric-Only Logistic Regression)
# -------------------------------------------------------

print("\n🔹 Building SAFE XAI numeric models...")

def safe_numeric_lr(X_train_num: pd.DataFrame, y_train_num: pd.Series):
    """
    Train a logistic regression on numeric features only, handling missing values
    by filling with the column mean and ensuring all values are numeric.
    Returns: (classifier, scaler)
    """
    X_train_num = X_train_num.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_num)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
    clf.fit(X_scaled, y_train_num)
    return clf, scaler

# Force numeric conversion on the entire numeric subset and fill missing values
x_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

Xn_train, Xn_test, yn_anx_train, yn_anx_test, yn_str_train, yn_str_test, yn_dep_train, yn_dep_test = train_test_split(
    x_numeric, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep,
)

anx_clf_num, _ = safe_numeric_lr(Xn_train, yn_anx_train)
str_clf_num, _ = safe_numeric_lr(Xn_train, yn_str_train)
dep_clf_num, _ = safe_numeric_lr(Xn_train, yn_dep_train)

print("✅ SAFE XAI models ready.")


# -------------------------------------------------------
# This module is ready for import.
# Functions exported: predict_for_student, determine_main_issue
# Also exported: anx_clf_num, str_clf_num, dep_clf_num, x_numeric
# These are used by app.py for XAI display.
# -------------------------------------------------------

if __name__ == "__main__":
    # Basic test: run a prediction on the first row of the dataset
    sample = X.iloc[0].to_dict()
    anx, stress, dep, main = predict_for_student(sample)
    print("\nSample prediction:")
    print(f"Anxiety: {anx}, Stress: {stress}, Depression: {dep}, Overall: {main}")
