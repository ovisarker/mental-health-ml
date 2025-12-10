# unified_mental_health_pipeline.py
# ----------------------------------
# Full unified ML pipeline for:
# - Anxiety_Binary
# - Stress_Binary
# - Depression_Binary
# - Overall Mental Health Status (main issue)
# - Basic XAI: numeric feature importance (Logistic Regression)
#
# Requires:
#   - Processed.csv in the same folder
#   - app.py in same folder (for Streamlit)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------------
print("🔹 Loading dataset from Processed.csv ...")
df = pd.read_csv("Processed.csv")
print("Shape:", df.shape)

# ---------------------------------------------------------
# STEP 2: Create binary labels
# ---------------------------------------------------------

# 2.1 Depression_Binary from 'Depression Label'
def make_depression_binary(label: str) -> int:
    if isinstance(label, str) and label.strip().lower() == "no depression":
        return 0
    return 1

if "Depression_Binary" not in df.columns:
    df["Depression_Binary"] = df["Depression Label"].apply(make_depression_binary)

# 2.2 Anxiety_Binary from GAD1–GAD7 (GAD_Total cutoff)
gad_cols = [col for col in df.columns if col.upper().startswith("GAD")]
df["GAD_Total"] = df[gad_cols].sum(axis=1)
df["Anxiety_Binary"] = df["GAD_Total"].apply(lambda x: 0 if x <= 4 else 1)

# 2.3 Stress_Binary from PSS1–PSS10 (PSS_Total cutoff)
pss_cols = [col for col in df.columns if col.upper().startswith("PSS")]
df["PSS_Total"] = df[pss_cols].sum(axis=1)
df["Stress_Binary"] = df["PSS_Total"].apply(lambda x: 0 if x <= 13 else 1)

print("\n🔹 Label distributions:")
print("Depression_Binary:\n", df["Depression_Binary"].value_counts())
print("Anxiety_Binary:\n", df["Anxiety_Binary"].value_counts())
print("Stress_Binary:\n", df["Stress_Binary"].value_counts())

# ---------------------------------------------------------
# STEP 3: Define features (X) & targets (y)
# ---------------------------------------------------------
drop_cols = [
    "Depression Label",
    "Depression_Binary",
    "Anxiety_Binary",
    "Stress_Binary",
    "Depression Value",    # if exists
    "GAD_Total",
    "PSS_Total",
]

drop_cols = list(set(drop_cols).intersection(df.columns))
X = df.drop(columns=drop_cols)
y_anx = df["Anxiety_Binary"]
y_str = df["Stress_Binary"]
y_dep = df["Depression_Binary"]

print("\n🔹 Feature shape:", X.shape)

# ---------------------------------------------------------
# STEP 4: Categorical vs numeric columns
# ---------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns (first few):", numeric_cols[:10])

# ---------------------------------------------------------
# STEP 5: Preprocessor (OneHot for cat, Scale for num)
# ---------------------------------------------------------
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols),
    ]
)

# ---------------------------------------------------------
# STEP 6: Train–test split (same split for all targets)
# ---------------------------------------------------------
X_train, X_test, y_anx_train, y_anx_test, y_str_train, y_str_test, y_dep_train, y_dep_test = train_test_split(
    X, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep
)

# ---------------------------------------------------------
# STEP 7: Build Logistic Regression pipeline
# ---------------------------------------------------------
def make_lr_pipeline():
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear"
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", clf),
    ])
    return model

anxiety_model = make_lr_pipeline()
stress_model = make_lr_pipeline()
depression_model = make_lr_pipeline()

# ---------------------------------------------------------
# STEP 8: Train models
# ---------------------------------------------------------
print("\n🔹 Training Anxiety model ...")
anxiety_model.fit(X_train, y_anx_train)

print("🔹 Training Stress model ...")
stress_model.fit(X_train, y_str_train)

print("🔹 Training Depression model ...")
depression_model.fit(X_train, y_dep_train)

# ---------------------------------------------------------
# (Optional) quick evaluation in console
# ---------------------------------------------------------
def _evaluate_model(name, model, X_t, y_t):
    print(f"\n===== {name} Model Evaluation =====")
    y_pred = model.predict(X_t)
    print("Accuracy:", accuracy_score(y_t, y_pred))
    print("Classification Report:\n", classification_report(y_t, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_t, y_pred))

_evaluate_model("Anxiety", anxiety_model, X_test, y_anx_test)
_evaluate_model("Stress", stress_model, X_test, y_str_test)
_evaluate_model("Depression", depression_model, X_test, y_dep_test)

# ---------------------------------------------------------
# STEP 9: Overall Mental Health (main issue) logic
# ---------------------------------------------------------
def determine_main_issue(anx: int, stre: int, dep: int) -> str:
    """
    anx, stre, dep are 0/1 predictions.
    Simple priority rule:
        - if all 0 -> No major issue
        - else Depression > Anxiety > Stress
    """
    if anx == 0 and stre == 0 and dep == 0:
        return "No major mental-health issue"

    if dep == 1:
        return "Depression-dominant"
    if anx == 1:
        return "Anxiety-dominant"
    if stre == 1:
        return "Stress-dominant"

    return "Mixed / Uncertain"

# ---------------------------------------------------------
# STEP 10: Predict for a single student (used by app.py)
# ---------------------------------------------------------
def predict_for_student(new_student_dict: dict):
    """
    new_student_dict: {column_name: value} same style as X columns.
    Called from Streamlit app.
    Returns: (anx_pred, str_pred, dep_pred, main_issue)
    """
    # 1 row DataFrame
    new_df = pd.DataFrame([new_student_dict])

    # ensure same columns as X (missing columns => NaN)
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[X.columns]

    anx_pred = anxiety_model.predict(new_df)[0]
    str_pred = stress_model.predict(new_df)[0]
    dep_pred = depression_model.predict(new_df)[0]

    main_issue = determine_main_issue(anx_pred, str_pred, dep_pred)

    # console print (for debugging)
    print("\nPrediction for new student:")
    print("Anxiety:", anx_pred, "| Stress:", str_pred, "| Depression:", dep_pred, "| Main:", main_issue)

    return anx_pred, str_pred, dep_pred, main_issue

# ---------------------------------------------------------
# STEP 11: Basic XAI – numeric-only LR models (with safe numeric handling)
# ---------------------------------------------------------
print("\n🔹 Training numeric-only LR models for basic XAI ...")

# numeric-only subset for simple interpretability
# convert everything to numeric, non-numeric -> NaN
x_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

Xn_train, Xn_test, yn_anx_train, yn_anx_test, yn_str_train, yn_str_test, yn_dep_train, yn_dep_test = train_test_split(
    x_numeric, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep
)

def train_lr_numeric(X_train_num, y_train_num):
    """
    Train a simple Logistic Regression only on numeric features
    for interpretability (feature importance via coefficients).
    """
    # ensure numeric and handle NaN
    X_train_num = X_train_num.apply(pd.to_numeric, errors="coerce")
    X_train_num = X_train_num.fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train_num)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear"
    )
    clf.fit(Xs, y_train_num)
    return clf, scaler

# Global models for XAI (imported in app.py)
anx_clf_num, _ = train_lr_numeric(Xn_train, yn_anx_train)
str_clf_num, _ = train_lr_numeric(Xn_train, yn_str_train)
dep_clf_num, _ = train_lr_numeric(Xn_train, yn_dep_train)

print("\n✅ Unified ML + Overall Mental Health + Basic XAI ready.")
