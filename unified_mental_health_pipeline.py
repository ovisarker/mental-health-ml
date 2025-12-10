# unified_mental_health_pipeline.py
# -------------------------------------------------------
# Full unified ML Pipeline for:
# - Anxiety Prediction (Binary)
# - Stress Prediction (Binary)
# - Depression Prediction (Binary)
# - Overall Mental Health Status (Dominant Issue)
# - Safe XAI: Numeric Feature Importance
#
# Works perfectly with Streamlit app.py
# -------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------------------------
# STEP 1 — Load Dataset
# -------------------------------------------------------
print("🔹 Loading dataset: Processed.csv ...")

df = pd.read_csv("Processed.csv")
print("Dataset shape:", df.shape)


# -------------------------------------------------------
# STEP 2 — Create Binary Labels
# -------------------------------------------------------

# 2.1 Depression_Binary
def make_depression_binary(label: str):
    if isinstance(label, str) and label.strip().lower() == "no depression":
        return 0
    return 1

df["Depression_Binary"] = df["Depression Label"].apply(make_depression_binary)

# 2.2 Anxiety (GAD total)
gad_cols = [c for c in df.columns if c.upper().startswith("GAD")]
df["GAD_Total"] = df[gad_cols].sum(axis=1)
df["Anxiety_Binary"] = df["GAD_Total"].apply(lambda x: 0 if x <= 4 else 1)

# 2.3 Stress (PSS total)
pss_cols = [c for c in df.columns if c.upper().startswith("PSS")]
df["PSS_Total"] = df[pss_cols].sum(axis=1)
df["Stress_Binary"] = df["PSS_Total"].apply(lambda x: 0 if x <= 13 else 1)

print("\n🔹 Label counts:")
print(df["Anxiety_Binary"].value_counts())
print(df["Stress_Binary"].value_counts())
print(df["Depression_Binary"].value_counts())


# -------------------------------------------------------
# STEP 3 — Build Features (X) and Labels (y)
# -------------------------------------------------------
drop_cols = [
    "Depression Label",
    "Depression_Binary",
    "Anxiety_Binary",
    "Stress_Binary",
    "Depression Value",
    "GAD_Total",
    "PSS_Total"
]

drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y_anx = df["Anxiety_Binary"]
y_str = df["Stress_Binary"]
y_dep = df["Depression_Binary"]

print("\n🔹 Features shape:", X.shape)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


# -------------------------------------------------------
# STEP 4 — Preprocessor
# -------------------------------------------------------
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ]
)


# -------------------------------------------------------
# STEP 5 — Split Dataset
# -------------------------------------------------------
X_train, X_test, y_anx_train, y_anx_test, y_str_train, y_str_test, y_dep_train, y_dep_test = train_test_split(
    X, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep
)


# -------------------------------------------------------
# STEP 6 — Build Pipeline Model
# -------------------------------------------------------
def make_model():
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear"
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", clf)
    ])


anxiety_model = make_model()
stress_model = make_model()
depression_model = make_model()


# -------------------------------------------------------
# STEP 7 — Train Models
# -------------------------------------------------------
print("\n🔹 Training Anxiety Model ...")
anxiety_model.fit(X_train, y_anx_train)

print("🔹 Training Stress Model ...")
stress_model.fit(X_train, y_str_train)

print("🔹 Training Depression Model ...")
depression_model.fit(X_train, y_dep_train)


# -------------------------------------------------------
# STEP 8 — Quick Evaluation (Console Only)
# -------------------------------------------------------
def _eval(name, model, X_t, y_t):
    print(f"\n===== {name} Evaluation =====")
    pred = model.predict(X_t)
    print("Accuracy:", accuracy_score(y_t, pred))
    print(classification_report(y_t, pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_t, pred))

_eval("Anxiety", anxiety_model, X_test, y_anx_test)
_eval("Stress", stress_model, X_test, y_str_test)
_eval("Depression", depression_model, X_test, y_dep_test)


# -------------------------------------------------------
# STEP 9 — Determine Overall Mental Health
# -------------------------------------------------------
def determine_main_issue(anx, stre, dep):
    if anx == 0 and stre == 0 and dep == 0:
        return "No major mental-health issue"
    if dep == 1:
        return "Depression-dominant"
    if anx == 1:
        return "Anxiety-dominant"
    if stre == 1:
        return "Stress-dominant"
    return "Mixed / Uncertain"


# -------------------------------------------------------
# STEP 10 — Predict for a Single Student (Used by Streamlit)
# -------------------------------------------------------
def predict_for_student(student_dict):
    new_df = pd.DataFrame([student_dict])

    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[X.columns]

    anx = anxiety_model.predict(new_df)[0]
    stress = stress_model.predict(new_df)[0]
    dep = depression_model.predict(new_df)[0]

    main = determine_main_issue(anx, stress, dep)

    return anx, stress, dep, main


# -------------------------------------------------------
# STEP 11 — SAFE XAI (Numeric-Only Logistic Regression)
# -------------------------------------------------------
print("\n🔹 Building SAFE XAI numeric models...")

# Convert numeric columns safely
x_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

Xn_train, Xn_test, yn_anx_train, yn_anx_test, yn_str_train, yn_str_test, yn_dep_train, yn_dep_test = train_test_split(
    x_numeric, y_anx, y_str, y_dep,
    test_size=0.2,
    random_state=42,
    stratify=y_dep
)


def train_numeric_lr(X_train_num, y_train_num):
    X_train_num = X_train_num.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_num)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear"
    )
    clf.fit(X_scaled, y_train_num)

    return clf, scaler


anx_clf_num, _ = train_numeric_lr(Xn_train, yn_anx_train)
str_clf_num, _ = train_numeric_lr(Xn_train, yn_str_train)
dep_clf_num, _ = train_numeric_lr(Xn_train, yn_dep_train)

print("✅ SAFE XAI models ready.")


# -------------------------------------------------------
# Pipeline READY
# -------------------------------------------------------
print("\n🎉 Unified Mental Health Pipeline READY.\n")
