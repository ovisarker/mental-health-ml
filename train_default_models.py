"""
train_default_models.py
=======================

This script trains baseline machine‑learning models for predicting student
mental‑health conditions (anxiety, stress and depression) from a processed
dataset (Processed.csv).  It is intended as a fallback tool to generate
joblib model files when pre‑trained models are missing.  The models
produced here approximate the original project’s best models: Logistic
Regression for Anxiety and Stress, and a tree‑based model (RandomForest)
for Depression.  After training, the models are saved into the joblib
filenames used by the Streamlit app:  

* best_model_Anxiety_Label_LogisticRegression.joblib  
* best_model_Stress_Label_LogisticRegression.joblib  
* best_model_Depression_Label_RandomForest.joblib  

To run this script from the repository root:

```
python train_default_models.py
```

Make sure the file `Processed.csv` is present in the same directory.

Note: this code is provided as a convenience; it may not reproduce the
exact performance of your original models but serves to generate
functional stand‑in models.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Constants for filenames
ANXIETY_MODEL_NAME = "best_model_Anxiety_Label_LogisticRegression.joblib"
STRESS_MODEL_NAME = "best_model_Stress_Label_LogisticRegression.joblib"
DEPRESSION_MODEL_NAME = "best_model_Depression_Label_RandomForest.joblib"

def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load the processed CSV and create binary labels for the three conditions.

    Args:
        csv_path: path to Processed.csv

    Returns:
        X: DataFrame of features
        y_anxiety: binary Series for anxiety
        y_stress: binary Series for stress
        y_depression: binary Series for depression
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # remove spaces from column names

    # Create binary labels for depression
    if "Depression Label" in df.columns:
        df["Depression_Binary"] = df["Depression Label"].apply(
            lambda x: 0 if str(x).strip().lower() == "no depression" else 1
        )
    elif "Depression Value" in df.columns:
        # if numeric score is available, define a simple threshold
        df["Depression_Binary"] = df["Depression Value"].apply(lambda x: 0 if x <= 4 else 1)
    else:
        raise ValueError("Depression label/score columns not found in dataset.")

    # Create binary labels for anxiety using GAD-7 total
    gad_cols = [col for col in df.columns if col.startswith("GAD")]
    if gad_cols:
        df["GAD_Total"] = df[gad_cols].sum(axis=1)
        df["Anxiety_Binary"] = df["GAD_Total"].apply(lambda x: 0 if x <= 4 else 1)
    else:
        raise ValueError("GAD columns not found in dataset.")

    # Create binary labels for stress using PSS total
    pss_cols = [col for col in df.columns if col.startswith("PSS")]
    if pss_cols:
        df["PSS_Total"] = df[pss_cols].sum(axis=1)
        df["Stress_Binary"] = df["PSS_Total"].apply(lambda x: 0 if x <= 13 else 1)
    else:
        raise ValueError("PSS columns not found in dataset.")

    # Prepare feature matrix by dropping target and helper columns
    drop_columns = [
        "Stress Value", "Stress Label", "Anxiety Value", "Anxiety Label",
        "Depression Value", "Depression Label", "GAD_Total", "PSS_Total",
        "Depression_Binary", "Anxiety_Binary", "Stress_Binary"
    ]
    # Only drop if present
    drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=drop_columns)
    y_anxiety = df["Anxiety_Binary"]
    y_stress = df["Stress_Binary"]
    y_depression = df["Depression_Binary"]

    return X, y_anxiety, y_stress, y_depression


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Construct a preprocessing pipeline for categorical and numeric features."""
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor


def train_and_save_models(csv_path: str) -> None:
    """Train baseline models for anxiety, stress and depression, then save them."""
    X, y_anxiety, y_stress, y_depression = load_and_prepare_data(csv_path)

    preprocessor = build_preprocessing_pipeline(X)

    # Train model for anxiety (Logistic Regression)
    anxiety_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    anxiety_pipeline.fit(X, y_anxiety)
    joblib.dump(anxiety_pipeline, ANXIETY_MODEL_NAME)
    print(f"Saved anxiety model → {ANXIETY_MODEL_NAME}")

    # Train model for stress (Logistic Regression)
    stress_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    stress_pipeline.fit(X, y_stress)
    joblib.dump(stress_pipeline, STRESS_MODEL_NAME)
    print(f"Saved stress model → {STRESS_MODEL_NAME}")

    # Train model for depression (Random Forest)
    depression_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    depression_pipeline.fit(X, y_depression)
    joblib.dump(depression_pipeline, DEPRESSION_MODEL_NAME)
    print(f"Saved depression model → {DEPRESSION_MODEL_NAME}")


def main():
    csv_file = "Processed.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot find {csv_file}. Make sure Processed.csv exists in the current directory.")
    train_and_save_models(csv_file)


if __name__ == "__main__":
    main()