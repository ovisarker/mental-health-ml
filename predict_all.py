# ==============================================
# predict_all.py
# Unified Prediction Helper for
# Anxiety, Stress, Depression
# ==============================================

import pandas as pd
from joblib import load

# -----------------------
#  Load trained models
# -----------------------
# Using YOUR existing filenames
anxiety_model = load("best_model_Anxiety_Label_LogisticRegression.joblib")
stress_model = load("best_model_Stress_Label_LogisticRegression.joblib")
depression_model = load("best_model_Depression_Label_CatBoost.joblib")

print("✅ Models loaded successfully!")


# -------------------------------------------------------------
#  Preprocess single user input → DataFrame
# -------------------------------------------------------------
def preprocess_input(user_dict: dict) -> pd.DataFrame:
    """
    Convert user input dictionary to a one-row DataFrame.

    IMPORTANT:
    - Keys in user_dict must match the feature names used in training:
      Age, Gender, University, Department, Academic_Year,
      Current_CGPA, waiver_or_scholarship,
      PSS1..PSS10, GAD1..GAD7, PHQ1..PHQ9
    """
    df = pd.DataFrame([user_dict])
    df.columns = df.columns.str.strip()  # remove unwanted spaces
    return df


# -------------------------------------------------------------
#  Unified Predict Function
# -------------------------------------------------------------
def predict_all(user_input_dict: dict) -> dict:
    """
    Runs all three ML models and returns predictions.

    OUTPUT example:
        {
            "Anxiety": "High Anxiety" / "No Anxiety",
            "Stress": "High Stress" / "No Stress",
            "Depression": "Depression Present" / "No Depression"
        }
    """

    # 1. Convert dict → DataFrame
    X_new = preprocess_input(user_input_dict)

    # 2. Run predictions
    anx_pred = anxiety_model.predict(X_new)[0]
    str_pred = stress_model.predict(X_new)[0]
    dep_pred = depression_model.predict(X_new)[0]

    # 3. Human-readable output
    result = {
        "Anxiety": "High Anxiety" if anx_pred == 1 else "No Anxiety",
        "Stress": "High Stress" if str_pred == 1 else "No Stress",
        "Depression": "Depression Present" if dep_pred == 1 else "No Depression"
    }

    return result


# -------------------------------------------------------------
#  Manual Test (Optional)
# -------------------------------------------------------------
if __name__ == "__main__":
    test_user = {
        "Age": "18-22",
        "Gender": "Male",
        "University": "Sample University",
        "Department": "CSE",
        "Academic_Year": "2nd Year",
        "Current_CGPA": "3.00-3.50",
        "waiver_or_scholarship": "No",

        # PSS
        "PSS1": 2, "PSS2": 3, "PSS3": 1, "PSS4": 2, "PSS5": 1,
        "PSS6": 2, "PSS7": 1, "PSS8": 2, "PSS9": 3, "PSS10": 2,

        # GAD
        "GAD1": 1, "GAD2": 1, "GAD3": 0, "GAD4": 1,
        "GAD5": 0, "GAD6": 1, "GAD7": 0,

        # PHQ
        "PHQ1": 1, "PHQ2": 0, "PHQ3": 1, "PHQ4": 1, "PHQ5": 0,
        "PHQ6": 0, "PHQ7": 1, "PHQ8": 1, "PHQ9": 0,
    }

    print("Test prediction:\n", predict_all(test_user))
