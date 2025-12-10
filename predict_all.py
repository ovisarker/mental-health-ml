import pandas as pd
from joblib import load
import os

# BASE PATH of this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using absolute paths
anxiety_model = load(os.path.join(BASE_DIR, "best_model_Anxiety_Label_LogisticRegression.joblib"))
stress_model = load(os.path.join(BASE_DIR, "best_model_Stress_Label_LogisticRegression.joblib"))
depression_model = load(os.path.join(BASE_DIR, "best_model_Depression_Label_CatBoost.joblib"))

print("✅ Models Loaded Successfully")


def preprocess_input(user_dict):
    df = pd.DataFrame([user_dict])
    df.columns = df.columns.str.strip()
    return df


def predict_all(user_input_dict):

    X = preprocess_input(user_input_dict)

    anx = anxiety_model.predict(X)[0]
    strx = stress_model.predict(X)[0]
    dep = depression_model.predict(X)[0]

    return {
        "Anxiety": "High Anxiety" if anx == 1 else "No Anxiety",
        "Stress": "High Stress" if strx == 1 else "No Stress",
        "Depression": "Depression Present" if dep == 1 else "No Depression"
    }
