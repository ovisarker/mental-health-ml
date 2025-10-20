# 🧠 AI-based Mental Health Detection & Support System

A research-driven web app that predicts **Anxiety**, **Stress**, and **Depression** levels among university students using machine learning and explainable AI.

---

## 🌿 Overview
This application uses pre-trained ML models (Logistic Regression, SVM, RandomForest, LightGBM, XGBoost, CatBoost, AdaBoost) trained on an **Extended Mental Health Dataset (2025)** collected and curated by **Ovi Sarker** and **BM Sabbir Hossen Riad**.

Each prediction is paired with:
- **Risk Level Classification** (Low / Mild / Moderate / Severe)
- **Intervention Suggestions**
- **Explainable AI (XAI)** visualizations
- **AI Wellness Assistant** for general guidance

---

## ⚙️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** scikit-learn, LightGBM, XGBoost, CatBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit Cloud
- **Language:** Python 3.12

---

## 📦 Files
| File | Description |
|------|--------------|
| `app.py` | Main Streamlit web app |
| `requirements.txt` | Dependencies for Streamlit Cloud |
| `final_anxiety_model.joblib` | Trained anxiety model |
| `final_stress_model.joblib` | Trained stress model |
| `final_depression_model.joblib` | Trained depression model |
| `final_*_encoder.joblib` | Label encoders for models |

---

## 🔍 How to Run Locally

1️⃣ Clone the repo:
```bash
git clone https://github.com/ovisarker/mental-health-ml.git
cd mental-health-ml
