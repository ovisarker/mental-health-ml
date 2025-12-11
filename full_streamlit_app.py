import streamlit as st
import pandas as pd
import requests

# Import ML pipeline utilities
from unified_mental_health_pipeline import (
    predict_for_student,
    risk_levels_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num,
)


# -----------------------------------------------------
# LANGUAGE PACK
# -----------------------------------------------------
LANG = {
    "Bangla": {
        "title": "🧠 শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন (ML ভিত্তিক)",
        "subtitle": "এই সিস্টেমটি Anxiety, Stress এবং Depression—ML ভিত্তিতে মূল্যায়ন করে।",
        "student_info": "👤 শিক্ষার্থীর তথ্য",
        "stress": "🟦 স্ট্রেস মূল্যায়ন (PSS-10)",
        "anxiety": "🟩 উৎকণ্ঠা মূল্যায়ন (GAD-7)",
        "depression": "🟥 বিষণ্নতা মূল্যায়ন (PHQ-9)",
        "run": "🔍 মূল্যায়ন চালান",
        "ml_results": "✅ মেশিন লার্নিং ফলাফল",
        "dominant": "🧠 প্রধান মানসিক স্বাস্থ্য সমস্যা:",
        "risk_level": "📊 ঝুঁকির স্তর (Risk Level; No Scores Shown)",
        "suggestions": "💡 পরামর্শ",
        "emergency": "🚨 জরুরি সহায়তা",
        "hotline": "🇧🇩 বাংলাদেশ হটলাইন: Kaan Pete Roi — ☎️ 09612-119911",
        "xai": "🔬 Explainable AI (সবচেয়ে প্রভাবশালী ফিচার)",
        "chatbot_header": "💬 মানসিক স্বাস্থ্য চ্যাটবট",
        "chatbot_placeholder": "এখানে আপনার প্রশ্ন লিখুন...",
    },

    "English": {
        "title": "🧠 ML-Based Student Mental Health Assessment",
        "subtitle": "This system evaluates Anxiety, Stress & Depression using Machine Learning.",
        "student_info": "👤 Student Information",
        "stress": "🟦 Stress Assessment (PSS-10)",
        "anxiety": "🟩 Anxiety Assessment (GAD-7)",
        "depression": "🟥 Depression Assessment (PHQ-9)",
        "run": "🔍 Run Assessment",
        "ml_results": "✅ ML Prediction Results",
        "dominant": "🧠 Dominant Mental-Health Issue:",
        "risk_level": "📊 Risk Levels (No Numeric Scores Shown)",
        "suggestions": "💡 Suggestions",
        "emergency": "🚨 Emergency Support",
        "hotline": "🇧🇩 Bangladesh Hotline: Kaan Pete Roi — ☎️ 09612-119911",
        "xai": "🔬 Explainable AI (Top Influential Features)",
        "chatbot_header": "💬 Mental Health Chatbot",
        "chatbot_placeholder": "Type your question here...",
    }
}


# -----------------------------------------------------
# CHATBOT (Stable Free Model)
# -----------------------------------------------------
def chatbot_reply(message):
    try:
        payload = {"inputs": message}
        r = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            json=payload,
            timeout=30
        )
        output = r.json()
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        return "I could not process that. Try again."
    except:
        return "Chatbot is temporarily unavailable."


# -----------------------------------------------------
# TOP FEATURES FOR XAI
# -----------------------------------------------------
def top_features(model, cols, k=6):
    coef = model.coef_[0]
    df = pd.DataFrame({"Feature": cols, "Coef": coef})
    df["Abs"] = df["Coef"].abs()
    return df.sort_values("Abs", ascending=False).head(k)


# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
st.set_page_config(page_title="Mental Health Assessment", layout="wide")

# Language selector
lang = st.sidebar.selectbox("🌐 Language", ["Bangla", "English"])
T = LANG[lang]

# Title
st.title(T["title"])
st.write(T["subtitle"])
st.markdown("---")

# -------------------- STUDENT INFO --------------------
st.header(T["student_info"])

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 16, 40, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    university = st.text_input("University")
with col2:
    department = st.text_input("Department")
    year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
    cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
scholar = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

st.markdown("---")

# -------------------- Stress --------------------
st.header(T["stress"])
PSS_LABELS = [
    "Upset due to academic issues",
    "Unable to control academic matters",
    "Nervous or stressed",
    "Could not cope with tasks",
    "Felt confident (Reverse)",
    "Felt things going well (Reverse)",
    "Controlled irritation (Reverse)",
    "Academic performance satisfactory (Reverse)",
    "Anger due to bad outcomes",
    "Difficulties piling up"
]
PSS = [st.slider(f"PSS{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PSS_LABELS)]

# -------------------- Anxiety --------------------
st.header(T["anxiety"])
GAD_LABELS = [
    "Nervous or on edge",
    "Unable to stop worrying",
    "Trouble relaxing",
    "Easily annoyed",
    "Worrying too much",
    "Restlessness",
    "Fear something bad might happen"
]
GAD = [st.slider(f"GAD{i+1}: {q}", 0, 4, 1) for i, q in enumerate(GAD_LABELS)]

# -------------------- Depression --------------------
st.header(T["depression"])
PHQ_LABELS = [
    "Little interest",
    "Feeling down",
    "Sleep issues",
    "Low energy",
    "Poor appetite",
    "Feeling bad about yourself",
    "Trouble concentrating",
    "Slow / restless movement",
    "Self-harm thoughts (⚠ Serious)"
]
PHQ = [st.slider(f"PHQ{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PHQ_LABELS)]

if st.button(T["run"]):

    # Build student dict
    student = {
        "Age": age, "Gender": gender, "University": university,
        "Department": department, "Academic_Year": year,
        "Current_CGPA": cgpa, "waiver_or_scholarship": scholar
    }
    for i in range(10): student[f"PSS{i+1}"] = PSS[i]
    for i in range(7): student[f"GAD{i+1}"] = GAD[i]
    for i in range(9): student[f"PHQ{i+1}"] = PHQ[i]

    anx, stress, dep, dominant = predict_for_student(student)
    risk = risk_levels_for_student(student)

    # ------------------ ML OUTPUT ------------------
    st.markdown("## " + T["ml_results"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Anxiety", "Present" if anx else "Absent")
    c2.metric("Stress", "Present" if stress else "Absent")
    c3.metric("Depression", "Present" if dep else "Absent")

    st.success(f"{T['dominant']} **{dominant}**")

    # ------------------ RISK LEVELS ------------------
    st.markdown("## " + T["risk_level"])
    r1, r2, r3 = st.columns(3)
    r1.info(f"Stress: {risk['Stress']}")
    r2.info(f"Anxiety: {risk['Anxiety']}")
    r3.info(f"Depression: {risk['Depression']}")

    # ------------------ SUGGESTIONS ------------------
    st.markdown("## " + T["suggestions"])
    if anx: st.write("• Try breathing exercises and grounding techniques.")
    if stress: st.write("• Break tasks into smaller steps and follow a routine.")
    if dep: st.write("• Talk to a trusted person and maintain daily routine.")
    if not any([anx, stress, dep]):
        st.write("• No major risks detected. Maintain healthy lifestyle.")

    # ------------------ EMERGENCY ------------------
    st.markdown("## " + T["emergency"])
    if PHQ[8] >= 3:
        st.error("⚠ High self-harm risk detected. Seek immediate help.")
    st.write(T["hotline"])

    st.markdown("---")

    # ------------------ XAI ------------------
    st.header(T["xai"])
    colA, colB, colC = st.columns(3)
    colA.write("### Anxiety")
    colA.dataframe(top_features(anx_clf_num, x_numeric.columns))
    colB.write("### Stress")
    colB.dataframe(top_features(str_clf_num, x_numeric.columns))
    colC.write("### Depression")
    colC.dataframe(top_features(dep_clf_num, x_numeric.columns))

st.markdown("---")

# ------------------ CHATBOT ------------------
st.header(T["chatbot_header"])
msg = st.text_input(T["chatbot_placeholder"])
if msg:
    st.write("🤖:", chatbot_reply(msg))