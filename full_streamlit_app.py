import streamlit as st
import pandas as pd
import requests

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
        "subtitle": "এই সিস্টেমটি Anxiety, Stress এবং Depression—Machine Learning দিয়ে মূল্যায়ন করে।",
        "student_info": "👤 শিক্ষার্থীর তথ্য",
        "stress": "🟦 স্ট্রেস মূল্যায়ন (PSS-10)",
        "anxiety": "🟩 উদ্বেগ/Anxiety মূল্যায়ন (GAD-7)",
        "depression": "🟥 বিষণ্নতা/Depression মূল্যায়ন (PHQ-9)",
        "run": "🔍 মূল্যায়ন চালান",
        "ml_results": "✅ মেশিন লার্নিং ভিত্তিক ফলাফল",
        "dominant": "🧠 প্রধান মানসিক স্বাস্থ্য সমস্যা:",
        "risk_level": "📊 ঝুঁকির স্তর (Risk Levels; স্কোর দেখানো হচ্ছে না)",
        "suggestions": "💡 সাধারণ পরামর্শ",
        "emergency": "🚨 জরুরি সহায়তা",
        "hotline": "🇧🇩 বাংলাদেশ মানসিক সহায়তা হটলাইন: Kaan Pete Roi — ☎️ 09612-119911",
        "xai": "🔬 Explainable AI (সর্বাধিক প্রভাবশালী ফিচারগুলো)",
        "chatbot_header": "💬 মানসিক স্বাস্থ্য চ্যাটবট",
        "chatbot_placeholder": "এখানে আপনার প্রশ্ন লিখুন...",
        "anxiety_label": "Anxiety",
        "stress_label": "Stress",
        "depression_label": "Depression",
        "no_risk_suggestion": "• বড় ধরনের ঝুঁকি দেখা যাচ্ছে না। সুস্থ রুটিন, ঘুম এবং স্টাডি ব্যালান্স বজায় রাখুন।",
        "self_harm_high": "⚠ আপনার উত্তর অনুযায়ী Self-harm বা আত্মহানির উচ্চ ঝুঁকি দেখা যাচ্ছে। অনুগ্রহ করে অবিলম্বে কাউকে জানিয়ে সাহায্য নিন।",
        "self_harm_generic": "যদি কখনও মনে হয় আপনি নিজের জন্য ঝুঁকিপূর্ণ অবস্থায় আছেন, দয়া করে একা থাকবেন না — trusted কারও সাথে কথা বলুন বা হেল্পলাইনে যোগাযোগ করুন।",
    },
    "English": {
        "title": "🧠 ML-Based Student Mental Health Assessment",
        "subtitle": "This system predicts Anxiety, Stress and Depression using Machine Learning.",
        "student_info": "👤 Student Information",
        "stress": "🟦 Stress Assessment (PSS-10)",
        "anxiety": "🟩 Anxiety Assessment (GAD-7)",
        "depression": "🟥 Depression Assessment (PHQ-9)",
        "run": "🔍 Run Assessment",
        "ml_results": "✅ ML Prediction Results",
        "dominant": "🧠 Dominant Mental-Health Issue:",
        "risk_level": "📊 Risk Levels (Scores not displayed)",
        "suggestions": "💡 Suggestions",
        "emergency": "🚨 Emergency Support",
        "hotline": "🇧🇩 Bangladesh Hotline: Kaan Pete Roi — ☎️ 09612-119911",
        "xai": "🔬 Explainable AI (Top Influential Features)",
        "chatbot_header": "💬 Mental Health Chatbot",
        "chatbot_placeholder": "Type your question here...",
        "anxiety_label": "Anxiety",
        "stress_label": "Stress",
        "depression_label": "Depression",
        "no_risk_suggestion": "• No major risks detected. Maintain good sleep, food, and study-life balance.",
        "self_harm_high": "⚠ Your responses indicate high self-harm risk. Please seek help immediately.",
        "self_harm_generic": "If you ever feel unsafe or think about self-harm, please talk to someone you trust or call the helpline.",
    },
}

# -----------------------------------------------------
# CHATBOT (HuggingFace FLAN-T5, free, no key)
# -----------------------------------------------------
def chatbot_reply(message: str) -> str:
    try:
        payload = {"inputs": message}
        r = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            json=payload,
            timeout=25,
        )
        out = r.json()
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"].strip()
        return "I'm not sure how to answer that. Please try again with a shorter question."
    except Exception:
        return "Chatbot is temporarily unavailable. Please try again later."

# -----------------------------------------------------
# XAI helper
# -----------------------------------------------------
def top_features(model, cols, k=6):
    coefs = model.coef_[0]
    df = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
    df["Abs"] = df["Coefficient"].abs()
    return df.sort_values("Abs", ascending=False).head(k)

# -----------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Student Mental Health ML App", layout="wide")

# Language selector
lang = st.sidebar.selectbox("🌐 Language / ভাষা", ["Bangla", "English"])
T = LANG[lang]

# Title & intro
st.title(T["title"])
st.write(T["subtitle"])
st.info("⚠ Research tool only — not a clinical diagnosis.")
st.markdown("---")

# -------------------- Student Info --------------------
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
scholarship = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

st.markdown("---")

# -------------------- STRESS (PSS-10) --------------------
st.header(T["stress"])
PSS_LABELS = [
    "Upset due to academic issues",
    "Unable to control academic matters",
    "Nervous or stressed from academics",
    "Could not cope with tasks/exams",
    "Felt confident handling problems (Reverse)",
    "Felt things going well academically (Reverse)",
    "Controlled irritation from academics (Reverse)",
    "Academic performance satisfactory (Reverse)",
    "Felt anger due to poor academic outcomes",
    "Academic difficulties piled up beyond control",
]
PSS = [st.slider(f"PSS{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PSS_LABELS)]

# -------------------- ANXIETY (GAD-7) --------------------
st.header(T["anxiety"])
GAD_LABELS = [
    "Nervous or on edge",
    "Unable to stop worrying",
    "Trouble relaxing",
    "Easily annoyed or irritated",
    "Worrying too much",
    "Restlessness",
    "Feeling something bad might happen",
]
GAD = [st.slider(f"GAD{i+1}: {q}", 0, 4, 1) for i, q in enumerate(GAD_LABELS)]

# -------------------- DEPRESSION (PHQ-9) --------------------
st.header(T["depression"])
PHQ_LABELS = [
    "Little interest or pleasure",
    "Feeling down or hopeless",
    "Sleep problems",
    "Low energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself",
    "Trouble concentrating",
    "Slow or restless movement",
    "Self-harm thoughts (⚠ Serious)",
]
PHQ = [st.slider(f"PHQ{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PHQ_LABELS)]

# -----------------------------------------------------
# Run Assessment
# -----------------------------------------------------
if st.button(T["run"]):
    # Build student dict for pipeline
    student = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship,
    }
    for i in range(10):
        student[f"PSS{i+1}"] = PSS[i]
    for i in range(7):
        student[f"GAD{i+1}"] = GAD[i]
    for i in range(9):
        student[f"PHQ{i+1}"] = PHQ[i]

    # ML predictions
    anx_pred, str_pred, dep_pred, dominant_issue = predict_for_student(student)
    risk_levels = risk_levels_for_student(student)

    # ---------- ML Prediction Results ----------
    st.markdown("## " + T["ml_results"])
    c1, c2, c3 = st.columns(3)
    c1.metric(T["anxiety_label"], "Present" if anx_pred else "Absent")
    c2.metric(T["stress_label"], "Present" if str_pred else "Absent")
    c3.metric(T["depression_label"], "Present" if dep_pred else "Absent")

    st.success(f"{T['dominant']} **{dominant_issue}**")

    # ---------- Risk Levels (text only) ----------
    st.markdown("## " + T["risk_level"])
    r1, r2, r3 = st.columns(3)
    r1.info(f"{T['stress_label']}: {risk_levels['Stress']}")
    r2.info(f"{T['anxiety_label']}: {risk_levels['Anxiety']}")
    r3.info(f"{T['depression_label']}: {risk_levels['Depression']}")

    # ---------- Suggestions ----------
    st.markdown("## " + T["suggestions"])
    any_flag = False
    if anx_pred:
        any_flag = True
        st.write("• Try breathing/grounding exercises; reduce overthinking around exams.")
    if str_pred:
        any_flag = True
        st.write("• Use a simple weekly plan and break assignments into small chunks.")
    if dep_pred:
        any_flag = True
        st.write("• Maintain a basic routine (sleep, food, light activity) and talk to someone you trust.")
    if not any_flag:
        st.write(T["no_risk_suggestion"])

    # ---------- Emergency Support ----------
    st.markdown("## " + T["emergency"])
    if PHQ[8] >= 3:
        st.error(T["self_harm_high"])
    else:
        st.warning(T["self_harm_generic"])
    st.write(T["hotline"])

    st.markdown("---")

    # ---------- XAI ----------
    st.header(T["xai"])
    colA, colB, colC = st.columns(3)
    colA.write("### " + T["anxiety_label"])
    colA.dataframe(top_features(anx_clf_num, x_numeric.columns))
    colB.write("### " + T["stress_label"])
    colB.dataframe(top_features(str_clf_num, x_numeric.columns))
    colC.write("### " + T["depression_label"])
    colC.dataframe(top_features(dep_clf_num, x_numeric.columns))

st.markdown("---")

# -------------------- Chatbot --------------------
st.header(T["chatbot_header"])
msg = st.text_input(T["chatbot_placeholder"])
if msg:
    st.write("🤖:", chatbot_reply(msg))