import streamlit as st
import pandas as pd
import requests

# Import ML pipeline utilities from your own module
from unified_mental_health_pipeline import (
    predict_for_student,
    x_numeric,
    anx_clf_num,
    str_clf_num,
    dep_clf_num,
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML-Based Student Mental Health Assessment",
    layout="wide"
)


# ---------------------------------------------------------
# CHATBOT (HuggingFace Free API)
# ---------------------------------------------------------
def hf_chatbot(message: str) -> str:
    """
    Simple chatbot using HuggingFace Inference API (no API key required).
    Model: facebook/blenderbot-400M-distill
    """
    try:
        payload = {"inputs": message}
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            json=payload,
            timeout=20
        )
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return "Sorry, I could not understand that. Please try again with a shorter question."
    except Exception:
        return "Sorry, the chatbot is not responding right now. Please try again later."


# ---------------------------------------------------------
# BUILD STUDENT DATA DICTIONARY FOR ML
# ---------------------------------------------------------
def build_student_dict(
    age,
    gender,
    university,
    department,
    year,
    cgpa,
    scholarship,
    PSS,
    GAD,
    PHQ,
):
    """
    Build a dictionary in the same format as training data,
    so the unified ML pipeline can use it directly.
    """
    data = {
        "Age": age,
        "Gender": gender,
        "University": university,
        "Department": department,
        "Academic_Year": year,
        "Current_CGPA": cgpa,
        "waiver_or_scholarship": scholarship,
    }

    for i in range(10):
        data[f"PSS{i+1}"] = PSS[i]
    for i in range(7):
        data[f"GAD{i+1}"] = GAD[i]
    for i in range(9):
        data[f"PHQ{i+1}"] = PHQ[i]

    return data


# ---------------------------------------------------------
# XAI – Top Numeric Features (Coefficient-based)
# ---------------------------------------------------------
def get_top_features(model, cols, top_k=8) -> pd.DataFrame:
    """
    Return top_k features by absolute coefficient from a numeric-only LR model.
    """
    coefs = model.coef_[0]
    df = pd.DataFrame({"Feature": cols, "Coefficient": coefs})
    df["Abs"] = df["Coefficient"].abs()
    return df.sort_values("Abs", ascending=False).head(top_k)


# ---------------------------------------------------------
# SIMPLE SUGGESTION ENGINE
# ---------------------------------------------------------
def get_suggestions(anx_pred: int, str_pred: int, dep_pred: int):
    suggestions = []

    if anx_pred == 1:
        suggestions.append("• অতিরিক্ত দুশ্চিন্তা বা চিন্তা বাড়লে ছোট ছোট শ্বাস-প্রশ্বাসের ব্যায়াম চেষ্টা করতে পারেন।")
        suggestions.append("• পরীক্ষার আগে অতিরিক্ত ক্যাফেইন (চা/কফি/এনার্জি ড্রিংক) কমানো ভালো।")

    if str_pred == 1:
        suggestions.append("• বড় অ্যাসাইনমেন্টকে ছোট ছোট ধাপে ভাগ করে কাজ করলে চাপ কম অনুভূত হয়।")
        suggestions.append("• সাপ্তাহিক স্টাডি প্ল্যান ও রুটিন তৈরি করে কাজ করলে স্ট্রেস কমে।")

    if dep_pred == 1:
        suggestions.append("• প্রতিদিন নির্দিষ্ট সময়ে ঘুম, খাওয়া ও হালকা হাঁটা/ব্যায়ামের মতো basic routine বজায় রাখার চেষ্টা করুন।")
        suggestions.append("• খুব বেশি খারাপ লাগলে একা না থেকে trusted কারও সাথে কথা বলুন (বন্ধু, পরিবার বা কাউন্সেলর)।")

    if not suggestions:
        suggestions.append("• এখন বড় ধরনের ঝুঁকি দেখা যাচ্ছে না। তারপরও ভালো ঘুম, ব্যালান্সড ডায়েট আর নিয়মিত স্টাডি রুটিন বজায় রাখা জরুরি।")

    return suggestions


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.title("🧠 ML-Based Student Mental Health Assessment (Bangladesh)")
    st.write(
        "এই সিস্টেমটি বিশ্ববিদ্যালয় শিক্ষার্থীদের **Anxiety, Stress এবং Depression** "
        "ঝুঁকি অনুমান করার জন্য একটি Machine Learning ভিত্তিক গবেষণা টুল।"
    )
    st.info("⚠️ এটি কোনো চিকিৎসা নির্ণয় (diagnosis) নয়, শুধুমাত্র স্ক্রিনিং ও গবেষণার জন্য ব্যবহারযোগ্য।")

    st.markdown("---")

    # =====================================================
    # INPUT FORM
    # =====================================================
    with st.form("mh_form"):

        st.markdown("## 👤 Student Information")

        colA, colB = st.columns(2)
        with colA:
            age = st.number_input("Age", min_value=16, max_value=40, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            university = st.text_input("University")
            department = st.text_input("Department")
        with colB:
            year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
            cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            scholarship = st.selectbox("Scholarship / Waiver", ["Yes", "No"])

        st.markdown("---")

        # ---------------- STRESS (PSS-10) ----------------
        st.markdown("## 🟦 Stress Assessment (PSS-10)")
        st.caption("Scale: 0 = Never • 1 = Almost Never • 2 = Sometimes • 3 = Fairly Often • 4 = Very Often")

        PSS_Q = [
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
        PSS = [st.slider(f"PSS{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PSS_Q)]

        # ---------------- ANXIETY (GAD-7) ----------------
        st.markdown("## 🟩 Anxiety Assessment (GAD-7)")
        GAD_Q = [
            "Nervous or on edge due to study pressure",
            "Unable to stop worrying about study/future",
            "Trouble relaxing because of academic tension",
            "Easily annoyed or irritated",
            "Worrying too much about different things",
            "Restlessness – hard to sit still",
            "Feeling something bad might happen (results, exams etc.)",
        ]
        GAD = [st.slider(f"GAD{i+1}: {q}", 0, 4, 1) for i, q in enumerate(GAD_Q)]

        # ---------------- DEPRESSION (PHQ-9) --------------
        st.markdown("## 🟥 Depression Assessment (PHQ-9)")
        PHQ_Q = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed or hopeless",
            "Trouble falling or staying asleep / sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself / like a failure",
            "Trouble concentrating on study/reading/TV",
            "Moving/speaking so slowly or restlessly others notice",
            "Thoughts of self-harm or being better off dead (⚠ Serious)",
        ]
        PHQ = [st.slider(f"PHQ{i+1}: {q}", 0, 4, 1) for i, q in enumerate(PHQ_Q)]

        submitted = st.form_submit_button("🔍 Run ML Assessment")

    # =====================================================
    # PREDICTION & OUTPUT
    # =====================================================
    if submitted:
        # Build per-student data
        student_data = build_student_dict(
            age, gender, university, department, year, cgpa, scholarship,
            PSS, GAD, PHQ
        )

        # ML prediction from unified pipeline
        anx_pred, str_pred, dep_pred, main_issue = predict_for_student(student_data)

        # ---------------- RESULTS SUMMARY ----------------
        st.markdown("## ✅ ML Prediction Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Anxiety (ML)", "Present" if anx_pred == 1 else "Absent")
        with c2:
            st.metric("Stress (ML)", "Present" if str_pred == 1 else "Absent")
        with c3:
            st.metric("Depression (ML)", "Present" if dep_pred == 1 else "Absent")

        st.success(f"🧠 Dominant Mental-Health Issue: **{main_issue}**")

        # ---------------- SUGGESTIONS ----------------
        st.markdown("## 💡 General Wellbeing Suggestions")
        for s in get_suggestions(anx_pred, str_pred, dep_pred):
            st.write(s)

        # ---------------- EMERGENCY SUPPORT (BANGLADESH ONLY) ----------------
        st.markdown("## 🚨 জরুরি সহায়তা (Emergency Support)")

        # PHQ[8] → 9th item (self-harm thoughts)
        if PHQ[8] >= 3:
            st.error(
                "⚠ আপনার উত্তর অনুযায়ী আত্মহানি বা Self-harm প্রবণতার উচ্চ ঝুঁকি দেখা যাচ্ছে। "
                "এই অবস্থা অত্যন্ত সংবেদনশীল। অনুগ্রহ করে অবিলম্বে সাহায্য নিন।"
            )
        else:
            st.warning(
                "যদি কখনও মনে হয় আপনি নিজের জন্য ঝুঁকিপূর্ণ অবস্থায় আছেন, "
                "বা নিজেকে ক্ষতি করার চিন্তা আসে, একা থাকবেন না — অবিলম্বে কারও সাথে কথা বলুন "
                "বা সাহায্য নিন।"
            )

        st.write("🇧🇩 **বাংলাদেশ জাতীয় মানসিক সহায়তা হটলাইন:** Kaan Pete Roi — ☎️ **09612-119911**")
        st.write("🕒 সেবা: ২৪/৭ গোপনীয় মানসিক সহায়তা")

        st.markdown("---")

        # ---------------- XAI SECTION ----------------
        st.markdown("## 🔬 Explainable AI (Top Influential Numeric Features)")
        st.write(
            "এই টেবিলগুলো দেখায়, numeric Logistic Regression model অনুযায়ী কোন feature (question score ইত্যাদি) "
            "Anxiety, Stress এবং Depression prediction-এ সবচেয়ে বেশি প্রভাব ফেলেছে।"
        )

        try:
            top_anx = get_top_features(anx_clf_num, x_numeric.columns)
            top_str = get_top_features(str_clf_num, x_numeric.columns)
            top_dep = get_top_features(dep_clf_num, x_numeric.columns)

            colX, colY, colZ = st.columns(3)
            with colX:
                st.write("### Anxiety – Top Features")
                st.dataframe(top_anx[["Feature", "Coefficient"]])
            with colY:
                st.write("### Stress – Top Features")
                st.dataframe(top_str[["Feature", "Coefficient"]])
            with colZ:
                st.write("### Depression – Top Features")
                st.dataframe(top_dep[["Feature", "Coefficient"]])
        except Exception as e:
            st.warning(f"XAI গণনার সময় সমস্যা হয়েছে: {e}")

    # =====================================================
    # CHATBOT SECTION
    # =====================================================
    st.markdown("---")
    st.header("💬 Mental Health Chatbot (Experimental)")

    st.write(
        "এখানে আপনি সাধারণভাবে anxiety, stress, depression বা মানসিক স্বাস্থ্য নিয়ে কিছু জানতে চাইলে লিখতে পারেন। "
        "চ্যাটবটটি একটি ফ্রি public language model ব্যবহার করে। এটি পেশাদার চিকিৎসার বিকল্প নয়।"
    )

    user_msg = st.text_input("এখানে আপনার প্রশ্ন লিখুন...")

    if user_msg:
        reply = hf_chatbot(user_msg)
        st.write("🤖:", reply)


if __name__ == "__main__":
    main()