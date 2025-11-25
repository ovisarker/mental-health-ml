import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
import os

# ------------------------------------------------------------------
# PAGE SETTINGS & GLOBAL STYLES
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Mental Health Assessment",
    page_icon="🧠",
    layout="wide"
)

# Global Medical UI Styling
st.markdown("""
<style>
body { background-color:#F4F7FB; color:#111827; }
h1,h2,h3,h4,h5,h6 { color:#111827 !important; font-weight:700 !important; }

.main-card {
    background:white;
    padding:26px;
    border-radius:18px;
    box-shadow:0 8px 18px rgba(0,0,0,0.06);
    margin-bottom:22px;
}

.section-card {
    background:white;
    padding:20px;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.05);
}

.scale-card {
    background:#E8F2FF;
    border:1px solid #C5DAFF;
    padding:16px;
    border-radius:14px;
    font-size:0.9rem;
}

.badge {
    padding:8px 14px;
    border-radius:999px;
    font-weight:600;
    font-size:0.9rem;
    display:inline-block;
    margin-top:10px;
}

.badge-low { background:#DCFCE7; color:#166534; }
.badge-mod { background:#FEF9C3; color:#854D0E; }
.badge-high { background:#FFEDD5; color:#9A3412; }
.badge-crit { background:#FEE2E2; color:#991B1B; }

.lang-toggle {
    background:#E0E7FF; padding:6px 12px; border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MULTI-LANG LANGUAGE TOGGLE
# ------------------------------------------------------------------
lang = st.sidebar.selectbox("Language:", ["English", "Bangla"])

T = {
    "English": {
        "title": "AI-based Mental Health Assessment",
        "choose": "What would you like to assess?",
        "screening_form": "Screening Form",
        "instructions": "Rate each statement from 1 (lowest) to 5 (highest).",
        "predict": "🔍 Predict Mental Health Status",
        "risk": "Risk Level",
        "suggested": "Suggested Actions",
        "disclaimer": "⚠ This tool does not replace professional diagnosis.",
        "emergency": "If you feel unsafe or in crisis, contact emergency services immediately."
    },
    "Bangla": {
        "title": "এআই-ভিত্তিক মানসিক স্বাস্থ্যের মূল্যায়ন",
        "choose": "আপনি কোনটি মূল্যায়ন করতে চান?",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "প্রতিটি প্রশ্নের উত্তর দিন ১ (সবচেয়ে কম) থেকে ৫ (সবচেয়ে বেশি)।",
        "predict": "🔍 মানসিক স্বাস্থ্য পূর্বাভাস দিন",
        "risk": "ঝুঁকির স্তর",
        "suggested": "প্রস্তাবিত পদক্ষেপ",
        "disclaimer": "⚠ এই টুল পেশাদার মানসিক স্বাস্থ্য নির্ণয়ের বিকল্প নয়।",
        "emergency": "আপনি যদি ঝুঁকিতে অনুভব করেন, জরুরি পরিষেবার সাথে যোগাযোগ করুন।"
    }
}[lang]

# ------------------------------------------------------------------
# CLINICAL SCORING LOGIC
# ------------------------------------------------------------------
def score_responses(values, target):
    """Return severity label based on standardized clinical scoring."""

    # GAD-7 (Anxiety)
    if target == "Anxiety":
        scaled = [v-1 for v in values]  # 0–3
        total = sum(scaled)
        if total <= 4: lvl = "Minimal"
        elif total <= 9: lvl = "Mild"
        elif total <= 14: lvl = "Moderate"
        else: lvl = "Severe"
        return f"{lvl} Anxiety"

    # PHQ-9 (Depression)
    if target == "Depression":
        scaled = [v-1 for v in values]
        total = sum(scaled)  # 0–27
        if total <= 4: lvl = "Minimal"
        elif total <= 9: lvl = "Mild"
        elif total <= 14: lvl = "Moderate"
        else: lvl = "Severe"
        return f"{lvl} Depression"

    # PSS-10 (Stress)
    scaled = [v-1 for v in values]  # 0–4
    total = sum(scaled)            # 0–40
    if total <= 13: lvl = "Minimal"
    elif total <= 26: lvl = "Moderate"
    else: lvl = "Severe"
    return f"{lvl} Stress"

def map_risk(label):
    ll = label.lower()
    if "minimal" in ll: return "Low"
    if "mild" in ll: return "Moderate"
    if "moderate" in ll: return "High"
    if "severe" in ll: return "Critical"
    return "Unknown"

# ------------------------------------------------------------------
# QUESTIONS
# ------------------------------------------------------------------
QUESTIONS = {
    "Anxiety": [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it is hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen",
    ],
    "Stress": [
        "Upset because of unexpected events",
        "Unable to control important things in life",
        "Felt nervous and stressed",
        "Confident about handling problems",
        "Things going your way",
        "Could not cope with all the things you had to do",
        "Able to control irritations",
        "Felt on top of things",
        "Angry because things were out of control",
        "Felt difficulties piling up too high",
    ],
    "Depression": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble sleeping or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or failure",
        "Trouble concentrating",
        "Moving/speaking slowly / restlessness",
        "Thoughts of self-harm",
    ],
}

# Scale meaning per target
SCALE = {
    "Anxiety": ["Not at all", "Several days", "Half the days", "Nearly every day", "Almost always"],
    "Depression": ["Not at all", "Several days", "Half the days", "Nearly every day", "Almost always"],
    "Stress": ["Never", "Almost never", "Sometimes", "Often", "Very often"]
}

# ------------------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🧩 Screening", "📊 Dashboard"])

# ------------------------------------------------------------------
# SCREENING PAGE
# ------------------------------------------------------------------
if page == "🧩 Screening":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.header(T["title"])
    st.write(f"**{T['disclaimer']}**")
    st.write(f"🚨 *{T['emergency']}*")

    target = st.selectbox(T["choose"], ["Anxiety", "Stress", "Depression"])

    st.subheader(f"🧾 {target} {T['screening_form']}")

    # layout row
    left, right = st.columns([3,1])

    # RIGHT COLUMN = SCALE CARD
    with right:
        st.markdown("<div class='scale-card'>", unsafe_allow_html=True)
        st.markdown("### Scale Meaning (1–5)")
        for i, s in enumerate(SCALE[target], 1):
            st.write(f"**{i} — {s}**")
        st.markdown("</div>", unsafe_allow_html=True)

    # LEFT COLUMN = QUESTIONS
    responses = []
    with left:
        for i, q in enumerate(QUESTIONS[target]):
            responses.append(
                st.slider(q, 1, 5, 3, key=f"{target}{i}")
            )

    if st.button(T["predict"]):
        label = score_responses(responses, target)
        risk = map_risk(label)

        badge_class = {
            "Low":"badge-low",
            "Moderate":"badge-mod",
            "High":"badge-high",
            "Critical":"badge-crit"
        }[risk]

        st.markdown(f"<div class='badge {badge_class}'>🎯 {label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='badge {badge_class}'>🩺 {T['risk']}: {risk}</div>", unsafe_allow_html=True)

        sug = {
            "Low":"Maintain a healthy routine and sleep schedule.",
            "Moderate":"Practice mindfulness, journaling, social support.",
            "High":"Reduce stress exposure and consult a professional.",
            "Critical":"Seek immediate professional help and crisis support."
        }[risk]

        st.write(f"### {T['suggested']}: {sug}")

        # save log
        log_row = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target": target,
            "label": label,
            "risk": risk
        }
        df = pd.DataFrame([log_row])
        if os.path.exists("log.csv"):
            df.to_csv("log.csv", mode="a", header=False, index=False)
        else:
            df.to_csv("log.csv", index=False)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Analytics Dashboard")

    if not os.path.exists("log.csv"):
        st.warning("No screening results yet.")
    else:
        df = pd.read_csv("log.csv")
        st.dataframe(df.tail(20))

        # Risk distribution
        st.subheader("Risk Distribution")
        tiers = df["risk"].value_counts()

        chart = pd.DataFrame({
            "Risk": tiers.index,
            "Count": tiers.values
        })
        st.bar_chart(chart, x="Risk", y="Count")

        df["datetime"] = pd.to_datetime(df["datetime"])
        trend = df.groupby(df["datetime"].dt.date).size()
        st.subheader("Screenings Over Time")
        st.line_chart(trend)

    st.markdown("</div>", unsafe_allow_html=True)
