################################################################################
# Minimal Mental Health Assessment (Thesis Version)
# Clean • Simple • Only Screening + Dashboard
# Supports: Anxiety, Stress, Depression
# Language: English + Bangla
#
# Developed by Team Dual Core (© 2025)
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from datetime import datetime

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------------------------------------------------------
# SIMPLE SAFE CSV LOADER
# ------------------------------------------------------------------------------
LOG_PATH = "log.csv"

def load_safe_csv(path: str):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        try: os.remove(path)
        except: pass
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# LANGUAGE
# ------------------------------------------------------------------------------
LANG = st.sidebar.selectbox("Language", ["English", "বাংলা (Bangla)"])

TEXT = {
    "English": {
        "title": "AI-based Mental Health Assessment",
        "screen": "🧩 Screening",
        "dash": "📊 Dashboard",
        "choose_target": "Select assessment",
        "screening_form": "Screening Form",
        "instructions": "Rate each statement from 1 (lowest) to 5 (highest)",
        "scale": "Scale Meaning",
        "predict": "🔍 Predict Mental Health Status",
        "risk_level": "Risk Level",
        "suggested": "Suggested Actions",
        "no_logs": "No screening records found.",
        "dash_title": "Analytics Dashboard",
        "dash_recent": "Recent Results",
        "dash_risk": "Risk Distribution",
        "dash_trend": "Trend Over Time",
    },
    "বাংলা (Bangla)": {
        "title": "এআই ভিত্তিক মানসিক স্বাস্থ্য মূল্যায়ন",
        "screen": "🧩 স্ক্রিনিং",
        "dash": "📊 ড্যাশবোর্ড",
        "choose_target": "মূল্যায়ন নির্বাচন করুন",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "প্রতিটি প্রশ্নের জন্য ১ (সর্বনিম্ন) থেকে ৫ (সর্বোচ্চ) রেটিং দিন",
        "scale": "স্কেল মানে",
        "predict": "🔍 মানসিক স্বাস্থ্যের ফলাফল দেখুন",
        "risk_level": "ঝুঁকির স্তর",
        "suggested": "প্রস্তাবিত পদক্ষেপ",
        "no_logs": "কোনও স্ক্রিনিং ডেটা পাওয়া যায়নি।",
        "dash_title": "অ্যানালিটিক্স ড্যাশবোর্ড",
        "dash_recent": "সাম্প্রতিক ফলাফল",
        "dash_risk": "ঝুঁকির বণ্টন",
        "dash_trend": "সময়ের সঙ্গে স্ক্রিনিং প্রবণতা",
    },
}[LANG]

# ------------------------------------------------------------------------------
# QUESTIONS
# ------------------------------------------------------------------------------
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
        "Could not cope with all things you had to do",
        "Able to control irritations in your life",
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
        "Feeling bad about yourself or like a failure",
        "Trouble concentrating on things",
        "Moving / speaking slowly or restlessness",
        "Thoughts of self-harm or death",
    ],
}

# Bangla Translation
QUESTIONS_BN = {
    "Anxiety": [
        "নার্ভাস বা উদ্বিগ্ন অনুভব",
        "দুশ্চিন্তা থামাতে বা নিয়ন্ত্রণ করতে না পারা",
        "বিভিন্ন বিষয়ে অতিরিক্ত দুশ্চিন্তা",
        "মনকে শান্ত করতে কষ্ট হওয়া",
        "স্থির হয়ে বসে থাকতে সমস্যা",
        "সহজেই বিরক্ত হয়ে যাওয়া",
        "মনে হওয়া কোনো খারাপ কিছু ঘটবে",
    ],
    "Stress": [
        "অপ্রত্যাশিত ঘটনায় খুব কষ্ট পাওয়া",
        "গুরুত্বপূর্ণ বিষয় নিয়ন্ত্রণ করতে না পারা",
        "নার্ভাস ও চাপগ্রস্ত অনুভব করা",
        "সমস্যা সামলাতে আত্মবিশ্বাসী হওয়া",
        "সব কিছু ইচ্ছেমতো হওয়া",
        "সব কাজ করতে না পারা",
        "বিরক্তিকর বিষয় নিয়ন্ত্রণ করতে পারা",
        "অনুভব করা সবকিছুর উপরে আছেন",
        "বিষয় নিয়ন্ত্রণের বাইরে গেলে রাগ হওয়া",
        "সমস্যা খুব দ্রুত জমে ওঠা",
    ],
    "Depression": [
        "কাজে আগ্রহ কমে যাওয়া",
        "মনখারাপ, বিষণ্ন বা আশাহীন লাগা",
        "ঘুমের সমস্যা বা বেশি ঘুমানো",
        "অল্পতেই ক্লান্ত হওয়া",
        "খাবারে অনাগ্রহ বা অতিরিক্ত খাওয়া",
        "নিজেকে ব্যর্থ মনে হওয়া",
        "কাজে মনোযোগ দিতে সমস্যা",
        "ধীরে চলা/অস্থিরতা",
        "নিজেকে আঘাত করার চিন্তা",
    ],
}

# ------------------------------------------------------------------------------
# SCORE CALCULATION
# ------------------------------------------------------------------------------
def score_and_risk(values, target):
    scaled = [v - 1 for v in values]

    if target == "Anxiety":
        total = sum(scaled)
        if total <= 4: return "Minimal Anxiety", "Low", total, 21
        if total <= 9: return "Mild Anxiety", "Moderate", total, 21
        if total <= 14: return "Moderate Anxiety", "High", total, 21
        return "Severe Anxiety", "Critical", total, 21

    if target == "Stress":
        total = sum(scaled)
        if total <= 13: return "Minimal Stress", "Low", total, 40
        if total <= 26: return "Moderate Stress", "High", total, 40
        return "Severe Stress", "Critical", total, 40

    if target == "Depression":
        total = sum(scaled)
        if total <= 4: return "Minimal Depression", "Low", total, 27
        if total <= 9: return "Mild Depression", "Moderate", total, 27
        if total <= 14: return "Moderate Depression", "High", total, 27
        return "Severe Depression", "Critical", total, 27

# ------------------------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------------------------
page = st.sidebar.radio("Navigation", [TEXT["screen"], TEXT["dash"]])

# ------------------------------------------------------------------------------
# PAGE: SCREENING
# ------------------------------------------------------------------------------
if page == TEXT["screen"]:
    st.title(TEXT["title"])

    # Select assessment
    target = st.selectbox(TEXT["choose_target"], ["Anxiety", "Stress", "Depression"])

    st.subheader(f"{target} — {TEXT['screening_form']}")
    st.write(TEXT["instructions"])

    questions = QUESTIONS[target] if LANG == "English" else QUESTIONS_BN[target]

    responses = []
    for i, q in enumerate(questions):
        st.write(f"**Q{i+1}. {q}**")
        responses.append(st.slider(f"Q{i+1}", 1, 5, 3, label_visibility="collapsed"))

    # Predict
    if st.button(TEXT["predict"]):
        label, risk, total, max_score = score_and_risk(responses, target)

        st.success(f"🎯 {label}")
        st.info(f"🩺 {TEXT['risk_level']}: **{risk}**")
        st.write(f"Score: **{total} / {max_score}**")

        st.write("### " + TEXT["suggested"])
        if risk == "Low":
            st.write("- Maintain healthy habits and regular routine.")
        elif risk == "Moderate":
            st.write("- Reduce stress sources; use relaxation techniques.")
        elif risk == "High":
            st.write("- Seek support from trusted people or counselors.")
        else:
            st.write("- Professional mental health support recommended.")

        # Save Result
        df = load_safe_csv(LOG_PATH)
        row = pd.DataFrame(
            [{
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": target,
                "label": label,
                "risk": risk,
                "score": total,
                "max_score": max_score,
            }]
        )
        if df.empty: row.to_csv(LOG_PATH, index=False)
        else:
            df = pd.concat([df, row], ignore_index=True)
            df.to_csv(LOG_PATH, index=False)
        st.success("Saved to history.")

# ------------------------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------------------------
elif page == TEXT["dash"]:
    st.title(TEXT["dash_title"])

    df = load_safe_csv(LOG_PATH)

    if df.empty:
        st.warning(TEXT["no_logs"])
    else:
        st.subheader(TEXT["dash_recent"])
        st.dataframe(df.tail(20), use_container_width=True)

        # Risk distribution
        st.subheader(TEXT["dash_risk"])
        risk_counts = df["risk"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        chart = alt.Chart(risk_counts).mark_bar().encode(
            x="risk:N", y="count:Q", color="risk:N"
        )
        st.altair_chart(chart, use_container_width=True)

        # Trend
        st.subheader(TEXT["dash_trend"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        trend = df.groupby(df["datetime"].dt.date).size().reset_index(name="screenings")
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="datetime:T", y="screenings:Q"
        )
        st.altair_chart(chart, use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "mh_log.csv",
            "text/csv"
        )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown(
    """
<div style='text-align:center;margin-top:40px;opacity:0.7;'>
AI Mental Health Assessment System<br>
Developed by <b>Team Dual Core</b><br>
© 2025 All Rights Reserved
</div>
""",
    unsafe_allow_html=True,
)
