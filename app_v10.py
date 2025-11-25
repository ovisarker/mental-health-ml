################################################################################
# AI-based Mental Health Assessment — ULTRA v11
# - English + Bangla
# - GAD-7 / PHQ-9 / PSS-10 + Sleep, Burnout, ADHD, PTSD, Anger
# - Screening, Dashboard, Coach, Mood Journal, Breathing
# - Motivation card, streaks, AI-style insights, crisis detection, timelines
# - Safe CSV, private mode, optional PDF report
# - Footer: Designed & Developed by Ovi Sarker
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import altair as alt
import os
import random

# Optional PDF support: safe import
try:
    from fpdf import FPDF  # pip install fpdf
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ------------------------------------------------------------------
# PAGE CONFIG + GLOBAL STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
body { background-color:#F4F7FB; color:#111827; }
h1, h2, h3, h4, h5, h6 { color:#111827 !important; font-weight:700 !important; }

.main-card {
    background:#FFFFFF;
    padding:26px;
    border-radius:18px;
    box-shadow:0 8px 18px rgba(15,23,42,0.08);
    margin-bottom:22px;
}

.scale-card {
    background:#E8F2FF;
    border:1px solid #C5DAFF;
    padding:16px;
    border-radius:14px;
    font-size:0.9rem;
}

.q-card {
    background:#F9FAFB;
    border-radius:12px;
    padding:12px 14px;
    margin-bottom:6px;
    border:1px solid #E5E7EB;
}

.badge {
    padding:8px 14px;
    border-radius:999px;
    font-weight:600;
    font-size:0.9rem;
    display:inline-block;
    margin-right:8px;
    margin-top:8px;
}

.badge-low  { background:#DCFCE7; color:#166534; }
.badge-mod  { background:#FEF9C3; color:#854D0E; }
.badge-high { background:#FFEDD5; color:#9A3412; }
.badge-crit { background:#FEE2E2; color:#991B1B; }

.small-muted {
    font-size:0.85rem;
    color:#6B7280;
}

.coach-card {
    background:#ECFEFF;
    border-radius:14px;
    padding:16px;
    border:1px solid #BAE6FD;
}

.journal-card {
    background:#F5F3FF;
    border-radius:14px;
    padding:16px;
    border:1px solid #DDD6FE;
}

.breath-card {
    background:#FFF7ED;
    border-radius:14px;
    padding:16px;
    border:1px solid #FED7AA;
}

.footer {
    margin-top:30px;
    padding:12px 0 4px 0;
    font-size:0.85rem;
    color:#6B7280;
    text-align:center;
}
</style>
""",
    unsafe_allow_html=True,
)

LOG_PATH = "log.csv"
USER_PATH = "users.csv"
JOURNAL_PATH = "journal.csv"

# ------------------------------------------------------------------
# SAFE CSV LOADER (AUTO-RESET ON CORRUPTION)
# ------------------------------------------------------------------
def load_safe_csv(path: str) -> pd.DataFrame:
    """
    Safe CSV loader that auto-resets corrupted CSV files.
    If CSV cannot be parsed, it is deleted and an empty DataFrame is returned.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        return pd.DataFrame()

# ------------------------------------------------------------------
# LANGUAGE STRINGS
# ------------------------------------------------------------------
LANG = st.sidebar.selectbox("Language", ["English", "বাংলা (Bangla)"])

TEXT = {
    "English": {
        "app_title": "AI-based Mental Health Assessment",
        "nav_screen": "🧩 Screening",
        "nav_breath": "🫁 Breathing & Relaxation",
        "nav_dash": "📊 Dashboard",
        "nav_coach": "🧑‍⚕️ Coach",
        "nav_journal": "📓 Mood Journal",
        "choose_target": "What would you like to assess?",
        "screening_form": "Screening Form",
        "instructions": "Rate each statement from 1 (lowest) to 5 (highest) based on the last 2 weeks.",
        "scale_title": "Scale Meaning (1–5)",
        "btn_predict": "🔍 Predict Mental Health Status",
        "risk_level": "Risk Level",
        "suggested_actions": "Suggested Actions",
        "disclaimer": "This tool does not replace professional diagnosis or treatment.",
        "emergency": "If you feel unsafe, suicidal, or in crisis, contact emergency services or a trusted professional immediately.",
        "no_logs": "No screenings have been saved yet.",
        "dash_title": "Analytics Dashboard",
        "dash_last": "Recent Screening Results",
        "dash_risk_dist": "Risk Distribution",
        "dash_over_time": "Screenings Over Time",
        "dash_pred": "AI Mood Prediction (next screening)",
        "dash_timeline": "Symptom Timeline by Scale",
        "profile_title": "User Profile",
        "profile_name": "Name (optional)",
        "profile_age": "Age group",
        "profile_save": "Save profile",
        "profile_saved": "Profile saved.",
        "private_mode": "Private mode (do NOT save my results)",
        "clear_data": "🗑 Clear all saved screenings & profiles",
        "clear_done": "All CSV data cleared.",
        "report_title": "Mental Health Screening Report",
        "coach_intro": "Get supportive, practical tips based on your last saved result or chosen severity.",
        "coach_choose": "Choose a severity level (or use your last result):",
        "coach_btn": "Get guidance",
        "coach_q": "Ask a short question (optional):",
        "coach_reply_title": "Supportive guidance",
        "journal_title": "Write about your day and mood",
        "journal_hint": "Example: I feel tired and worried about my exams...",
        "journal_btn": "Save mood entry",
        "journal_saved": "Mood entry saved.",
        "journal_none": "No mood entries yet.",
        "streak_title": "Daily Screening Streak",
        "streak_none": "No streak yet — start by doing a screening today.",
        "motiv_title": "Daily Mental Health Card",
    },
    "বাংলা (Bangla)": {
        "app_title": "এআই ভিত্তিক মানসিক স্বাস্থ্যের মূল্যায়ন",
        "nav_screen": "🧩 স্ক্রিনিং",
        "nav_breath": "🫁 শ্বাস-প্রশ্বাস ও রিল্যাক্সেশন",
        "nav_dash": "📊 ড্যাশবোর্ড",
        "nav_coach": "🧑‍⚕️ কোচ",
        "nav_journal": "📓 মুড জার্নাল",
        "choose_target": "আপনি কোনটি মূল্যায়ন করতে চান?",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "গত ২ সপ্তাহের ভিত্তিতে প্রতিটি প্রশ্নের জন্য ১ (সবচেয়ে কম) থেকে ৫ (সবচেয়ে বেশি) নির্বাচন করুন।",
        "scale_title": "স্কেল মানে (১–৫)",
        "btn_predict": "🔍 মানসিক স্বাস্থ্যের পূর্বাভাস দেখুন",
        "risk_level": "ঝুঁকির স্তর",
        "suggested_actions": "পরামর্শকৃত পদক্ষেপ",
        "disclaimer": "এই টুল কখনোই পেশাদার ডাক্তারের পরামর্শ বা চিকিৎসার বিকল্প নয়।",
        "emergency": "আপনি যদি খুব খারাপ অনুভব করেন, আত্মহত্যার চিন্তা আসে বা সংকটে থাকেন, অবিলম্বে জরুরি পরিষেবা বা বিশ্বস্ত পেশাদারের সাথে যোগাযোগ করুন।",
        "no_logs": "এখনও কোনো স্ক্রিনিং সংরক্ষণ করা হয়নি।",
        "dash_title": "অ্যানালিটিক্স ড্যাশবোর্ড",
        "dash_last": "সাম্প্রতিক স্ক্রিনিং ফলাফল",
        "dash_risk_dist": "ঝুঁকির মাত্রা বণ্টন",
        "dash_over_time": "সময়ের সাথে স্ক্রিনিং সংখ্যা",
        "dash_pred": "এআই মুড প্রেডিকশন (পরবর্তী স্ক্রিনিংয়ের পূর্বাভাস)",
        "dash_timeline": "স্কেল অনুযায়ী লক্ষণ পরিবর্তন (টাইমলাইন)",
        "profile_title": "ইউজার প্রোফাইল",
        "profile_name": "নাম (ইচ্ছামত)",
        "profile_age": "বয়সের গ্রুপ",
        "profile_save": "প্রোফাইল সেভ করুন",
        "profile_saved": "প্রোফাইল সংরক্ষণ হয়েছে।",
        "private_mode": "প্রাইভেট মোড (ফলাফল সেভ হবে না)",
        "clear_data": "🗑 সব সেভ করা ডেটা মুছে ফেলুন",
        "clear_done": "সব CSV ডেটা মুছে ফেলা হয়েছে।",
        "report_title": "মানসিক স্বাস্থ্য স্ক্রিনিং রিপোর্ট",
        "coach_intro": "আপনার সর্বশেষ ফলাফল বা নির্বাচিত স্তরের উপর ভিত্তি করে সহায়ক গাইডলাইন পাবেন।",
        "coach_choose": "একটি তীব্রতার স্তর বেছে নিন (বা শেষ ফলাফল ব্যবহার করুন):",
        "coach_btn": "পরামর্শ দেখান",
        "coach_q": "কোনো ছোট প্রশ্ন থাকলে লিখুন (ঐচ্ছিক):",
        "coach_reply_title": "সহায়ক নির্দেশনা",
        "journal_title": "আজকের দিন ও মুড সম্পর্কে লিখুন",
        "journal_hint": "উদাহরণ: আজ খুব ক্লান্ত লাগছে, পরীক্ষার চিন্তা হচ্ছে...",
        "journal_btn": "মুড এন্ট্রি সেভ করুন",
        "journal_saved": "মুড এন্ট্রি সেভ হয়েছে।",
        "journal_none": "এখনও কোনো মুড এন্ট্রি নেই।",
        "streak_title": "দৈনিক স্ক্রিনিং স্ট্রিক",
        "streak_none": "এখনও স্ট্রিক শুরু হয়নি — আজ একটি স্ক্রিনিং করুন।",
        "motiv_title": "দৈনিক মানসিক স্বাস্থ্য কার্ড",
    },
}[LANG]

# ------------------------------------------------------------------
# MOTIVATION CARDS
# ------------------------------------------------------------------
MOTIVATIONS_EN = [
    "You don’t have to be perfect to deserve rest.",
    "Small steps still move you forward.",
    "Your feelings are valid, even if others don’t see them.",
    "Taking care of yourself is a quiet form of courage.",
    "You have survived 100% of your hardest days so far.",
    "It’s okay to ask for help — it means you’re human.",
]
MOTIVATIONS_BN = [
    "আপনাকে নিখুঁত হতে হবে না — বিশ্রাম আপনারও প্রাপ্য।",
    "ছোট ছোট পদক্ষেপও এগিয়ে যাওয়া হিসেবেই গুনে।",
    "আপনার অনুভূতিগুলো সত্যি, অন্য কেউ না বুঝলেও।",
    "নিজের যত্ন নেওয়া এক ধরনের নীরব সাহস।",
    "এর আগে আপনার সব কঠিন দিনই আপনি পার করেছেন।",
    "সাহায্য চাওয়া দুর্বলতা নয় — এটা মানুষ হওয়ার প্রমাণ।",
]

# ------------------------------------------------------------------
# QUESTIONS — ENGLISH + BANGLA
# ------------------------------------------------------------------
QUESTIONS_EN = {
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
        "Moving/speaking slowly or restlessness",
        "Thoughts of self-harm or death",
    ],
    "Sleep": [
        "Difficulty falling asleep",
        "Difficulty staying asleep during the night",
        "Waking up earlier than desired",
        "Overall satisfaction with sleep",
        "Noticeable sleep problems to others",
        "Worry or distress about sleep",
        "Impact of poor sleep on daily functioning",
    ],
    "Burnout": [
        "Feeling emotionally drained from work/study",
        "Used up at the end of the day",
        "Tired when starting the day",
        "Dealing with people all day is a strain",
        "Becoming more callous toward people",
        "Feeling overwhelmed by responsibilities",
        "Feeling less effective in your role",
        "Feeling you are not achieving many worthwhile things",
        "Feeling detached from your work/study",
        "Considering quitting your current work/study situation",
    ],
    "ADHD": [
        "Difficulty finishing tasks you start",
        "Trouble organizing things",
        "Avoiding tasks that require sustained mental effort",
        "Losing things needed for tasks or activities",
        "Easily distracted by external stimuli",
        "Forgetful in daily activities",
        "Fidgeting or difficulty remaining seated",
        "Feeling 'on the go' or driven by a motor",
        "Talking excessively",
        "Interrupting or intruding on others",
    ],
    "PTSD": [
        "Upsetting memories about a stressful experience",
        "Nightmares related to the event",
        "Sudden emotional or physical reactions when reminded",
        "Avoiding thoughts or feelings about the event",
        "Avoiding places or activities that remind you of it",
        "Loss of interest in activities you used to enjoy",
        "Feeling distant or cut off from others",
        "Feeling watchful, on guard or easily startled",
    ],
    "Anger": [
        "Feeling angry over small things",
        "Difficulty controlling your anger",
        "Thinking about past events that make you angry",
        "Shouting or arguing more than you would like",
        "Breaking or hitting things when angry",
        "Regretting your reactions after calming down",
        "Others say they feel scared or uncomfortable when you are angry",
    ],
}

QUESTIONS_BN = {
    "Anxiety": [
        "আপনি কি নার্ভাস, উৎকণ্ঠিত বা অস্থির বোধ করছেন?",
        "আপনি কি দুশ্চিন্তা থামাতে বা নিয়ন্ত্রণ করতে পারেন না?",
        "আপনি কি বিভিন্ন বিষয় নিয়ে অতিরিক্ত দুশ্চিন্তা করছেন?",
        "আপনার কি আরাম করতে কষ্ট হয়?",
        "আপনি কি এতটাই অস্থির যে এক জায়গায় বসে থাকতে পারেন না?",
        "আপনি কি খুব সহজে বিরক্ত বা রাগান্বিত হয়ে যান?",
        "আপনার কি মনে হয়, যেন কিছু খারাপ ঘটতে যাচ্ছে?",
    ],
    "Stress": [
        "অপ্রত্যাশিত ঘটনার কারণে কি আপনি খুব বিরক্ত বা কষ্ট পেয়েছেন?",
        "জীবনের গুরুত্বপূর্ণ বিষয়গুলো নিয়ন্ত্রণ করতে না পারার অনুভূতি কি হয়েছে?",
        "আপনি কি নার্ভাস ও চাপগ্রস্ত অনুভব করেছেন?",
        "আপনি কি সমস্যাগুলো সামলাতে আত্মবিশ্বাসী বোধ করেছেন?",
        "সব কিছু কি আপনার ইচ্ছে মতো এগিয়েছে?",
        "করার মতো সব কাজ সামলাতে না পারার অনুভূতি কি হয়েছে?",
        "আপনি কি আপনার জীবনের বিরক্তিকর বিষয়গুলো নিয়ন্ত্রণ করতে পেরেছেন?",
        "আপনি কি অনুভব করেছেন যে আপনি সব কিছুর উপরে আছেন?",
        "বিষয়গুলো নিয়ন্ত্রণের বাইরে চলে যাওয়ায় কি আপনি রাগান্বিত হয়েছেন?",
        "আপনি কি মনে করেছেন যে আপনার সমস্যাগুলো খুব দ্রুত জমে উঠছে?",
    ],
    "Depression": [
        "কার্যকলাপ বা কাজকর্মে আগ্রহ বা আনন্দ কি কমে গেছে?",
        "আপনি কি মনখারাপ, বিষণ্ন বা আশাহীন অনুভব করেছেন?",
        "ঘুম আসতে সমস্যা, মাঝরাতে ঘুম ভাঙা বা বেশি ঘুমানো—এমন সমস্যা কি হয়েছে?",
        "আপনি কি খুব ক্লান্ত বোধ করছেন বা শক্তি কম মনে হচ্ছে?",
        "আপনার কি খাবারের আগ্রহ কমে গেছে বা বেশি খেয়ে ফেলছেন?",
        "আপনি কি মনে করেছেন আপনি খুব খারাপ, ব্যর্থ বা নিজেকে অপছন্দ করছেন?",
        "কোনো কাজে মনোযোগ ধরে রাখতে কি কষ্ট হচ্ছে?",
        "আপনি কি খুব ধীরে কথা বলেন/হাঁটেন বা অস্থিরভাবে নড়াচড়া করেন?",
        "আপনার কি কখনও মনে হয়েছে নিজেকে আঘাত করা বা মৃত্যুর কথা?",
    ],
    "Sleep": [
        "ঘুমাতে যেতে কি অনেক সময় লাগে?",
        "রাতে ঘুম ভেঙে গেলে আবার ঘুমাতে কি কষ্ট হয়?",
        "ইচ্ছার চেয়ে আগেই কি ঘুম ভেঙে যায়?",
        "মোটের উপর আপনার ঘুম নিয়ে কতটা সন্তুষ্ট?",
        "অন্যরা কি আপনার ঘুমের সমস্যা লক্ষ্য করে?",
        "ঘুম নিয়ে কি আপনি দুশ্চিন্তা বা কষ্ট অনুভব করেন?",
        "খারাপ ঘুম আপনার দৈনন্দিন কাজকে কতটা প্রভাবিত করছে?",
    ],
    "Burnout": [
        "কাজ/পড়াশোনা থেকে কি মানসিকভাবে ক্লান্ত বোধ করেন?",
        "দিনের শেষে কি পুরোপুরি ক্লান্ত হয়ে পড়েন?",
        "দিনের শুরুতেই কি ক্লান্তি নিয়ে শুরু করেন?",
        "সারাদিন মানুষের সাথে কাজ করা কি আপনাকে ক্লান্ত করে?",
        "আপনি কি মানুষের প্রতি কিছুটা কঠোর/উদাসীন হয়ে গেছেন?",
        "দায়িত্বগুলো কি আপনাকে চাপে ফেলে দিচ্ছে?",
        "নিজের ভূমিকায় কি আগের মত কার্যকর বোধ করেন না?",
        "আপনি কি মনে করেন খুব বেশি অর্থবহ কাজ করতে পারছেন না?",
        "কাজ/পড়াশোনা থেকে কি নিজেকে দূরে মনে হয়?",
        "বর্তমান কাজ/পড়াশোনা ছেড়ে দিতে চান কিনা এমন ভাবনা আসে?",
    ],
    "ADHD": [
        "শুরু করা কাজ শেষ করতে কি কষ্ট হয়?",
        "কাজগুলো সংগঠিত করতে কি সমস্যা হয়?",
        "যে কাজগুলোতে দীর্ঘ সময় মনোযোগ দরকার সেগুলো এড়িয়ে যান?",
        "কাজের জিনিসপত্র সহজে হারিয়ে ফেলেন?",
        "বাইরের শব্দ বা ঘটনা কি সহজে আপনাকে বিভ্রান্ত করে?",
        "দৈনন্দিন কাজ ভুলে যান কি?",
        "বসে থাকতে কি অস্থির লাগে বা ফিজেট করেন?",
        "সব সময় যেন কাজের মধ্যে থাকতে হয় এমন অনুভূতি হয়?",
        "খুব বেশি কথা বলে ফেলেন কি?",
        "অন্যের কথা কেটে কথা বলা বা হস্তক্ষেপ করে ফেলেন কি?",
    ],
    "PTSD": [
        "কোনো স্ট্রেসফুল ঘটনার স্মৃতি কি আপনাকে বিরক্ত করে?",
        "সেই ঘটনা নিয়ে দুঃস্বপ্ন দেখেন কি?",
        "ঘটনার কথা মনে পড়লে কি হঠাৎ মানসিক/শারীরিক প্রতিক্রিয়া হয়?",
        "ঘটনা নিয়ে ভাবা বা অনুভূতি এড়িয়ে যান?",
        "ঘটনার সাথে সম্পর্কিত জায়গা/কাজ এড়িয়ে চলেন?",
        "আগে যেগুলো করতে ভালো লাগত সেগুলোর প্রতি আগ্রহ কমে গেছে?",
        "অন্যদের থেকে কি নিজেকে বিচ্ছিন্ন মনে হয়?",
        "সব সময় কি সজাগ, টেনশনে বা সহজে ভয় পেয়ে যান?",
    ],
    "Anger": [
        "ছোটখাটো বিষয়েও কি রাগ উঠে যায়?",
        "রাগ নিয়ন্ত্রণ করতে কি কষ্ট হয়?",
        "আগের রাগের ঘটনা নিয়ে কি বারবার ভাবেন?",
        "প্রায়ই কি ঝগড়া/উচ্চস্বরে কথা বলে ফেলেন?",
        "রাগের সময় কি জিনিসপত্র ভাঙা বা মারধর করার ইচ্ছা হয়?",
        "শান্ত হওয়ার পর কি নিজের আচরণের জন্য আফসোস হয়?",
        "অনেকে কি বলে যে আপনি রেগে গেলে তারা ভয় পায় বা অস্বস্তি বোধ করে?",
    ],
}

# SCALE MEANING
SCALE_EN = {
    "Anxiety": [
        "Not at all",
        "Several days",
        "More than half the days",
        "Nearly every day",
        "Almost always",
    ],
    "Depression": [
        "Not at all",
        "Several days",
        "More than half the days",
        "Nearly every day",
        "Almost always",
    ],
    "Stress": ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"],
    "Sleep": ["No problem", "Mild problem", "Somewhat", "Quite a bit", "Very severe"],
    "Burnout": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "ADHD": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "PTSD": ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"],
    "Anger": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
}

SCALE_BN = {
    "Anxiety": [
        "একদমই না",
        "কিছুদিন",
        "অর্ধেকের বেশি দিন",
        "প্রায় প্রতিদিন",
        "প্রায় সব সময়",
    ],
    "Depression": [
        "একদমই না",
        "কিছুদিন",
        "অর্ধেকের বেশি দিন",
        "প্রায় প্রতিদিন",
        "প্রায় সব সময়",
    ],
    "Stress": ["কখনোই না", "খুব কম", "মাঝে মাঝে", "প্রায়ই", "প্রায় সব সময়"],
    "Sleep": ["কোন সমস্যা নেই", "হালকা সমস্যা", "মাঝারি সমস্যা", "অনেক বেশি", "খুব তীব্র"],
    "Burnout": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
    "ADHD": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
    "PTSD": ["একদমই না", "সামান্য", "মাঝারি", "অনেক বেশি", "অত্যন্ত বেশি"],
    "Anger": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
}

# ------------------------------------------------------------------
# SCORING
# ------------------------------------------------------------------
def score_and_risk(values, target):
    """
    values: list of slider values 1–5
    Returns:
        label_str (e.g. "Mild Anxiety"),
        risk_tier ("Low/Moderate/High/Critical"),
        total_score, max_score
    """
    if target == "Anxiety":
        scaled = [v - 1 for v in values]  # 0–3
        total = sum(scaled)  # 0–21
        max_score = 3 * 7
        if total <= 4:
            level, risk = "Minimal", "Low"
        elif total <= 9:
            level, risk = "Mild", "Moderate"
        elif total <= 14:
            level, risk = "Moderate", "High"
        else:
            level, risk = "Severe", "Critical"
        return f"{level} Anxiety", risk, total, max_score

    if target == "Depression":
        scaled = [v - 1 for v in values]
        total = sum(scaled)  # 0–27
        max_score = 3 * 9
        if total <= 4:
            level, risk = "Minimal", "Low"
        elif total <= 9:
            level, risk = "Mild", "Moderate"
        elif total <= 14:
            level, risk = "Moderate", "High"
        else:
            level, risk = "Severe", "Critical"
        return f"{level} Depression", risk, total, max_score

    if target == "Stress":
        scaled = [v - 1 for v in values]  # 0–4
        total = sum(scaled)  # 0–40
        max_score = 4 * 10
        if total <= 13:
            level, risk = "Minimal", "Low"
        elif total <= 26:
            level, risk = "Moderate", "High"
        else:
            level, risk = "Severe", "Critical"
        return f"{level} Stress", risk, total, max_score

    # Generic scoring for other scales: 0–4 each
    scaled = [v - 1 for v in values]
    total = sum(scaled)
    max_score = 4 * len(values)
    pct = total / max_score if max_score else 0
    if pct <= 0.25:
        level, risk = "Minimal", "Low"
    elif pct <= 0.5:
        level, risk = "Mild", "Moderate"
    elif pct <= 0.75:
        level, risk = "Moderate", "High"
    else:
        level, risk = "Severe", "Critical"
    return f"{level} {target}", risk, total, max_score


def risk_badge_class(risk):
    return {
        "Low": "badge-low",
        "Moderate": "badge-mod",
        "High": "badge-high",
        "Critical": "badge-crit",
    }.get(risk, "badge-mod")

# ------------------------------------------------------------------
# STREAK CALCULATION
# ------------------------------------------------------------------
def compute_streak(df: pd.DataFrame) -> int:
    """
    Compute consecutive-day streak based on 'datetime' column.
    """
    if df.empty or "datetime" not in df.columns:
        return 0
    try:
        df["datetime"] = pd.to_datetime(df["datetime"])
        dates = sorted({d.date() for d in df["datetime"]})
        if not dates:
            return 0
        today = max(dates)
        streak = 0
        current = today
        while current in dates:
            streak += 1
            current = current - timedelta(days=1)
        return streak
    except Exception:
        return 0

# ------------------------------------------------------------------
# USER PROFILE HELPERS
# ------------------------------------------------------------------
def save_profile(name, age_group):
    df_users = load_safe_csv(USER_PATH)
    new_row = pd.DataFrame(
        [
            {
                "name": name,
                "age_group": age_group,
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )
    if df_users.empty:
        new_row.to_csv(USER_PATH, index=False)
    else:
        df_users = pd.concat([df_users, new_row], ignore_index=True)
        df_users.to_csv(USER_PATH, index=False)


def get_last_profile():
    df_users = load_safe_csv(USER_PATH)
    if df_users.empty:
        return "", ""
    last = df_users.iloc[-1]
    return last.get("name", ""), last.get("age_group", "")

# ------------------------------------------------------------------
# REPORT GENERATION
# ------------------------------------------------------------------
def build_report_text(
    profile_name, target, label_str, risk, total_score, max_score, lang
) -> bytes:
    title = TEXT["report_title"]
    lines = [
        f"{title}",
        "-" * len(title),
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Language: {lang}",
        "",
        f"Name: {profile_name if profile_name else 'N/A'}",
        f"Assessment Type: {target}",
        f"Severity: {label_str}",
        f"Risk Level: {risk}",
        f"Score: {total_score} / {max_score}",
        "",
        "Note: This is a self-assessment screening report and does not replace",
        "any clinical diagnosis, treatment or professional consultation.",
    ]
    return "\n".join(lines).encode("utf-8")


def build_pdf_from_text(report_bytes: bytes):
    """Create a simple PDF from text if fpdf is available."""
    if not HAS_FPDF:
        return None
    text = report_bytes.decode("utf-8")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.splitlines():
        pdf.multi_cell(0, 8, line)
    pdf_str = pdf.output(dest="S")
    if isinstance(pdf_str, str):
        return pdf_str.encode("latin-1", "ignore")
    return pdf_str

# ------------------------------------------------------------------
# COACH REPLY (simple rule-based)
# ------------------------------------------------------------------
def generate_coach_reply(severity_label: str, question: str, lang: str) -> str:
    q = (question or "").lower()
    base = ""

    if "sleep" in q or "insomnia" in q or "ঘুম" in q:
        base = (
            "Try to keep a fixed sleep and wake-up time, avoid screens 1 hour "
            "before bed and reduce caffeine in the evening."
        )
    elif "study" in q or "exam" in q or "পরীক্ষা" in q:
        base = (
            "Break tasks into small parts, use short focused study blocks with "
            "regular breaks and remind yourself that progress is more important "
            "than perfection."
        )
    elif "relationship" in q or "friend" in q or "বন্ধু" in q:
        base = (
            "Healthy communication, clear boundaries and listening with respect "
            "help relationships feel safer and more supportive."
        )
    else:
        base = (
            "Focus on small, realistic steps: sleep, food, movement and one "
            "connection with a supportive person each day."
        )

    if "Severe" in severity_label:
        tail = (
            " Because your current severity seems high, it would be wise to "
            "speak with a mental health professional soon."
        )
    elif "Moderate" in severity_label:
        tail = (
            " Your symptoms are noticeable, so if they stay the same for a few "
            "weeks, consider taking professional help."
        )
    else:
        tail = (
            " Right now your scores are on the lower side, which is good. "
            "Keep using simple healthy habits to protect this."
        )

    return base + tail

# ------------------------------------------------------------------
# SIDEBAR: PROFILE + SETTINGS + STREAK
# ------------------------------------------------------------------
st.sidebar.markdown(f"### {TEXT['profile_title']}")

last_name, last_age = get_last_profile()
age_options = ["", "<18", "18-24", "25-34", "35-44", "45-59", "60+"]

profile_name = st.sidebar.text_input(TEXT["profile_name"], value=last_name or "")
age_group = st.sidebar.selectbox(
    TEXT["profile_age"],
    age_options,
    index=(age_options.index(last_age) if last_age in age_options else 0),
)

if st.sidebar.button(TEXT["profile_save"]):
    if profile_name or age_group:
        save_profile(profile_name, age_group)
        st.sidebar.success(TEXT["profile_saved"])

private_mode = st.sidebar.checkbox(TEXT["private_mode"], value=False)

if st.sidebar.button(TEXT["clear_data"]):
    for path in [LOG_PATH, USER_PATH, JOURNAL_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    st.sidebar.success(TEXT["clear_done"])

# Streak view
st.sidebar.markdown(f"#### {TEXT['streak_title']}")
df_log_sidebar = load_safe_csv(LOG_PATH)
streak = compute_streak(df_log_sidebar)
if streak <= 0:
    st.sidebar.caption(TEXT["streak_none"])
else:
    st.sidebar.markdown(f"🔥 **{streak} day(s)** in a row")

# ------------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        TEXT["nav_screen"],
        TEXT["nav_breath"],
        TEXT["nav_dash"],
        TEXT["nav_coach"],
        TEXT["nav_journal"],
    ],
)

# ------------------------------------------------------------------
# 🧩 SCREENING PAGE
# ------------------------------------------------------------------
if page == TEXT["nav_screen"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.header(TEXT["app_title"])
    st.markdown(f"<p class='small-muted'>⚠ {TEXT['disclaimer']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-muted'>🚨 {TEXT['emergency']}</p>", unsafe_allow_html=True)

    # Daily motivation card
    st.markdown(f"### {TEXT['motiv_title']}")
    if LANG == "English":
        mot = random.choice(MOTIVATIONS_EN)
    else:
        mot = random.choice(MOTIVATIONS_BN)
    st.info(mot)

    target = st.selectbox(
        TEXT["choose_target"],
        ["Anxiety", "Stress", "Depression", "Sleep", "Burnout", "ADHD", "PTSD", "Anger"],
    )

    st.subheader(f"🧾 {target} {TEXT['screening_form']}")
    st.write(TEXT["instructions"])

    left_col, right_col = st.columns([3.2, 1.3], vertical_alignment="top")

    # RIGHT: SCALE CARD
    with right_col:
        st.markdown("<div class='scale-card'>", unsafe_allow_html=True)
        st.markdown(f"**{TEXT['scale_title']}**")
        scale_list = SCALE_EN[target] if LANG == "English" else SCALE_BN[target]
        for i, label in enumerate(scale_list, start=1):
            st.write(f"{i} — {label}")
        st.markdown("</div>", unsafe_allow_html=True)

    # LEFT: QUESTIONS (no live preview)
    responses = []
    with left_col:
        qs = QUESTIONS_EN[target] if LANG == "English" else QUESTIONS_BN[target]
        for i, q_text in enumerate(qs):
            st.markdown(f"<div class='q-card'>{q_text}</div>", unsafe_allow_html=True)
            responses.append(
                st.slider(
                    label="",  # question text is shown above
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{target}_{i}",
                )
            )

    # NOSTALGIC PREDICT BUTTON — ONLY FINAL RESULT SHOWN
    if st.button(TEXT["btn_predict"]):
        label_str, risk, total_score, max_score = score_and_risk(responses, target)
        badge_cls = risk_badge_class(risk)

        st.markdown(
            f"<span class='badge {badge_cls}'>🎯 {label_str}</span>"
            f"<span class='badge {badge_cls}'>🩺 {TEXT['risk_level']}: {risk}</span>",
            unsafe_allow_html=True,
        )

        # Explanation
        st.write("#### Explanation")
        if "Minimal" in label_str:
            st.write(
                "Your current answers suggest only mild or occasional symptoms. "
                "This is a good time to keep healthy habits and stay aware of any changes."
            )
        elif "Mild" in label_str:
            st.write(
                "Your symptoms are present but still on the lighter side. "
                "Lifestyle adjustments and regular self-checks may help you feel better."
            )
        elif "Moderate" in label_str:
            st.write(
                "Your responses show clear, ongoing symptoms. "
                "They are affecting your daily life and deserve attention and support."
            )
        else:
            st.write(
                "Your scores indicate strong symptoms. "
                "Please consider talking with a mental health professional as soon as you can."
            )

        # AI-style insights
        st.write("### 🔍 Insights about your pattern")
        pct = (total_score / max_score) if max_score else 0
        pct_disp = pct * 100
        st.write(f"- Overall severity is approximately **{pct_disp:.1f}%** of the maximum for this scale.")

        if target in ["Anxiety", "Stress"] and pct > 0.6:
            st.write(
                "- High levels on this scale often show up as difficulty relaxing, overthinking "
                "and feeling 'on edge' during daily tasks."
            )
        if target == "Depression" and pct > 0.6:
            st.write(
                "- This pattern can be linked with low energy, loss of interest and harsh self-judgement. "
                "It deserves kind attention and support."
            )
        if target == "Sleep" and pct > 0.6:
            st.write(
                "- Sleep difficulties can amplify both stress and mood symptoms. Improving sleep hygiene "
                "often helps other scores slowly improve."
            )
        if target == "Burnout" and pct > 0.6:
            st.write(
                "- Burnout scores like this are common when responsibilities feel constant and rest "
                "does not feel refreshing anymore."
            )
        if target == "PTSD" and pct > 0.6:
            st.write(
                "- Higher PTSD-like scores may reflect the impact of past stressful or traumatic events "
                "that are still affecting your present life."
            )
        if target == "Anger" and pct > 0.6:
            st.write(
                "- Anger at this level can sometimes cover other emotions like hurt or fear. "
                "Learning safe ways to express it can be very helpful."
            )

        if pct <= 0.4:
            st.write(
                "- Your current level is on the lower side. This is a good time to build and protect "
                "healthy routines so things stay manageable."
            )

        # Suggested actions
        st.write(f"### {TEXT['suggested_actions']}")
        suggestions = {
            "Low": "Maintain good sleep, food, exercise and keep monitoring your mood.",
            "Moderate": "Try relaxation, journaling, breathing exercises and talk to trusted people.",
            "High": "Reduce workload if possible and strongly consider talking with a mental health professional.",
            "Critical": "Please seek immediate support from a licensed mental health professional or crisis service.",
        }
        st.write(suggestions.get(risk, ""))

        # Crisis safety message (for very high severity)
        if risk == "Critical" or (
            target in ["Depression", "PTSD"] and pct > 0.7
        ):
            st.error(
                "⚠ Your responses suggest significant distress. This screening cannot diagnose you, "
                "but it strongly suggests that talking to a mental health professional or doctor "
                "would be very important. If you feel at risk of harming yourself or others, "
                "please contact local emergency services or a trusted crisis helpline immediately."
            )

        # Save to CSV if not in private mode
        if not private_mode:
            df_log = load_safe_csv(LOG_PATH)
            new_row = pd.DataFrame(
                [
                    {
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "language": LANG,
                        "user_name": profile_name,
                        "age_group": age_group,
                        "target": target,
                        "label": label_str,
                        "risk": risk,
                        "score": total_score,
                        "max_score": max_score,
                    }
                ]
            )
            if df_log.empty:
                new_row.to_csv(LOG_PATH, index=False)
            else:
                df_log = pd.concat([df_log, new_row], ignore_index=True)
                df_log.to_csv(LOG_PATH, index=False)
            st.success("✅ Screening saved.")
        else:
            st.info("🔒 Private mode enabled — result not saved.")

        # Build downloadable text report (+ optional PDF)
        report_bytes = build_report_text(
            profile_name, target, label_str, risk, total_score, max_score, LANG
        )
        st.download_button(
            "⬇️ Download text report",
            data=report_bytes,
            file_name="mental_health_report.txt",
            mime="text/plain",
        )

        if HAS_FPDF:
            pdf_bytes = build_pdf_from_text(report_bytes)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF report",
                    data=pdf_bytes,
                    file_name="mental_health_report.pdf",
                    mime="application/pdf",
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🫁 BREATHING & RELAXATION PAGE
# ------------------------------------------------------------------
elif page == TEXT["nav_breath"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["nav_breath"])

    st.markdown("<div class='breath-card'>", unsafe_allow_html=True)
    st.write(
        "These simple breathing and grounding exercises are not a treatment, "
        "but they can help your body and mind calm down in the moment."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Box Breathing", "4–7–8 Breathing", "5–4–3–2–1 Grounding"])

    with tab1:
        st.subheader("Box Breathing (4–4–4–4)")
        st.write(
            "1️⃣ Inhale through your nose for 4 seconds.\n"
            "2️⃣ Hold your breath gently for 4 seconds.\n"
            "3️⃣ Exhale slowly through your mouth for 4 seconds.\n"
            "4️⃣ Pause for 4 seconds before the next breath.\n\n"
            "Repeat this cycle 4–8 times."
        )

    with tab2:
        st.subheader("4–7–8 Breathing")
        st.write(
            "1️⃣ Inhale quietly through your nose for 4 seconds.\n"
            "2️⃣ Hold your breath for 7 seconds.\n"
            "3️⃣ Exhale completely through your mouth for 8 seconds.\n\n"
            "Repeat 4–6 times, especially helpful before sleep."
        )

    with tab3:
        st.subheader("5–4–3–2–1 Grounding")
        st.write(
            "Look around you and slowly name:\n"
            "• 5 things you can see\n"
            "• 4 things you can feel (e.g., chair, clothes)\n"
            "• 3 things you can hear\n"
            "• 2 things you can smell\n"
            "• 1 thing you can taste\n\n"
            "This helps bring your mind back to the present moment."
        )

    st.markdown("---")
    st.subheader("Optional: Guided Audio (add your own files)")
    audio_files = {
        "Calm breathing (short)": "calm_breathing_short.mp3",
        "Sleep relaxation": "sleep_relaxation.mp3",
    }

    for label, filename in audio_files.items():
        if os.path.exists(filename):
            st.write(f"🎧 {label}")
            st.audio(filename)
        else:
            st.caption(f"ℹ To use **{label}**, place an audio file named `{filename}` in the app folder.")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 📊 DASHBOARD PAGE
# ------------------------------------------------------------------
elif page == TEXT["nav_dash"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["dash_title"])

    df = load_safe_csv(LOG_PATH)

    if df.empty:
        st.warning(TEXT["no_logs"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.subheader(TEXT["dash_last"])
        st.dataframe(df.tail(20), use_container_width=True)

        # Risk distribution
        st.subheader(TEXT["dash_risk_dist"])
        risk_counts = df["risk"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        risk_chart = (
            alt.Chart(risk_counts)
            .mark_bar()
            .encode(
                x=alt.X("risk:N", sort="-y"),
                y="count:Q",
                color="risk:N",
            )
        )
        st.altair_chart(risk_chart, use_container_width=True)

        # Over time
        st.subheader(TEXT["dash_over_time"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        trend = df.groupby(df["datetime"].dt.date).size().reset_index(name="screenings")
        trend_chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(x="datetime:T", y="screenings:Q")
        )
        st.altair_chart(trend_chart, use_container_width=True)

        # AI Mood Prediction
        st.subheader(TEXT["dash_pred"])
        try:
            df_sorted = df.sort_values("datetime")
            x = np.arange(len(df_sorted))
            y = df_sorted["score"].values / df_sorted["max_score"].values * 100
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                next_x = len(x)
                next_y = coeffs[0] * next_x + coeffs[1]
                next_y = float(np.clip(next_y, 0, 100))
                st.write(f"📈 Predicted next overall severity: **{next_y:.1f}% of max**")
                st.progress(int(next_y))
            else:
                st.write("Not enough screenings yet to predict trend.")
        except Exception:
            st.write("Could not compute prediction from existing data.")

        # Symptom timeline by scale
        st.subheader(TEXT["dash_timeline"])
        targets = sorted(df["target"].unique())
        chosen_t = st.selectbox("Choose scale", targets)
        subset = df[df["target"] == chosen_t].copy()
        if not subset.empty:
            subset["datetime"] = pd.to_datetime(subset["datetime"])
            subset["date"] = subset["datetime"].dt.date
            subset["severity_pct"] = subset["score"] / subset["max_score"] * 100
            tl = (
                subset.groupby("date")["severity_pct"]
                .mean()
                .reset_index()
                .rename(columns={"severity_pct": "Severity (%)"})
            )
            timeline_chart = (
                alt.Chart(tl)
                .mark_line(point=True)
                .encode(x="date:T", y="Severity (%):Q")
            )
            st.altair_chart(timeline_chart, use_container_width=True)
        else:
            st.caption("No data yet for this scale.")

        # Download logs
        st.download_button(
            "⬇️ Download all results (CSV)",
            data=df.to_csv(index=False),
            file_name="mental_health_log.csv",
            mime="text/csv",
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🧑‍⚕️ COACH PAGE
# ------------------------------------------------------------------
elif page == TEXT["nav_coach"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["nav_coach"])
    st.markdown(f"<p class='small-muted'>{TEXT['coach_intro']}</p>", unsafe_allow_html=True)

    df = load_safe_csv(LOG_PATH)
    last_label = None
    if not df.empty:
        last_label = df.iloc[-1].get("label", None)

    st.write(TEXT["coach_choose"])
    severity_choice = st.selectbox(
        "Severity",
        ["Use my last result"] + ["Minimal", "Mild", "Moderate", "Severe"],
    )

    if severity_choice == "Use my last result" and last_label is not None:
        base_label = last_label
    elif severity_choice == "Use my last result":
        base_label = "Minimal"
    else:
        base_label = f"{severity_choice} level"

    question = st.text_input(TEXT["coach_q"])

    if st.button(TEXT["coach_btn"]):
        st.markdown("<div class='coach-card'>", unsafe_allow_html=True)
        st.write(f"**Current severity:** {base_label}")
        reply = generate_coach_reply(base_label, question, LANG)
        st.write(f"### {TEXT['coach_reply_title']}")
        st.write(reply)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 📓 MOOD JOURNAL PAGE
# ------------------------------------------------------------------
else:  # Mood journal
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["nav_journal"])

    st.markdown("<div class='journal-card'>", unsafe_allow_html=True)
    st.write(f"**{TEXT['journal_title']}**")
    text = st.text_area(" ", placeholder=TEXT["journal_hint"], height=180)
    mood_rating = st.slider("Overall mood today (1 = very bad, 5 = very good)", 1, 5, 3)

    if st.button(TEXT["journal_btn"]):
        df_j = load_safe_csv(JOURNAL_PATH)
        new_row = pd.DataFrame(
            [
                {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "language": LANG,
                    "user_name": profile_name,
                    "age_group": age_group,
                    "mood_rating": mood_rating,
                    "text": text,
                }
            ]
        )
        if df_j.empty:
            new_row.to_csv(JOURNAL_PATH, index=False)
        else:
            df_j = pd.concat([df_j, new_row], ignore_index=True)
            df_j.to_csv(JOURNAL_PATH, index=False)
        st.success(TEXT["journal_saved"])

    # Advanced journal insight
    df_j = load_safe_csv(JOURNAL_PATH)
    if df_j.empty:
        st.info(TEXT["journal_none"])
    else:
        last = df_j.iloc[-1]
        st.write("----")
        st.write("**Last saved mood entry (summary):**")
        st.write(f"🕒 {last['datetime']}")
        st.write(f"🙂 Mood rating: {last['mood_rating']}/5")

        txt = str(last["text"]).lower()
        neg_words = [
            "tired",
            "sad",
            "alone",
            "stress",
            "worried",
            "anxious",
            "হতাশ",
            "একাকী",
            "টেনশন",
            "চাপ",
        ]
        pos_words = [
            "happy",
            "excited",
            "grateful",
            "relaxed",
            "উৎসাহী",
            "খুশি",
            "শান্ত",
            "আনন্দ",
        ]
        neg_hits = sum(w in txt for w in neg_words)
        pos_hits = sum(w in txt for w in pos_words)

        if neg_hits > pos_hits:
            st.write(
                "Your words contain more stress/negative feelings. "
                "Try doing one small kind thing for yourself today (rest, a short walk, "
                "listening to music or talking to someone you trust)."
            )
        elif pos_hits > neg_hits:
            st.write(
                "Your entry shows some positive or hopeful words. "
                "Notice what helped you feel this way and try to keep those habits nearby."
            )
        else:
            st.write(
                "Your entry seems balanced or neutral. Writing regularly can help you notice "
                "which people, places or activities affect your mood most."
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# GLOBAL FOOTER (ALL PAGES)
# ------------------------------------------------------------------
st.markdown(
    """
<div class='footer'>
🧠 AI Mental Health Assessment System<br>
Designed &amp; Developed by <strong>Ovi Sarker</strong><br>
© 2025 All Rights Reserved
</div>
""",
    unsafe_allow_html=True,
)

