################################################################################
# app_v10.py
#
# AI-based Mental Health Assessment System
# Version 10 — Full Feature, GitHub Ready, Dark/Light Friendly
#
# Features:
# - Multi-language (English + Bangla)
# - 8 mental health domains:
#   Anxiety, Stress, Depression, Sleep, Burnout, ADHD, PTSD, Anger
# - No live preview (final result on Predict)
# - Clinical-style insights
# - Bangladesh crisis info (999, Kaan Pete Roi)
# - User profile (name + age group)
# - Daily screening streak
# - Private mode (don’t save results)
# - Auto-reset corrupted CSV (log.csv, users.csv, journal.csv)
# - Analytics Dashboard (risk distribution, time trend, timeline, simple prediction)
# - Breathing & Relaxation page
# - Mood Journal with simple sentiment signal
# - Footer: Designed & Developed by Ovi Sarker
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import random
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------------------------------------------------------
# BASIC CSS (works with both Light & Dark mode)
# (We do NOT override body text color to avoid dark-mode invisibility)
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
.main-card {
    background: rgba(255, 255, 255, 0.85);
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
    margin-bottom: 20px;
}
@media (prefers-color-scheme: dark) {
    .main-card {
        background: rgba(15, 23, 42, 0.95);
    }
}
.q-card {
    background: rgba(249, 250, 251, 0.9);
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 4px;
    border: 1px solid rgba(209, 213, 219, 0.9);
    font-size: 0.95rem;
}
@media (prefers-color-scheme: dark) {
    .q-card {
        background: rgba(31, 41, 55, 0.9);
        border-color: rgba(55, 65, 81, 0.9);
    }
}
.scale-card {
    background: rgba(239, 246, 255, 0.9);
    border-radius: 12px;
    padding: 10px 12px;
    border: 1px solid rgba(191, 219, 254, 0.9);
    font-size: 0.9rem;
}
@media (prefers-color-scheme: dark) {
    .scale-card {
        background: rgba(30, 64, 175, 0.25);
        border-color: rgba(147, 197, 253, 0.8);
    }
}
.badge {
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    margin-right: 8px;
    margin-top: 8px;
}
.badge-low  { background:#DCFCE7; color:#166534; }
.badge-mod  { background:#FEF9C3; color:#854D0E; }
.badge-high { background:#FFEDD5; color:#9A3412; }
.badge-crit { background:#FEE2E2; color:#991B1B; }

.small-muted {
    font-size: 0.85rem;
    opacity: 0.8;
}

.coach-card {
    background: rgba(224, 242, 254, 0.95);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(186, 230, 253, 0.95);
}
@media (prefers-color-scheme: dark) {
    .coach-card {
        background: rgba(8, 47, 73, 0.85);
        border-color: rgba(56, 189, 248, 0.8);
    }
}

.journal-card {
    background: rgba(243, 244, 246, 0.96);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(209, 213, 219, 0.95);
}
@media (prefers-color-scheme: dark) {
    .journal-card {
        background: rgba(31, 41, 55, 0.95);
        border-color: rgba(75, 85, 99, 0.9);
    }
}

.breath-card {
    background: rgba(255, 247, 237, 0.96);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(254, 215, 170, 0.95);
}
@media (prefers-color-scheme: dark) {
    .breath-card {
        background: rgba(30, 64, 175, 0.25);
        border-color: rgba(251, 191, 36, 0.8);
    }
}

.footer {
    margin-top: 30px;
    padding: 12px 0 4px 0;
    font-size: 0.85rem;
    opacity: 0.7;
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------------------
LOG_PATH = "log.csv"
USER_PATH = "users.csv"
JOURNAL_PATH = "journal.csv"

# ------------------------------------------------------------------------------
# SAFE CSV LOADER (auto-reset corrupted)
# ------------------------------------------------------------------------------
def load_safe_csv(path: str) -> pd.DataFrame:
    """Read CSV safely; reset file if corrupted."""
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

# ------------------------------------------------------------------------------
# LANGUAGE SETUP
# ------------------------------------------------------------------------------
LANG = st.sidebar.selectbox("Language", ["English", "বাংলা (Bangla)"])

TEXT = {
    "English": {
        "app_title": "AI-based Mental Health Assessment",
        "screen": "🧩 Screening",
        "breath": "🫁 Breathing & Relaxation",
        "dash": "📊 Dashboard",
        "coach": "🧑‍⚕️ Coach",
        "journal": "📓 Mood Journal",
        "choose_target": "What would you like to assess?",
        "screening_form": "Screening Form",
        "instructions": "Rate each item from 1 (lowest) to 5 (highest) based on the last 2 weeks.",
        "scale_title": "Scale Meaning (1–5)",
        "btn_predict": "🔍 Predict Mental Health Status",
        "risk_level": "Risk Level",
        "suggested_actions": "Suggested Actions",
        "disclaimer": "This tool never replaces a professional diagnosis or treatment.",
        "emergency": "If you feel suicidal, unsafe, or in crisis, contact emergency services or a mental health professional immediately.",
        "no_logs": "No screening results have been saved yet.",
        "dash_title": "Analytics Dashboard",
        "dash_last": "Recent Screening Results",
        "dash_risk_dist": "Risk Distribution",
        "dash_over_time": "Screenings Over Time",
        "dash_pred": "Severity Prediction (next screening)",
        "dash_timeline": "Timeline by Scale",
        "profile_title": "User Profile",
        "profile_name": "Name (optional)",
        "profile_age": "Age group",
        "profile_save": "Save profile",
        "profile_saved": "Profile saved.",
        "private_mode": "Private mode (do NOT save my results)",
        "clear_data": "🗑 Clear all saved data",
        "clear_done": "All saved CSV data cleared.",
        "coach_intro": "Get supportive, clinical-style suggestions based on your severity level.",
        "coach_choose": "Choose a severity level (or your last result):",
        "coach_btn": "Get guidance",
        "coach_q": "Short question (optional):",
        "coach_reply_title": "Guidance",
        "journal_title": "Write about your day and mood",
        "journal_hint": "Example: I feel tired and worried about my exams...",
        "journal_btn": "Save mood entry",
        "journal_saved": "Mood entry saved.",
        "journal_none": "No mood entries yet.",
        "streak_title": "Daily Screening Streak",
        "streak_none": "No streak yet — try completing one screening today.",
        "motiv_title": "Daily Mental Health Card",
    },
    "বাংলা (Bangla)": {
        "app_title": "এআই ভিত্তিক মানসিক স্বাস্থ্যের মূল্যায়ন",
        "screen": "🧩 স্ক্রিনিং",
        "breath": "🫁 শ্বাস-প্রশ্বাস ও রিল্যাক্সেশন",
        "dash": "📊 ড্যাশবোর্ড",
        "coach": "🧑‍⚕️ কোচ",
        "journal": "📓 মুড জার্নাল",
        "choose_target": "আপনি কোনটি মূল্যায়ন করতে চান?",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "গত ২ সপ্তাহের হিসেবে প্রতিটি প্রশ্নের জন্য ১ (সর্বনিম্ন) থেকে ৫ (সর্বোচ্চ) নির্বাচন করুন।",
        "scale_title": "স্কেল মানে (১–৫)",
        "btn_predict": "🔍 মানসিক স্বাস্থ্যের পূর্বাভাস দেখুন",
        "risk_level": "ঝুঁকির স্তর",
        "suggested_actions": "পরামর্শকৃত পদক্ষেপ",
        "disclaimer": "এই টুল কখনই পেশাদার ডাক্তারের পরামর্শ বা চিকিৎসার বিকল্প নয়।",
        "emergency": "আপনি যদি খুব খারাপ অনুভব করেন, আত্মহত্যার চিন্তা আসে বা সংকটে থাকেন, অবিলম্বে জরুরি পরিষেবা বা মানসিক স্বাস্থ্য বিশেষজ্ঞের সাথে যোগাযোগ করুন।",
        "no_logs": "এখনও কোনো স্ক্রিনিং সেভ করা হয়নি।",
        "dash_title": "অ্যানালিটিক্স ড্যাশবোর্ড",
        "dash_last": "সাম্প্রতিক স্ক্রিনিং ফলাফল",
        "dash_risk_dist": "ঝুঁকির মাত্রা বণ্টন",
        "dash_over_time": "সময়ের সাথে স্ক্রিনিং সংখ্যা",
        "dash_pred": "পরবর্তী স্ক্রিনিংয়ের পূর্বাভাস",
        "dash_timeline": "স্কেল অনুযায়ী টাইমলাইন",
        "profile_title": "ব্যবহারকারী প্রোফাইল",
        "profile_name": "নাম (ইচ্ছামত)",
        "profile_age": "বয়সের গ্রুপ",
        "profile_save": "প্রোফাইল সেভ করুন",
        "profile_saved": "প্রোফাইল সংরক্ষণ হয়েছে।",
        "private_mode": "প্রাইভেট মোড (ফলাফল সেভ হবে না)",
        "clear_data": "🗑 সব সেভ করা ডেটা মুছে ফেলুন",
        "clear_done": "সব CSV ডেটা মুছে ফেলা হয়েছে।",
        "coach_intro": "আপনার তীব্রতার স্তর অনুযায়ী ক্লিনিক্যাল ধাঁচের সহায়ক পরামর্শ পাবেন।",
        "coach_choose": "একটি তীব্রতার স্তর বেছে নিন (বা আপনার শেষ ফলাফল):",
        "coach_btn": "পরামর্শ দেখান",
        "coach_q": "ছোট কোনো প্রশ্ন থাকলে লিখুন (ঐচ্ছিক):",
        "coach_reply_title": "পরামর্শ",
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

# ------------------------------------------------------------------------------
# QUESTIONS (EN + BN)
# ------------------------------------------------------------------------------
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
        "Difficulty staying asleep",
        "Waking up earlier than desired",
        "Overall satisfaction with sleep",
        "Sleep problems noticed by others",
        "Worry or distress about sleep",
        "Impact of poor sleep on daily life",
    ],
    "Burnout": [
        "Feeling emotionally drained from work/study",
        "Feeling used up at the end of the day",
        "Tired when starting the day",
        "Dealing with people is a strain",
        "Feeling more callous toward others",
        "Feeling overwhelmed by responsibilities",
        "Feeling less effective in your role",
        "Feeling you are not achieving worthwhile things",
        "Feeling detached from your work/study",
        "Considering quitting your current work/study situation",
    ],
    "ADHD": [
        "Difficulty finishing tasks you start",
        "Trouble organizing tasks/activities",
        "Avoiding tasks requiring sustained mental effort",
        "Often losing things needed for tasks",
        "Easily distracted by external stimuli",
        "Forgetful in daily activities",
        "Fidgeting or difficulty remaining seated",
        "Feeling 'on the go' as if driven by a motor",
        "Talking excessively",
        "Interrupting or intruding on others",
    ],
    "PTSD": [
        "Upsetting memories about a stressful event",
        "Nightmares related to the event",
        "Emotional or physical reactions when reminded",
        "Avoiding thoughts or feelings about the event",
        "Avoiding places or activities that remind you of it",
        "Loss of interest in activities you once enjoyed",
        "Feeling distant or cut off from others",
        "Feeling watchful, on guard, or easily startled",
    ],
    "Anger": [
        "Feeling angry over small things",
        "Having difficulty controlling your anger",
        "Thinking repeatedly about things that made you angry",
        "Shouting or arguing more than you would like",
        "Breaking or hitting things when angry",
        "Regretting your reactions after calming down",
        "Others say they feel scared/uncomfortable when you are angry",
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
        "আপনার কি মনে হয় যেন কিছু খারাপ ঘটতে যাচ্ছে?",
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
        "দিনের শেষে কি সম্পূর্ণ ক্লান্ত হয়ে পড়েন?",
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
        "অনেকে কি বলে আপনি রেগে গেলে তারা ভয় পায় বা অস্বস্তি বোধ করে?",
    ],
}

# ------------------------------------------------------------------------------
# SCALE TEXT
# ------------------------------------------------------------------------------
SCALE_EN = {
    "Anxiety": ["Not at all", "Several days", "Half the days", "Nearly every day", "Almost always"],
    "Depression": ["Not at all", "Several days", "Half the days", "Nearly every day", "Almost always"],
    "Stress": ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"],
    "Sleep": ["No problem", "Mild problem", "Somewhat", "Quite a bit", "Very severe"],
    "Burnout": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "ADHD": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "PTSD": ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"],
    "Anger": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
}
SCALE_BN = {
    "Anxiety": ["একদমই না", "কিছুদিন", "অর্ধেক দিন", "প্রায় প্রতিদিন", "প্রায় সব সময়"],
    "Depression": ["একদমই না", "কিছুদিন", "অর্ধেক দিন", "প্রায় প্রতিদিন", "প্রায় সব সময়"],
    "Stress": ["কখনোই না", "খুব কম", "মাঝে মাঝে", "প্রায়ই", "প্রায় সব সময়"],
    "Sleep": ["কোন সমস্যা নেই", "হালকা সমস্যা", "মাঝারি", "অনেক বেশি", "অত্যন্ত বেশি"],
    "Burnout": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
    "ADHD": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
    "PTSD": ["একদমই না", "সামান্য", "মাঝারি", "অনেক বেশি", "অত্যন্ত বেশি"],
    "Anger": ["কখনোই না", "কম", "মাঝে মাঝে", "প্রায়ই", "খুব প্রায়ই"],
}

# ------------------------------------------------------------------------------
# SCORING + RISK
# ------------------------------------------------------------------------------
def score_and_risk(values, target):
    """
    Map raw 1–5 responses into:
    - label_str (e.g., 'Moderate Anxiety')
    - risk tier: Low / Moderate / High / Critical
    - total numeric score
    - max score
    """
    # Turn into 0–4 for scoring
    scaled = [v - 1 for v in values]

    if target == "Anxiety":
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
        total = sum(scaled)  # 0–40
        max_score = 4 * 10
        if total <= 13:
            level, risk = "Minimal", "Low"
        elif total <= 26:
            level, risk = "Moderate", "High"
        else:
            level, risk = "Severe", "Critical"
        return f"{level} Stress", risk, total, max_score

    # Generic for others: interpret via percentage
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


def risk_badge_class(risk: str) -> str:
    return {
        "Low": "badge-low",
        "Moderate": "badge-mod",
        "High": "badge-high",
        "Critical": "badge-crit",
    }.get(risk, "badge-mod")

# ------------------------------------------------------------------------------
# STREAK CALCULATION
# ------------------------------------------------------------------------------
def compute_streak(df: pd.DataFrame) -> int:
    if df.empty or "datetime" not in df.columns:
        return 0
    try:
        df["datetime"] = pd.to_datetime(df["datetime"])
        dates = sorted({d.date() for d in df["datetime"]})
        if not dates:
            return 0
        today = max(dates)
        streak = 0
        cur = today
        while cur in dates:
            streak += 1
            cur = cur - timedelta(days=1)
        return streak
    except Exception:
        return 0

# ------------------------------------------------------------------------------
# PROFILE HELPERS
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# CLINICAL-STYLE COACH REPLY
# ------------------------------------------------------------------------------
def generate_coach_reply(severity_label: str, question: str, lang: str) -> str:
    q = (question or "").lower()
    base = ""

    if any(k in q for k in ["sleep", "insomnia", "ঘুম"]):
        base = (
            "Your description suggests a pattern of sleep dysregulation. "
            "Structuring a consistent sleep–wake cycle, limiting screens and caffeine before bed, "
            "and keeping a calm pre-sleep routine can reduce physiological arousal over time."
        )
    elif any(k in q for k in ["exam", "study", "পরীক্ষা"]):
        base = (
            "Your concerns point toward performance-related stress. Breaking study tasks into smaller, "
            "time-limited segments, using short breaks and realistic daily goals can reduce cognitive overload."
        )
    elif any(k in q for k in ["relationship", "friend", "বন্ধু"]):
        base = (
            "These themes indicate interpersonal stress. Clear communication, boundaries and expressing needs "
            "in a non-judgmental way often improve relationship safety and emotional stability."
        )
    else:
        base = (
            "Your situation reflects a combination of emotional and cognitive pressure. Strengthening basic routines "
            "— sleep, nutrition, movement and supportive contact — is a clinically sound starting point."
        )

    if "Severe" in severity_label:
        tail = (
            " Given the severe level indicated, it would be clinically appropriate to consult "
            "a mental health professional as soon as possible."
        )
    elif "Moderate" in severity_label:
        tail = (
            " With a moderate level, self-help strategies may help, but if symptoms persist for "
            "several weeks, professional assessment is recommended."
        )
    else:
        tail = (
            " At a minimal or mild level, maintaining protective habits and monitoring symptoms "
            "usually supports long-term stability."
        )

    return base + tail

# ------------------------------------------------------------------------------
# MOTIVATION CARDS
# ------------------------------------------------------------------------------
MOTIVATIONS_EN = [
    "You do not have to be perfect to deserve rest.",
    "Small steps count as real progress.",
    "Your feelings are valid even if others don’t understand them.",
    "Taking care of yourself is a form of strength, not weakness.",
    "You have survived 100% of your hardest days so far.",
]
MOTIVATIONS_BN = [
    "আপনাকে নিখুঁত হতে হবে না — বিশ্রাম আপনারও প্রাপ্য।",
    "ছোট ছোট অগ্রগতি মিলিয়েই বড় পরিবর্তন হয়।",
    "অন্য কেউ না বুঝলেও আপনার অনুভূতিগুলো সত্যি।",
    "নিজের যত্ন নেওয়া এক ধরনের শক্তি, দুর্বলতা নয়।",
    "এর আগে আপনার সব কঠিন দিনই আপনি পার করেছেন।",
]

# ------------------------------------------------------------------------------
# SIDEBAR: PROFILE + SETTINGS + STREAK
# ------------------------------------------------------------------------------
st.sidebar.markdown(f"### {TEXT['profile_title']}")

last_name, last_age = get_last_profile()
age_options = ["", "<18", "18–24", "25–34", "35–44", "45–59", "60+"]

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

# Streak
st.sidebar.markdown(f"#### {TEXT['streak_title']}")
df_log_side = load_safe_csv(LOG_PATH)
streak = compute_streak(df_log_side)
if streak <= 0:
    st.sidebar.caption(TEXT["streak_none"])
else:
    st.sidebar.markdown(f"🔥 **{streak} day(s)** in a row")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [TEXT["screen"], TEXT["breath"], TEXT["dash"], TEXT["coach"], TEXT["journal"]],
)

# ------------------------------------------------------------------------------
# PAGE 1 — SCREENING
# ------------------------------------------------------------------------------
if page == TEXT["screen"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["app_title"])
    st.markdown(f"<p class='small-muted'>⚠ {TEXT['disclaimer']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-muted'>🚨 {TEXT['emergency']}</p>", unsafe_allow_html=True)

    # Motivation card
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

    left, right = st.columns([3, 1.4])

    # Scale meaning
    with right:
        st.markdown("<div class='scale-card'>", unsafe_allow_html=True)
        st.markdown(f"**{TEXT['scale_title']}**", unsafe_allow_html=True)
        scale_list = SCALE_EN[target] if LANG == "English" else SCALE_BN[target]
        for i, label in enumerate(scale_list, start=1):
            st.write(f"{i} — {label}")
        st.markdown("</div>", unsafe_allow_html=True)

    responses = []
    with left:
        qs = QUESTIONS_EN[target] if LANG == "English" else QUESTIONS_BN[target]
        for i, q_text in enumerate(qs):
            st.markdown(f"<div class='q-card'>{q_text}</div>", unsafe_allow_html=True)
            # label must NOT be empty (for accessibility warning)
            responses.append(
                st.slider(
                    label=f"Q{i+1}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{target}_q{i+1}",
                )
            )

    if st.button(TEXT["btn_predict"]):
        label_str, risk, total_score, max_score = score_and_risk(responses, target)
        badge_cls = risk_badge_class(risk)

        st.markdown(
            f"<span class='badge {badge_cls}'>🎯 {label_str}</span>"
            f"<span class='badge {badge_cls}'>🩺 {TEXT['risk_level']}: {risk}</span>",
            unsafe_allow_html=True,
        )

        # Clinical-style explanation
        st.write("#### Clinical interpretation (simplified)")
        pct = (total_score / max_score) if max_score else 0
        pct_disp = pct * 100
        st.write(f"- Your severity on this scale is approximately **{pct_disp:.1f}%** of its maximum.")

        if target == "Anxiety":
            st.write(
                "- This reflects the level of nervousness, worry and physiological tension you have been experiencing."
            )
        if target == "Depression":
            st.write(
                "- This score relates to mood, interest, energy and self-worth over roughly the last two weeks."
            )
        if target == "Stress":
            st.write(
                "- This reflects how unpredictable, uncontrollable and overloaded your life has felt recently."
            )

        if pct <= 0.25:
            st.write(
                "- Symptoms appear limited. Monitoring your mental health and maintaining healthy routines is recommended."
            )
        elif pct <= 0.5:
            st.write(
                "- Symptoms are clinically relevant but in a mild range. Lifestyle changes and support can be protective."
            )
        elif pct <= 0.75:
            st.write(
                "- Symptoms are in a moderate range and may impact daily functioning. Clinical consultation could be helpful."
            )
        else:
            st.write(
                "- Symptoms are severe and likely impactful. Professional assessment and support are strongly recommended."
            )

        # Suggestions
        st.write(f"### {TEXT['suggested_actions']}")
        suggestions = {
            "Low": "Maintain sleep, nutrition, exercise and supportive relationships.",
            "Moderate": "Introduce structured routines, breathing exercises, journaling and talk to trusted people.",
            "High": "Reduce overload where possible, seek counseling or a mental health professional.",
            "Critical": "Prioritize safety and urgently contact a qualified mental health professional or emergency support.",
        }
        st.write(suggestions.get(risk, ""))

        # Bangladesh crisis info
        st.error(
            "Bangladesh crisis support:\n"
            "- 🚑 Emergency services: **999**\n"
            "- ☎ Emotional support (Kaan Pete Roi): **+8809609900999**\n"
            "If you feel at immediate risk of self-harm, please contact these services or trusted people around you."
        )

        # Save
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
            st.success("✅ Screening result saved.")
        else:
            st.info("🔒 Private mode: result not saved to database.")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 2 — BREATHING & RELAXATION
# ------------------------------------------------------------------------------
elif page == TEXT["breath"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["breath"])

    st.markdown("<div class='breath-card'>", unsafe_allow_html=True)
    st.write(
        "These techniques do not replace treatment, but they can reduce immediate "
        "physiological arousal and help you feel more grounded."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Box Breathing", "4–7–8 Breathing", "5–4–3–2–1 Grounding"])

    with tab1:
        st.subheader("Box Breathing (4–4–4–4)")
        st.write(
            "1️⃣ Inhale through your nose for 4 seconds.\n"
            "2️⃣ Hold gently for 4 seconds.\n"
            "3️⃣ Exhale through your mouth for 4 seconds.\n"
            "4️⃣ Pause for 4 seconds before the next breath.\n\n"
            "Repeat 4–8 cycles."
        )

    with tab2:
        st.subheader("4–7–8 Breathing")
        st.write(
            "1️⃣ Inhale quietly through your nose for 4 seconds.\n"
            "2️⃣ Hold for 7 seconds.\n"
            "3️⃣ Exhale slowly through your mouth for 8 seconds.\n\n"
            "Use 4–6 cycles, especially before sleep."
        )

    with tab3:
        st.subheader("5–4–3–2–1 Grounding")
        st.write(
            "Identify around you:\n"
            "• 5 things you can see\n"
            "• 4 things you can feel\n"
            "• 3 things you can hear\n"
            "• 2 things you can smell\n"
            "• 1 thing you can taste\n\n"
            "This shifts attention back to the present moment."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 3 — DASHBOARD
# ------------------------------------------------------------------------------
elif page == TEXT["dash"]:
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
        if "risk" in df.columns:
            st.subheader(TEXT["dash_risk_dist"])
            risk_counts = df["risk"].value_counts().reset_index()
            risk_counts.columns = ["risk", "count"]
            chart = (
                alt.Chart(risk_counts)
                .mark_bar()
                .encode(
                    x=alt.X("risk:N", sort="-y"),
                    y="count:Q",
                    color="risk:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)

        # Over time
        if "datetime" in df.columns:
            st.subheader(TEXT["dash_over_time"])
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            trend = (
                df.dropna(subset=["datetime"])
                .groupby(df["datetime"].dt.date)
                .size()
                .reset_index(name="screenings")
            )
            if not trend.empty:
                chart = (
                    alt.Chart(trend)
                    .mark_line(point=True)
                    .encode(x="datetime:T", y="screenings:Q")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not enough valid dates to show trend.")

        # Simple prediction
        st.subheader(TEXT["dash_pred"])
        try:
            df_sorted = df.sort_values("datetime")
            x = np.arange(len(df_sorted))
            y = df_sorted["score"].values / df_sorted["max_score"].values * 100
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                next_x = len(x)
                next_y = float(np.clip(coeffs[0] * next_x + coeffs[1], 0, 100))
                st.write(f"📈 Predicted next average severity: **{next_y:.1f}%** of maximum.")
                st.progress(int(next_y))
            else:
                st.write("Not enough screenings to estimate prediction.")
        except Exception:
            st.write("Could not compute prediction from data.")

        # Timeline by scale
        st.subheader(TEXT["dash_timeline"])
        if "target" in df.columns:
            scales = sorted(df["target"].dropna().unique())
            choice = st.selectbox("Select scale", scales)
            sub = df[df["target"] == choice].copy()
            if not sub.empty and "datetime" in sub.columns:
                sub["datetime"] = pd.to_datetime(sub["datetime"], errors="coerce")
                sub = sub.dropna(subset=["datetime"])
                if not sub.empty:
                    sub["date"] = sub["datetime"].dt.date
                    sub["severity_pct"] = sub["score"] / sub["max_score"] * 100
                    tl = (
                        sub.groupby("date")["severity_pct"]
                        .mean()
                        .reset_index()
                        .rename(columns={"severity_pct": "Severity (%)"})
                    )
                    chart = (
                        alt.Chart(tl)
                        .mark_line(point=True)
                        .encode(x="date:T", y="Severity (%):Q")
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("No valid dates for this scale.")
            else:
                st.caption("No data for the selected scale.")

        # Download
        st.download_button(
            "⬇️ Download all results (CSV)",
            data=df.to_csv(index=False),
            file_name="mental_health_log.csv",
            mime="text/csv",
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 4 — COACH
# ------------------------------------------------------------------------------
elif page == TEXT["coach"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["coach"])
    st.markdown(f"<p class='small-muted'>{TEXT['coach_intro']}</p>", unsafe_allow_html=True)

    df = load_safe_csv(LOG_PATH)
    last_label = None
    if not df.empty and "label" in df.columns:
        last_label = df.iloc[-1]["label"]

    st.write(TEXT["coach_choose"])
    severity_choice = st.selectbox(
        "Severity",
        ["Use my last result"] + ["Minimal", "Mild", "Moderate", "Severe"],
    )

    if severity_choice == "Use my last result" and last_label is not None:
        base_label = last_label
    elif severity_choice == "Use my last result":
        base_label = "Minimal level"
    else:
        base_label = f"{severity_choice} level"

    question = st.text_input(TEXT["coach_q"])

    if st.button(TEXT["coach_btn"]):
        st.markdown("<div class='coach-card'>", unsafe_allow_html=True)
        st.write(f"**Current severity reference:** {base_label}")
        reply = generate_coach_reply(base_label, question, LANG)
        st.write(f"### {TEXT['coach_reply_title']}")
        st.write(reply)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 5 — MOOD JOURNAL
# ------------------------------------------------------------------------------
else:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["journal"])

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

    # Show last entry summary
    df_j = load_safe_csv(JOURNAL_PATH)
    if df_j.empty:
        st.info(TEXT["journal_none"])
    else:
        last = df_j.iloc[-1]
        st.write("----")
        st.write("**Last entry (summary):**")
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
                "Your text contains more signals of stress/low mood. "
                "Clinically, it is helpful to include rest, social support and small pleasant activities in your day."
            )
        elif pos_hits > neg_hits:
            st.write(
                "Your entry includes positive or hopeful signals. Noticing what supports this state "
                "can help you repeat those behaviors."
            )
        else:
            st.write(
                "Your entry appears mixed or neutral. Regular journaling can clarify patterns over time."
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
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
