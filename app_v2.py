################################################################################
# AI-based Mental Health Assessment — v8 FINAL
# - English + Bangla
# - GAD-7 / PHQ-9 / PSS-10 inspired scoring
# - User profile, live preview, dashboard, mood prediction, coach
# - Private mode + auto-reset CSV logs
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
import os

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
</style>
""",
    unsafe_allow_html=True,
)

LOG_PATH = "log.csv"
USER_PATH = "users.csv"

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
        "nav_dash": "📊 Dashboard",
        "nav_coach": "🧑‍⚕️ Coach",
        "choose_target": "What would you like to assess?",
        "screening_form": "Screening Form",
        "instructions": "Rate each statement from 1 (lowest) to 5 (highest) based on the last 2 weeks.",
        "scale_title": "Scale Meaning (1–5)",
        "btn_predict": "🔍 Save & Download Report",
        "live_preview": "Live Score Preview",
        "risk_level": "Risk Level",
        "suggested_actions": "Suggested Actions",
        "disclaimer": "This tool does not replace professional diagnosis or treatment.",
        "emergency": "If you feel unsafe, suicidal, or in crisis, contact emergency services or a trusted professional immediately.",
        "no_logs": "No screenings have been saved yet.",
        "dash_title": "Analytics Dashboard",
        "dash_last": "Recent Screening Results",
        "dash_risk_dist": "Risk Distribution",
        "dash_over_time": "Screenings Over Time",
        "dash_pred": "Simple Mood Prediction (next screening)",
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
    },
    "বাংলা (Bangla)": {
        "app_title": "এআই ভিত্তিক মানসিক স্বাস্থ্যের মূল্যায়ন",
        "nav_screen": "🧩 স্ক্রিনিং",
        "nav_dash": "📊 ড্যাশবোর্ড",
        "nav_coach": "🧑‍⚕️ কোচ",
        "choose_target": "আপনি কোনটি মূল্যায়ন করতে চান?",
        "screening_form": "স্ক্রিনিং ফর্ম",
        "instructions": "গত ২ সপ্তাহের ভিত্তিতে প্রতিটি প্রশ্নের জন্য ১ (সবচেয়ে কম) থেকে ৫ (সবচেয়ে বেশি) নির্বাচন করুন।",
        "scale_title": "স্কেল মানে (১–৫)",
        "btn_predict": "🔍 সেভ ও রিপোর্ট ডাউনলোড",
        "live_preview": "লাইভ স্কোর প্রিভিউ",
        "risk_level": "ঝুঁকির স্তর",
        "suggested_actions": "পরামর্শকৃত পদক্ষেপ",
        "disclaimer": "এই টুল কখনোই পেশাদার ডাক্তারের পরামর্শ বা চিকিৎসার বিকল্প নয়।",
        "emergency": "আপনি যদি খুব খারাপ অনুভব করেন, আত্মহত্যার চিন্তা আসে বা সংকটে থাকেন, অবিলম্বে জরুরি পরিষেবা বা বিশ্বস্ত পেশাদারের সাথে যোগাযোগ করুন।",
        "no_logs": "এখনও কোনো স্ক্রিনিং সংরক্ষণ করা হয়নি।",
        "dash_title": "অ্যানালিটিক্স ড্যাশবোর্ড",
        "dash_last": "সাম্প্রতিক স্ক্রিনিং ফলাফল",
        "dash_risk_dist": "ঝুঁকির মাত্রা বণ্টন",
        "dash_over_time": "সময়ের সাথে স্ক্রিনিং সংখ্যা",
        "dash_pred": "সহজ মুড প্রেডিকশন (পরবর্তী স্ক্রিনিংয়ের পূর্বাভাস)",
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
    },
}[LANG]

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
    "Stress": [
        "Never",
        "Almost never",
        "Sometimes",
        "Fairly often",
        "Very often",
    ],
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
    "Stress": [
        "কখনোই না",
        "খুব কম",
        "মাঝে মাঝে",
        "প্রায়ই",
        "প্রায় সব সময়",
    ],
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
        scaled = [v - 1 for v in values]        # 0–3
        total = sum(scaled)                    # 0–21
        max_score = 3 * 7
        if total <= 4:
            level = "Minimal"
            risk = "Low"
        elif total <= 9:
            level = "Mild"
            risk = "Moderate"
        elif total <= 14:
            level = "Moderate"
            risk = "High"
        else:
            level = "Severe"
            risk = "Critical"
        return f"{level} Anxiety", risk, total, max_score

    if target == "Depression":
        scaled = [v - 1 for v in values]
        total = sum(scaled)                    # 0–27
        max_score = 3 * 9
        if total <= 4:
            level = "Minimal"
            risk = "Low"
        elif total <= 9:
            level = "Mild"
            risk = "Moderate"
        elif total <= 14:
            level = "Moderate"
            risk = "High"
        else:
            level = "Severe"
            risk = "Critical"
        return f"{level} Depression", risk, total, max_score

    # Stress (PSS-10 style)
    scaled = [v - 1 for v in values]          # 0–4
    total = sum(scaled)                       # 0–40
    max_score = 4 * 10
    if total <= 13:
        level = "Minimal"
        risk = "Low"
    elif total <= 26:
        level = "Moderate"
        risk = "High"     # moderate PSS = high stress
    else:
        level = "Severe"
        risk = "Critical"
    return f"{level} Stress", risk, total, max_score


def risk_badge_class(risk):
    return {
        "Low": "badge-low",
        "Moderate": "badge-mod",
        "High": "badge-high",
        "Critical": "badge-crit",
    }.get(risk, "badge-mod")

# ------------------------------------------------------------------
# USER PROFILE HELPERS
# ------------------------------------------------------------------
def save_profile(name, age_group):
    df_users = load_safe_csv(USER_PATH)
    new_row = pd.DataFrame(
        [{"name": name, "age_group": age_group, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
    )
    if df_users.empty:
        new_row.to_csv(USER_PATH, index=False)
    else:
        # overwrite or append (simple append now)
        df_users = pd.concat([df_users, new_row], ignore_index=True)
        df_users.to_csv(USER_PATH, index=False)


def get_last_profile():
    df_users = load_safe_csv(USER_PATH)
    if df_users.empty:
        return "", ""
    last = df_users.iloc[-1]
    return last.get("name", ""), last.get("age_group", "")

# ------------------------------------------------------------------
# REPORT GENERATION (TEXT FILE)
# ------------------------------------------------------------------
def build_report_text(
    profile_name, target, label_str, risk, total_score, max_score, lang
) -> str:
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
        "Note: This is a self-assessment screening report and does not replace\n"
        "any clinical diagnosis, treatment or professional consultation.",
    ]
    return "\n".join(lines).encode("utf-8")

# ------------------------------------------------------------------
# SIDEBAR: PROFILE + SETTINGS
# ------------------------------------------------------------------
st.sidebar.markdown(f"### {TEXT['profile_title']}")

last_name, last_age = get_last_profile()

profile_name = st.sidebar.text_input(TEXT["profile_name"], value=last_name or "")
age_group = st.sidebar.selectbox(
    TEXT["profile_age"],
    ["", "<18", "18-24", "25-34", "35-44", "45-59", "60+"],
    index=(["", "<18", "18-24", "25-34", "35-44", "45-59", "60+"].index(last_age) if last_age in ["", "<18", "18-24", "25-34", "35-44", "45-59", "60+"] else 0),
)

if st.sidebar.button(TEXT["profile_save"]):
    if profile_name or age_group:
        save_profile(profile_name, age_group)
        st.sidebar.success(TEXT["profile_saved"])

private_mode = st.sidebar.checkbox(TEXT["private_mode"], value=False)

if st.sidebar.button(TEXT["clear_data"]):
    for path in [LOG_PATH, USER_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    st.sidebar.success(TEXT["clear_done"])

# ------------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [TEXT["nav_screen"], TEXT["nav_dash"], TEXT["nav_coach"]],
)

# ------------------------------------------------------------------
# 🧩 SCREENING PAGE
# ------------------------------------------------------------------
if page == TEXT["nav_screen"]:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.header(TEXT["app_title"])
    st.markdown(f"<p class='small-muted'>⚠ {TEXT['disclaimer']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-muted'>🚨 {TEXT['emergency']}</p>", unsafe_allow_html=True)

    target = st.selectbox(
        TEXT["choose_target"],
        ["Anxiety", "Stress", "Depression"],
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

    # LEFT: QUESTIONS + LIVE PREVIEW
    responses = []
    with left_col:
        qs = QUESTIONS_EN[target] if LANG == "English" else QUESTIONS_BN[target]
        for i, q in enumerate(qs):
            st.markdown(f"<div class='q-card'>{q}</div>", unsafe_allow_html=True)
            responses.append(
                st.slider(
                    label="",  # we show question above in card
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{target}_{i}",
                )
            )

        # Live preview after sliders
        label_str, risk, total_score, max_score = score_and_risk(responses, target)
        norm = total_score / max_score if max_score > 0 else 0

        st.markdown(f"### {TEXT['live_preview']}")
        st.write(f"**Score:** {total_score} / {max_score}")
        st.progress(int(norm * 100))
        st.write(f"**Severity:** {label_str}")
        st.write(f"**{TEXT['risk_level']}:** {risk}")

    # SAVE, LOG, REPORT
    if st.button(TEXT["btn_predict"]):
        label_str, risk, total_score, max_score = score_and_risk(responses, target)
        badge_cls = risk_badge_class(risk)

        st.markdown(
            f"<span class='badge {badge_cls}'>🎯 {label_str}</span>"
            f"<span class='badge {badge_cls}'>🩺 {TEXT['risk_level']}: {risk}</span>",
            unsafe_allow_html=True,
        )

        # Suggested actions
        suggestions = {
            "Low": "Maintain good sleep, food, exercise and keep monitoring your mood.",
            "Moderate": "Try relaxation, journaling, breathing exercises and talk to trusted people.",
            "High": "Reduce workload if possible and strongly consider talking with a mental health professional.",
            "Critical": "Please seek immediate support from a licensed mental health professional or crisis service.",
        }
        st.write(f"### {TEXT['suggested_actions']}")
        st.write(suggestions.get(risk, ""))

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

        # Build downloadable text report
        report_bytes = build_report_text(
            profile_name, target, label_str, risk, total_score, max_score, LANG
        )
        st.download_button(
            "⬇️ Download text report",
            data=report_bytes,
            file_name="mental_health_report.txt",
            mime="text/plain",
        )

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

        # Simple mood prediction (linear trend on score)
        st.subheader(TEXT["dash_pred"])
        try:
            # map dates to integer index for regression
            df_sorted = df.sort_values("datetime")
            x = np.arange(len(df_sorted))
            y = df_sorted["score"].values / df_sorted["max_score"].values * 100
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                next_x = len(x)
                next_y = coeffs[0] * next_x + coeffs[1]
                next_y = float(np.clip(next_y, 0, 100))
                st.write(f"📈 Predicted next severity (overall): **{next_y:.1f}% of max**")
                st.progress(int(next_y))
            else:
                st.write("Not enough screenings yet to predict trend.")
        except Exception:
            st.write("Could not compute prediction from existing data.")

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
else:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header(TEXT["nav_coach"])
    st.markdown(f"<p class='small-muted'>{TEXT['coach_intro']}</p>", unsafe_allow_html=True)

    df = load_safe_csv(LOG_PATH)
    last_label = None
    last_risk = None
    if not df.empty:
        last = df.iloc[-1]
        last_label = last.get("label", None)
        last_risk = last.get("risk", None)

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
        # generic label
        base_label = f"{severity_choice} level"

    if st.button(TEXT["coach_btn"]):
        # simple rule-based guidance
        st.markdown("<div class='coach-card'>", unsafe_allow_html=True)
        st.write(f"**Current severity:** {base_label}")

        if "Minimal" in base_label:
            st.write(
                "- Keep following your healthy habits (sleep, food, exercise).\n"
                "- Stay connected with people who make you feel safe.\n"
                "- Repeat screening once in a while to monitor changes."
            )
        elif "Mild" in base_label:
            st.write(
                "- Add 10–20 minutes of walking or light exercise daily.\n"
                "- Try basic breathing exercises or short meditation.\n"
                "- Write down your thoughts in a journal to clear your mind.\n"
                "- Talk with a trusted friend or family member about how you feel."
            )
        elif "Moderate" in base_label:
            st.write(
                "- Prioritize tasks and reduce overload where possible.\n"
                "- Fix a regular sleep and wake-up time.\n"
                "- Avoid too much caffeine, nicotine and scrolling late at night.\n"
                "- Consider booking an appointment with a counselor or psychologist."
            )
        else:  # Severe or higher
            st.write(
                "- Your symptoms seem strong. You deserve proper support.\n"
                "- Please reach out to a licensed mental-health professional soon.\n"
                "- If you have thoughts of self-harm or feel unsafe, contact\n"
                "  emergency services or a crisis helpline immediately.\n"
                "- Share how you feel with someone you trust right now."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
