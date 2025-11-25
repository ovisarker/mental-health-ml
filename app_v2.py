import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
from datetime import datetime

# ------------------------------------------------------------------
# ⚙️ Page config + global styles
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI-based Mental Health Detection System",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
<style>
body {background-color:#F4F7FB;color:#111827;}
h1,h2,h3,h4,h5 {color:#111827;}

.main-card {
    background-color:#FFFFFF;
    padding:24px 26px;
    border-radius:18px;
    box-shadow:0 8px 18px rgba(15,23,42,0.08);
    margin-bottom:18px;
}

.scale-card {
    background:#E8F2FF;
    padding:14px 16px;
    border-radius:14px;
    border:1px solid #C5DAFF;
    font-size:0.88rem;
}
.scale-title {
    font-weight:700;
    margin-bottom:4px;
}
.scale-item {
    margin:0;
    padding:0;
}

.badge-pill {
    display:inline-flex;
    align-items:center;
    padding:3px 10px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:600;
    background:#EEF2FF;
    color:#4338CA;
    margin-left:6px;
}

.result-badge {
    padding:8px 14px;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    font-weight:600;
    font-size:0.9rem;
}
.result-low {background:#DCFCE7;color:#166534;}
.result-mod {background:#FEF9C3;color:#854D0E;}
.result-high {background:#FFEDD5;color:#9A3412;}
.result-crit {background:#FEE2E2;color:#991B1B;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 🔧 Heuristic scoring (fallback / main logic)
# ------------------------------------------------------------------
def fallback_label_from_responses(responses, target: str) -> str:
    """
    Compute severity label based on standard clinical cut-offs.

    Sliders are 1–5. We map them to 0–3 or 0–4 and sum.
    """

    # Anxiety (GAD-7 style: 7 items, each 0–3, total 0–21)
    if target == "Anxiety":
        scaled = [max(0, min(3, v - 1)) for v in responses]  # 0–3
        total = sum(scaled)
        if total <= 4:
            level = "Minimal"
        elif total <= 9:
            level = "Mild"
        elif total <= 14:
            level = "Moderate"
        else:
            level = "Severe"
        return f"{level} Anxiety"

    # Depression (PHQ-9 style: 9 items, each 0–3, total 0–27)
    if target == "Depression":
        scaled = [max(0, min(3, v - 1)) for v in responses]
        total = sum(scaled)
        if total <= 4:
            level = "Minimal"
        elif total <= 9:
            level = "Mild"
        elif total <= 14:
            level = "Moderate"
        else:
            level = "Severe"
        return f"{level} Depression"

    # Stress (PSS-10 style: 10 items, each 0–4, total 0–40)
    # Here our sliders are already 1–5, so map to 0–4
    scaled = [max(0, min(4, v - 1)) for v in responses]
    total = sum(scaled)
    if total <= 13:
        level = "Minimal"
    elif total <= 26:
        level = "Moderate"
    else:
        level = "Severe"
    return f"{level} Stress"


def risk_tier_map(label: str) -> str:
    """Map severity label to risk tier."""
    lower = label.lower()
    if "minimal" in lower:
        return "Low"
    if "mild" in lower:
        return "Moderate"
    if "moderate" in lower:
        return "High"
    if "severe" in lower:
        return "Critical"
    return "Unknown"


def save_prediction_log(row: dict) -> None:
    df = pd.DataFrame([row])
    log_path = "prediction_log.csv"
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def safe_predict(responses, target: str) -> str:
    """
    For now we use the clinically-inspired heuristic for all predictions.
    If in future you want to plug in ML models again, you can modify here.
    """
    return fallback_label_from_responses(responses, target)

# ------------------------------------------------------------------
# 🧭 Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a page", ["🧩 Prediction", "📊 Dashboard"])

# ------------------------------------------------------------------
# 🧩 Prediction Page (with nicer layout)
# ------------------------------------------------------------------
if page == "🧩 Prediction":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.subheader("🧠 AI-based Mental Health Detection & Support System")
    st.caption("Developed for thesis & real-world use | Uses clinically inspired scoring")

    target = st.selectbox("What would you like to screen for?", ["Anxiety", "Stress", "Depression"])

    # Question sets
    question_sets = {
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
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or feeling like a failure",
            "Trouble concentrating on things",
            "Moving/speaking slowly or restlessness",
            "Thoughts of self-harm or death",
        ],
    }

    # 1–5 meaning for each scale, shown in a corner card
    scale_labels = {
        "Anxiety": [
            "1 – Not at all",
            "2 – Several days",
            "3 – More than half the days",
            "4 – Nearly every day",
            "5 – Almost every day",
        ],
        "Depression": [
            "1 – Not at all",
            "2 – Several days",
            "3 – More than half the days",
            "4 – Nearly every day",
            "5 – Almost every day",
        ],
        "Stress": [
            "1 – Never",
            "2 – Almost never",
            "3 – Sometimes",
            "4 – Fairly often",
            "5 – Very often",
        ],
    }

    st.markdown("### 🧾 {} Screening Form".format(target))
    # top row: left = intro, right = 1–5 scale card
    left_col, right_col = st.columns([3, 1], vertical_alignment="top")

    with left_col:
        st.info("Rate each statement from **1 (lowest)** to **5 (highest)** based on your experience over the last 2 weeks.")

    with right_col:
        items_html = "".join(
            f"<li class='scale-item'>{txt}</li>" for txt in scale_labels[target]
        )
        st.markdown(
            f"""
            <div class="scale-card">
              <div class="scale-title">
                1–5 Response Scale
                <span class="badge-pill">{target}</span>
              </div>
              <ul style="padding-left:18px;margin-top:6px;">
                {items_html}
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # small spacing

    # Sliders in the left column (form area)
    responses = []
    with left_col:
        for idx, question in enumerate(question_sets[target]):
            slider_key = f"{target}_q{idx}"
            val = st.slider(
                label=question,
                min_value=1,
                max_value=5,
                value=3,
                key=slider_key,
            )
            responses.append(val)

    # Prediction button
    if st.button("🔍 Predict Mental Health Status"):
        label = safe_predict(responses, target)
        risk = risk_tier_map(label)

        # choose badge class
        if risk == "Low":
            badge_class = "result-low"
        elif risk == "Moderate":
            badge_class = "result-mod"
        elif risk == "High":
            badge_class = "result-high"
        elif risk == "Critical":
            badge_class = "result-crit"
        else:
            badge_class = "result-mod"

        st.markdown("---")

        # Predicted label
        st.markdown(
            f"""
            <div class="result-badge {badge_class}">
                🎯 Predicted: {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk level
        st.markdown(
            f"""
            <div style="margin-top:10px;" class="result-badge {badge_class}">
                🩺 Risk Level: {risk}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested actions
        actions_text = {
            "Low": "Maintain a healthy routine • Sleep 7–9 hours • Practice simple relaxation breathing.",
            "Moderate": "Consider light exercise, journaling, talking with a trusted friend or counselor.",
            "High": "Reduce workload where possible • Talk to a mental-health professional soon • Practice mindfulness.",
            "Critical": "Strongly consider speaking with a licensed mental-health professional or helpline immediately • Lean on your support network.",
        }.get(risk, "Monitor your mental health regularly and repeat this screening when needed.")

        st.markdown(f"**Suggested Actions:** {actions_text}")

        # Save log
        save_prediction_log(
            {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": target,
                "predicted_label": label,
                "risk_tier": risk,
            }
        )

    st.markdown("</div>", unsafe_allow_html=True)  # end main-card

# ------------------------------------------------------------------
# 📊 Dashboard Page
# ------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📊 Mental Health Analytics Dashboard")

    log_path = "prediction_log.csv"
    if not os.path.exists(log_path):
        st.warning("No predictions have been made yet. Please perform a screening first.")
    else:
        df = pd.read_csv(log_path)
        st.dataframe(df.tail(10), use_container_width=True)

        st.markdown("### 📈 Risk Distribution Overview")

        tiers = df["risk_tier"].value_counts(normalize=True).mul(100)
        colors = {
            "Low": "#22C55E",
            "Moderate": "#EAB308",
            "High": "#F97316",
            "Critical": "#EF4444",
        }
        for tier in ["Low", "Moderate", "High", "Critical"]:
            val = float(tiers.get(tier, 0))
            st.markdown(
                f"<span style='font-weight:600;color:{colors[tier]}'>{tier}: {val:.1f}%</span>",
                unsafe_allow_html=True,
            )
            st.progress(int(val))

        df["datetime"] = pd.to_datetime(df["datetime"])
        trend = df.groupby(df["datetime"].dt.date).size().reset_index(name="Predictions")

        st.markdown("### ⏱ Screenings over time")
        st.altair_chart(
            alt.Chart(trend).mark_line(point=True, color="#0EA5E9").encode(
                x="datetime:T",
                y="Predictions:Q",
            ),
            use_container_width=True,
        )

        dist = df["risk_tier"].value_counts().reset_index()
        dist.columns = ["Risk Tier", "Count"]

        st.markdown("### 🧪 Risk tier counts")
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x=alt.X("Risk Tier:N", sort="-y"),
                y="Count:Q",
                color="Risk Tier:N",
            ),
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Prediction Log",
            data=df.to_csv(index=False),
            file_name="prediction_log.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
