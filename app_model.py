import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(page_title='AI‑based Mental Health Assessment', layout='wide')

# Top-level title
st.title('AI‑based Mental Health Assessment')
st.write("This app assists with screening for Anxiety, Stress, and Depression. It does not replace professional diagnosis.")

# Load models and encoders (with fallback if missing)
def load_model_and_encoder(target):
    models = {
        'Anxiety': 'best_model_Anxiety_Label_Logistic_Regression.joblib',
        'Stress': 'best_model_Stress_Label_Logistic_Regression.joblib',
        'Depression': 'best_model_Depression_Label_CatBoost.joblib'
    }
    encoders = {
        'Anxiety': 'final_anxiety_encoder.joblib',
        'Stress': 'final_stress_encoder.joblib',
        'Depression': 'final_depression_encoder.joblib'
    }
    model_file = models.get(target)
    encoder_file = encoders.get(target)
    model = joblib.load(model_file) if os.path.exists(model_file) else None
    encoder = joblib.load(encoder_file) if os.path.exists(encoder_file) else None
    return model, encoder

# Map numeric output to qualitative labels if encoder is missing
def fallback_label(pred, target):
    mapping = {
        'Anxiety': ["Minimal Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"],
        'Stress':  ["Minimal Stress",  "Mild Stress",  "Moderate Stress",  "Severe Stress"],
        'Depression': ["Minimal Depression", "Mild Depression", "Moderate Depression", "Severe Depression"]
    }
    labels = mapping.get(target, [])
    # ensure pred is int index
    idx = int(pred) % len(labels) if labels else 0
    return labels[idx]

# Determine risk tier from label
def risk_tier(label):
    if 'Severe' in label: return 'Critical'
    elif 'Moderate' in label: return 'High'
    elif 'Mild' in label: return 'Moderate'
    elif 'Minimal' in label: return 'Low'
    return 'Unknown'

# Suggested actions based on risk level
def suggested_actions(tier):
    suggestions = {
        'Low': 'Maintain healthy lifestyle; sleep 7–9 hours; practice regular exercise and mindfulness.',
        'Moderate': 'Incorporate stress‑reduction techniques; moderate exercise; balanced diet; journaling.',
        'High': 'Seek counseling support; consider reducing workload; practice daily relaxation and meditation.',
        'Critical': 'Seek professional help immediately; reach out to trusted support network or hotlines.'
    }
    return suggestions.get(tier, '')

# Define screening questions
questions = {
    'Anxiety': [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless it’s hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ],
    'Stress': [
        "Upset because of unexpected events",
        "Unable to control important things",
        "Felt nervous and stressed",
        "Confident about handling problems",
        "Things going your way",
        "Could not cope with all the tasks you had to do",
        "Able to control irritations in your life",
        "Felt on top of things",
        "Angry because things were out of control",
        "Felt difficulties piling up too high"
    ],
    'Depression': [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or feeling like a failure",
        "Trouble concentrating on things",
        "Moving/speaking slowly or restlessness",
        "Thoughts of self‑harm or death"
    ]
}

# Sidebar: choose page (screening or dashboard)
page = st.sidebar.radio('Navigation', options=['Screening', 'Dashboard'])
target = st.sidebar.selectbox('Choose Assessment', options=['Anxiety','Stress','Depression'])

# Global log file to store predictions
LOG_PATH = 'prediction_log.csv'

if page == 'Screening':
    st.header(f'{target} Screening Form')

    # Display the scale legend on the right
    with st.sidebar:
        st.subheader('Scale Meaning (1–5)')
        st.write('1 — Not at all')
        st.write('2 — Several days')
        st.write('3 — Half the days')
        st.write('4 — Nearly every day')
        st.write('5 — Almost always')

    # Show sliders for the selected assessment
    responses = []
    for idx, q in enumerate(questions[target], 1):
        responses.append(st.slider(f'{idx}. {q}', min_value=1, max_value=5, value=3))

    if st.button('Predict Mental Health Status'):
        try:
            model, encoder = load_model_and_encoder(target)
            df = pd.DataFrame([responses])

            # Some models expect specific column names; fallback by adding missing columns if needed
            if hasattr(model, 'feature_names_in_'):
                expected_cols = list(model.feature_names_in_)
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = 0
                df = df[expected_cols]

            # Use the model to predict; handle case when model is None
            if model is not None:
                pred = model.predict(df)[0]
                label = encoder.inverse_transform([pred])[0] if encoder is not None else fallback_label(pred, target)
            else:
                # If model cannot be loaded, use simple scoring average to approximate severity
                average_score = np.mean(responses)
                pred_idx = int((average_score - 1) / 4 * 3.99)  # scale 1–5 to 0–3 index
                label = fallback_label(pred_idx, target)

            tier = risk_tier(label)
            actions = suggested_actions(tier)

            st.success(f'Predicted: {label}')
            st.info(f'Risk Level: {tier}')
            if actions:
                st.markdown(f'**Suggested Actions:** {actions}')

            # Save prediction to log
            log_row = {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'target': target,
                'predicted_label': label,
                'risk_tier': tier
            }
            log_df = pd.DataFrame([log_row])
            if os.path.exists(LOG_PATH):
                log_df.to_csv(LOG_PATH, mode='a', header=False, index=False)
            else:
                log_df.to_csv(LOG_PATH, index=False)

        except Exception as e:
            st.error(f'Prediction failed: {e}')

elif page == 'Dashboard':
    st.header('Mental Health Analytics Dashboard')
    if not os.path.exists(LOG_PATH):
        st.write('No predictions yet. Please complete a screening.')
    else:
        log_df = pd.read_csv(LOG_PATH)
        st.write(log_df.tail(10))
        st.subheader('Distribution of Risk Levels')
        risk_counts = log_df['risk_tier'].value_counts()
        st.bar_chart(risk_counts)

# Footer
st.markdown(
    '<hr style="border-top:1px solid #666;"><p style="text-align:center;color:grey;">© 2025 Team Dual Core. All rights reserved.</p>',
    unsafe_allow_html=True)
