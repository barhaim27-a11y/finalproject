
# app.py - Streamlit application for Parkinson's Prediction

import streamlit as st
import numpy as np
import joblib

# טען את המודל המאומן
model = joblib.load("model.pkl")

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Prediction")
st.write("הזן את פרטי הקול של המטופל כדי לבדוק אם קיימת סבירות למחלת פרקינסון.")

# רשימת התכונות שהמודל דורש
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# קבלת קלט מהמשתמש
user_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0, format="%0.4f")
    user_input.append(value)

# לחצן להרצת התחזית
if st.button("🔍 נבא עכשיו"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ קיים חשד למחלת פרקינסון (סבירות: {proba:.2%})")
    else:
        st.success(f"✅ לא זוהו סימנים למחלת פרקינסון (סבירות: {proba:.2%})")

    # גרף עמודה קטן של הסבירות
    st.subheader("📊 סבירות למחלה")
    st.bar_chart({"Probability": [proba]})
