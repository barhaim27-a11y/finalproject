
import streamlit as st
import numpy as np
import joblib

# טען את המודל המאומן
model = joblib.load("model.pkl")

# הגדרות עמוד
st.set_page_config(page_title="Parkinson's Predictor", layout="centered")

# כותרת מעוצבת
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>הזן את פרמטרי הקול שלך ולקבל תחזית אם קיימת סבירות לפרקינסון.</p>", unsafe_allow_html=True)
st.markdown("---")

# סרגל צד
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1785/1785363.png", width=100)
    st.title("🧪 מידע על האפליקציה")
    st.markdown("""• מבוסס על דאטה אמיתי מ־UCI  
• נעשה שימוש ב־Random Forest  
• הדגמה אינטראקטיבית""")

# רשימת הפיצ'רים שהמודל צריך
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# טופס קלט בשלוש עמודות
st.subheader("🔧 הזן ערכים (ניתן להשאיר ברירת מחדל)")
user_input = []
cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        value = st.number_input(f"{feature}", value=0.0, format="%0.4f", key=feature)
        user_input.append(value)

# תחזית
if st.button("🔍 נבא עכשיו"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.markdown(f"<div style='background-color:#FFCCCC;padding:20px;border-radius:10px;'><h3>⚠️ קיים חשד לפרקינסון</h3><b>סבירות: {proba:.2%}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#CCFFCC;padding:20px;border-radius:10px;'><h3>✅ אין סימנים לפרקינסון</h3><b>סבירות: {proba:.2%}</b></div>", unsafe_allow_html=True)

    st.subheader("📊 סיכוי למחלה")
    st.progress(proba)

# תחתית מעוצבת
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<center><small>Final ML & AI Project – 2025 | by OpenAI Assistant</small></center>", unsafe_allow_html=True)
