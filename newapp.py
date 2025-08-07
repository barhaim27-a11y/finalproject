
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("model.pkl")

# Feature descriptions
feature_descriptions = {
    'MDVP:Fo(Hz)': "תדירות בסיסית (Hz)",
    'MDVP:Fhi(Hz)': "התדירות הגבוהה ביותר (Hz)",
    'MDVP:Flo(Hz)': "התדירות הנמוכה ביותר (Hz)",
    'MDVP:Jitter(%)': "שונות בתדר הדיבור באחוזים",
    'MDVP:Jitter(Abs)': "שונות מוחלטת בתדר",
    'MDVP:RAP': "שונות תדר ממוצעת",
    'MDVP:PPQ': "מדד יציבות תדר ממוצע",
    'Jitter:DDP': "שונות תדר כפולה",
    'MDVP:Shimmer': "שונות בעוצמת הקול",
    'MDVP:Shimmer(dB)': "שונות בעוצמה בדציבלים",
    'Shimmer:APQ3': "שונות ממוצעת ב־3 מחזורים",
    'Shimmer:APQ5': "שונות ממוצעת ב־5 מחזורים",
    'MDVP:APQ': "שונות כללית בעוצמה",
    'Shimmer:DDA': "שונות דיפרנציאלית",
    'NHR': "יחס רעש/הרמוניה",
    'HNR': "יחס הרמוניה/רעש",
    'RPDE': "מדד אי־דטרמיניסטיות",
    'DFA': "נפח ורעש בתדירות נמוכה",
    'spread1': "סטייה מהתדירות האידיאלית",
    'spread2': "סטייה מורחבת",
    'D2': "מידת מורכבות של הקול",
    'PPE': "מדד אקראיות הקול"
}

features = list(feature_descriptions.keys())

st.set_page_config(page_title="Parkinson's App", layout="centered")
st.sidebar.title("📚 ניווט")
app_mode = st.sidebar.radio("בחר פעולה:", [
    "🔍 תחזית בודדת", 
    "📤 חיזוי לפי CSV", 
    "📊 גרפים ודשבורד", 
    "🔁 אימון מחדש", 
    "ℹ️ הסבר על מאפיינים"
])

# ---------------- PAGE 1 ----------------
if app_mode == "🔍 תחזית בודדת":
    st.title("🔍 תחזית בודדת עם הסברים")
    user_input = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            value = st.number_input(f"{feature}", help=feature_descriptions[feature], value=0.0, format="%0.4f")
            user_input.append(value)
    if st.button("🔮 נבא"):
        X_input = np.array(user_input).reshape(1, -1)
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]
        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ חשד למחלת פרקינסון | סבירות: {proba:.2%}")
        else:
            st.success(f"✅ אין סימנים למחלת פרקינסון | סבירות: {proba:.2%}")
        st.progress(proba)

# ---------------- PAGE 2 ----------------
elif app_mode == "📤 חיזוי לפי CSV":
    st.title("📤 העלאת קובץ CSV")
    uploaded_file = st.file_uploader("העלה קובץ עם פיצ'רים", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing = set(features) - set(df.columns)
            if missing:
                st.error(f"חסרות עמודות: {', '.join(missing)}")
            else:
                preds = model.predict(df)
                probs = model.predict_proba(df)[:, 1]
                df["Prediction"] = preds
                df["Probability"] = np.round(probs * 100, 2).astype(str) + "%"
                st.success("✔️ תחזיות הושלמו")
                st.dataframe(df)
                st.download_button("📥 הורד תוצאות", data=df.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"שגיאה: {e}")

# ---------------- PAGE 3 ----------------
elif app_mode == "📊 גרפים ודשבורד":
    st.title("📊 ניתוח ביצועים")
    uploaded = st.file_uploader("📥 העלה קובץ עם עמודת 'status'", type="csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "status" not in df.columns:
                st.error("עמודת 'status' נדרשת")
            else:
                X = df[features]
                y_true = df["status"]
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                # ROC
                st.subheader("📈 ROC Curve")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.legend()
                st.pyplot(fig)

                # Confusion matrix
                st.subheader("📉 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig2, ax2 = plt.subplots()
                ConfusionMatrixDisplay(cm).plot(ax=ax2)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"שגיאה: {e}")

# ---------------- PAGE 4 ----------------
elif app_mode == "🔁 אימון מחדש":
    st.title("🔁 אימון מחדש של המודל")
    st.info("העלה קובץ CSV עם עמודת `status` כדי לאמן מחדש את המודל.")
    uploaded = st.file_uploader("📥 קובץ אימון", type="csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "status" not in df.columns:
                st.error("עמודת 'status' חסרה")
            else:
                X = df[features]
                y = df["status"]
                if st.button("🚀 התחל אימון"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
                    ])
                    pipeline.fit(X_train, y_train)
                    acc = accuracy_score(y_test, pipeline.predict(X_test))
                    joblib.dump(pipeline, "model.pkl")
                    st.success(f"✅ המודל עודכן | דיוק: {acc:.2%}")
        except Exception as e:
            st.error(f"שגיאה: {e}")

# ---------------- PAGE 5 ----------------
elif app_mode == "ℹ️ הסבר על מאפיינים":
    st.title("ℹ️ הסבר על כל פיצ'ר")
    for feature in features:
        st.markdown(f"**{feature}**: {feature_descriptions[feature]}")
