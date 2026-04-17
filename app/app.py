import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================
# LOAD MODEL SAFELY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model, scaler, feature_names = pickle.load(open(model_path, "rb"))

# =========================
# APP UI
# =========================
st.set_page_config(page_title="CVD Risk Predictor", page_icon="❤️")

st.title("❤️ Cardiovascular Risk Prediction")
st.markdown("Predict your heart disease risk using clinical + lifestyle data")

# =========================
# INPUTS
# =========================
st.subheader("Enter Patient Details")

age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chol = st.number_input("Cholesterol", 100, 400, 200)
bp = st.number_input("Blood Pressure", 80, 200, 120)

steps = st.slider("Daily Steps", 0, 20000, 5000)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.5)
hr = st.slider("Average Heart Rate", 40, 120, 70)

model = pickle.load(open(model_path, "rb"))

# Dummy fallback
scaler = None
feature_names = [
    "age","sex","chol","trestbps",
    "steps","sleep_hours","avg_hr",
    "lifestyle_score","stress_index"
]

# Encode
sex = 1 if sex == "Male" else 0

# =========================
# FEATURE ENGINEERING
# =========================
lifestyle_score = (steps * 0.3) + (sleep * 0.3) + (1/(hr+1) * 0.4)
stress_index = hr / (sleep + 1)

# =========================
# BUILD INPUT DATAFRAME
# =========================
input_dict = {
    "age": age,
    "sex": sex,
    "chol": chol,
    "trestbps": bp,
    "steps": steps,
    "sleep_hours": sleep,
    "avg_hr": hr,
    "lifestyle_score": lifestyle_score,
    "stress_index": stress_index
}

# Fill missing features safely
for col in feature_names:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[feature_names]

# =========================
# PREDICT
# =========================
if st.button("🔍 Predict Risk"):

    try:
        if scaler:
            X_scaled = scaler.transform(input_df)
        else:
            X_scaled = input_df.values

        # Risk category
        if risk < 0.3:
            level = "Low Risk 🟢"
        elif risk < 0.6:
            level = "Moderate Risk 🟡"
        else:
            level = "High Risk 🔴"

        st.subheader("Result")
        st.metric("Risk Score", f"{risk:.2f}")
        st.success(level)

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Machine Learning + Wearable Data")
