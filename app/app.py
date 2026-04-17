import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# LOAD FILES
# ===============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("feature_cols.pkl", "rb"))
importance = pd.read_csv("importance.csv")

# ===============================
# UI
# ===============================
st.title("💓 CVD Risk Prediction App")

st.write("Enter patient details:")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

import pickle
pickle.dump(model, open("models/model.pkl", "wb"))

import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = pickle.load(open(model_path, "rb"))

# ===============================
# PREDICT
# ===============================
if st.button("Predict Risk"):

    df = pd.DataFrame([input_data])

    try:
        df_scaled = scaler.transform(df)
    except:
        df_scaled = df.values

    prob = model.predict_proba(df_scaled)[0][1]

    # Risk level
    if prob < 0.3:
        level = "🟢 Low"
    elif prob < 0.6:
        level = "🟡 Moderate"
    else:
        level = "🔴 High"

    st.subheader(f"Risk Score: {prob:.2f}")
    st.subheader(f"Risk Level: {level}")

    st.write("### Top Risk Factors")
    st.dataframe(importance.head(5))
