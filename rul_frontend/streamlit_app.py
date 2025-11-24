import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="RUL Prediction Dashboard")

st.title("🔧 Remaining Useful Life – Prediction UI")

API_URL = "YOUR_RAILWAY_BACKEND_URL/predict"   # <-- UPDATE THIS

uploaded = st.file_uploader("Upload CSV (38 columns)", type=["csv"])

def classify_rul(x):
    if x > 125:
        return "🟢 Healthy"
    if x > 60:
        return "🟡 Warning"
    return "🔴 Critical"

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded, header=None)

        # If header contains "f1", drop it
        if str(df_raw.iloc[0, 0]).lower() == "f1":
            df_raw = df_raw.iloc[1:]

        # Force first 38 columns only
        df_raw = df_raw.iloc[:, :38]
        df_raw.columns = [f"f{i}" for i in range(1, 39)]

        # Convert to float
        df = df_raw.astype(float)

        # Clean data
        data = df.values.tolist()

        st.write("### Preview")
        st.dataframe(df)

        payload = {"data": data}

        st.write("⏳ Sending to API...")

        response = requests.post(API_URL, json=payload)
        result = response.json()

        if "predictions" in result:
            preds = np.array(result["predictions"])
            df["Predicted_RUL"] = preds
            df["Status"] = df["Predicted_RUL"].apply(classify_rul)

            st.success("Prediction successful!")
            st.dataframe(df)

        else:
            st.error(result.get("error", "Unknown API error"))

    except Exception as e:
        st.error(f"Error: {e}")
