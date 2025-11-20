import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Predictive Maintenance RUL")
st.title("🔧 Remaining Useful Life Prediction")

API_URL = "http://localhost:5000/predict"

uploaded = st.file_uploader("Upload CSV (38 columns)", type=["csv"])

def classify_rul(x):
    if x > 125: return "🟢 Healthy"
    if x > 60: return "🟡 Warning"
    return "🔴 Critical"

if uploaded:
    try:
        # 1) READ CSV WITHOUT HEADER
        df_raw = pd.read_csv(uploaded, header=None)

        # 2) REMOVE HEADER ROW IF FOUND
        if str(df_raw.iloc[0, 0]).lower() == "f1":
            df_raw = df_raw.iloc[1:]

        # 3) TRIM TO EXACTLY 38 FEATURES
        df_raw = df_raw.iloc[:, :38]

        # 4) RENAME COLUMNS AS f1..f38
        df_raw.columns = [f"f{i}" for i in range(1, 39)]

        # 5) CONVERT EVERYTHING TO FLOAT
        df = df_raw.astype(float)

        st.write("📊 Preview:")
        st.dataframe(df.head())

        # 6) SEND CLEAN DATA TO API
        if st.button("Predict"):
            # Convert to list of lists — CORRECT FORMAT
            data = df.values.tolist()   # NOT flatten

            st.write("DEBUG FIRST ROW:", data[0])
            st.write("DEBUG LENGTH OF FIRST ROW:", len(data[0]))

            payload = {"data": data}


            st.write("DEBUG: First row being sent:", payload["data"][0])

            response = requests.post(API_URL, json=payload).json()

            if "predictions" in response:
                preds = np.array(response["predictions"])
                df["Predicted_RUL"] = preds
                df["Status"] = df["Predicted_RUL"].apply(classify_rul)

                st.success("Prediction Successful!")
                st.dataframe(df)
            else:
                st.error(response.get("error", "Unknown error"))

    except Exception as e:
        st.error(f"Error: {e}")
