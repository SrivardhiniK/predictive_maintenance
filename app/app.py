import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib, os

st.set_page_config(page_title="Predictive Maintenance RUL", layout="wide")
st.title("🔧 Predict Remaining Useful Life (RUL)")

MODEL_DIR = "models"
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "optimized_lstm.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "lstm_predictive_maintenance.h5")
SCALER_X = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y = os.path.join(MODEL_DIR, "scaler_y.pkl")

# ---------------------------------------------------
# LOAD MODEL + SCALERS
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_file = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else MODEL_PATH_H5
    model = tf.keras.models.load_model(model_file)
    scaler_x = joblib.load(SCALER_X)
    scaler_y = joblib.load(SCALER_Y)
    return model, scaler_x, scaler_y

model, scaler_X, scaler_y = load_model()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
st.subheader("📁 Upload Sensor Data CSV *(NO RUL column)*")
uploaded = st.file_uploader("Choose file", type=["csv"])

def classify_rul(x):
    if x > 125: return "🟢 Healthy"
    elif x > 60: return "🟡 Warning"
    return "🔴 Critical"

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("📊 Input Preview:", df.head())

        X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean()).values.astype(np.float32)

        # SCALE → RESHAPE CORRECTLY
        X_scaled = scaler_X.transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        pred_scaled = model.predict(X_scaled)
        preds = scaler_y.inverse_transform(pred_scaled).flatten()

        df["Predicted_RUL"] = preds
        df["Status"] = df["Predicted_RUL"].apply(classify_rul)

        st.success("✔ Prediction Successful!")
        st.dataframe(df)

        st.download_button("⬇ Download Results CSV", df.to_csv(index=False), file_name="predicted_rul.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
