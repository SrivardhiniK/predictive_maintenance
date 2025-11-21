import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/optimized_lstm_tf220.keras"
SCALER_X  = "models/scaler_X.pkl"
SCALER_Y  = "models/scaler_y.pkl"





print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X)
scaler_y = joblib.load(SCALER_Y)
print("Model loaded successfully!")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw = request.json.get("data")

        if not isinstance(raw, list) or not isinstance(raw[0], list):
            return jsonify({"error": "Input must be list of lists"}), 400

        X = np.array(raw, dtype=np.float32)

        # scale
        X_scaled = scaler_X.transform(X)

# FIX: reshape correctly for model expecting (batch, 1, 38)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Predict
        pred_scaled = model.predict(X_scaled)
        preds = scaler_y.inverse_transform(pred_scaled).flatten()


        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
