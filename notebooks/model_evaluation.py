# src/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# 1️⃣ Load the processed dataset
data = pd.read_csv(r'C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\data\processed_train.csv')

# 2️⃣ Separate features and target
X = data.drop('RUL', axis=1).values
y = data['RUL'].values

# 3️⃣ Scale features (same as in training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Reshape for LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
# 5️⃣ Load trained model
model = tf.keras.models.load_model(r'C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\models\lstm_rul_model.keras')

# 6️⃣ Predict
y_pred = model.predict(X_scaled).flatten()
y_pred = np.nan_to_num(y_pred)  # replaces NaN with 0 or finite numbers

# 7️⃣ Calculate metrics
print("Checking for NaNs...")
print("NaNs in y:", np.isnan(y).sum())
print("NaNs in y_pred:", np.isnan(y_pred).sum())

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n MODEL EVALUATION METRICS ")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8️⃣ Plot Actual vs Predicted RUL
plt.figure(figsize=(10,5))
plt.plot(y[:200], label='Actual RUL', color='blue')
plt.plot(y_pred[:200], label='Predicted RUL', color='red', linestyle='--')
plt.title('Actual vs Predicted Remaining Useful Life')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.legend()
plt.grid(True)
plt.show()

# 9️⃣ Plot Error Distribution
errors = y - y_pred
plt.figure(figsize=(8,5))
plt.hist(errors, bins=30, color='purple', alpha=0.7)
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
