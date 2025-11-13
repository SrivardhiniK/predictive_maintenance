import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =====================================================
# 1️⃣ Load and prepare data
# =====================================================
print("Loading preprocessed data...")
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\data\processed_train.csv")

# Drop missing values (if any)
data = data.dropna()

# Features and target
X = data.drop(columns=["RUL"]).values
y = data["RUL"].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM input: (samples, timesteps, features)
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =====================================================
# 2️⃣ Define model builder
# =====================================================
def build_lstm_model(units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(units, activation='tanh', input_shape=(X_train.shape[1], 1)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# =====================================================
# 3️⃣ Define hyperparameter grid
# =====================================================
param_grid = {
    "units": [32, 64, 128],
    "dropout_rate": [0.2, 0.3, 0.5],
    "learning_rate": [0.001, 0.0005]
}

results = []

# =====================================================
# 4️⃣ Manual tuning loop
# =====================================================
print("\n Starting manual hyperparameter tuning...\n")

for units in param_grid["units"]:
    for dropout_rate in param_grid["dropout_rate"]:
        for lr in param_grid["learning_rate"]:
            print(f"Training model with units={units}, dropout={dropout_rate}, lr={lr}...")
            
            model = build_lstm_model(units=units, dropout_rate=dropout_rate, learning_rate=lr)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            val_mae = min(history.history["val_mae"])
            print(f"Validation MAE: {val_mae:.4f}\n")
            
            results.append({
                "units": units,
                "dropout_rate": dropout_rate,
                "learning_rate": lr,
                "val_mae": val_mae
            })

results_df = pd.DataFrame(results)
best_params = results_df.loc[results_df["val_mae"].idxmin()]

print("\nManual tuning complete!")
print("Best Parameters Found:")
print(best_params)

import os

# Create 'results' folder if it doesn't exist
os.makedirs("results", exist_ok=True)

results_df.to_csv("results/hyperparameter_tuning_results.csv", index=False)
print("\nResults saved to results/hyperparameter_tuning_results.csv")
