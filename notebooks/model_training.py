# notebooks/model_training.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1️⃣ Load processed dataset
data = pd.read_csv(r'C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\data\processed_train.csv')

# 2️⃣ Check if target exists
if 'RUL' not in data.columns:
    raise KeyError("'target' column not found in data. Please ensure preprocessing step added it.")

# 3️⃣ Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data.drop('RUL', axis=1))
y_scaled = data['RUL'].values

# 4️⃣ Reshape for LSTM [samples, timesteps, features]
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# 5️⃣ Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 6️⃣ Build Model
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# 7️⃣ Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 8️⃣ Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# 9️⃣ Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 🔟 Save model
model.save('models/optimized_lstm.keras')
print(" Optimized LSTM model trained and saved successfully!")
