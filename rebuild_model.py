import tensorflow as tf

# Paths
OLD_MODEL = r"C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\models\lstm_predictive_maintenance.h5"
NEW_MODEL = r"models/optimized_lstm_clean.keras"

print("Loading .h5 model...")
old_model = tf.keras.models.load_model(OLD_MODEL)
weights = old_model.get_weights()

print("Building clean model with shape (1, 38)...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 38)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="linear")
])

model.set_weights(weights)

model.save(NEW_MODEL)

print("\nClean model saved as:", NEW_MODEL)
