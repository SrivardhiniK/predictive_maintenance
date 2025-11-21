import tensorflow as tf

old_model_path = r"models/optimized_lstm.keras"
new_model_path = r"models/optimized_lstm_tf220.keras"

print("Loading model with TF 2.20...")
model = tf.keras.models.load_model(old_model_path)

print("Saving model in new TF 2.20 format...")
model.save(new_model_path)

print("Done!")
