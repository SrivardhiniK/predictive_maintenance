import tensorflow as tf

model = tf.keras.models.load_model('models/lstm_rul_model.keras')
print("Model input shape:", model.input_shape)
