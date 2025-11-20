import numpy as np
import tensorflow as tf

print("TF version:", tf.__version__)

m = tf.keras.models.load_model(r"C:\Users\ADMIN\OneDrive\Desktop\predictive_maintenance\notebooks\models\final_clean_rul_model.h5")
print("Model loaded successfully!")
m.summary()

dummy = np.zeros((1,1,38), dtype=np.float32)
print("Prediction:", m.predict(dummy))
