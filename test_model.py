# test_model.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the model
model = tf.keras.models.load_model("mnist_model.h5")

# Load test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize


x_test = x_test.reshape(-1, 28, 28, 1)

prediction = model.predict(np.expand_dims(x_test[1], axis=0))
predicted_digit = np.argmax(prediction)

print(f"Predicted: {predicted_digit}, Actual: {y_test[1]}")
