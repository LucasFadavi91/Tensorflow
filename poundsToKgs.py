import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

pounds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
kgs = np.array([0.4536, 0.9072, 1.3608, 1.8144, 2.2680, 2.7216, 3.1751, 3.6287, 4.0823, 4.5359], dtype=float)

layer = tf.keras.layers.Dense(units=3, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Training...")
record = model.fit(pounds, kgs, epochs=100, verbose=False)
print("Successful")

plt.xlabel("# Epoch")
plt.ylabel("Magnitude of loss")
plt.plot(record.history["loss"])

print("Prediction!")
result = model.predict([198.416]) #number in pounds
print("Result:" + str(result) + "Kgs")

print("Internal variables of model")
print(layer.get_weights())
