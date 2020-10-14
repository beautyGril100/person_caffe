import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./xor.h5')

# 0 ^ 0 =  [[0.01974997]]
print('0 ^ 0 = ', model.predict(np.array([[0, 0]])))

# 0 ^ 1 =  [[0.99141496]]
print('0 ^ 1 = ', model.predict(np.array([[0, 1]])))

# 1 ^ 0 =  [[0.9897714]]
print('1 ^ 0 = ', model.predict(np.array([[1, 0]])))

# 1 ^ 1 =  [[0.00406971]]
print('1 ^ 1 = ', model.predict(np.array([[1, 1]])))