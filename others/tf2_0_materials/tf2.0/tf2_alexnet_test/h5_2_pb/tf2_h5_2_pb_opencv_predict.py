import numpy as np
import cv2 as cv


model = cv.dnn.readNetFromTensorflow('./xor.pb')

# 0 ^ 0 =  [[0.01974997]]
model.setInput(np.array([[0, 0]]), name='dense_input')
print('0 ^ 0 = ', model.forward(outputName='dense_4/Sigmoid'))

# 0 ^ 1 =  [[0.99141496]]
model.setInput(np.array([[0, 1]]), name='dense_input')
print('0 ^ 1 = ', model.forward(outputName='dense_4/Sigmoid'))

# 1 ^ 0 =  [[0.9897714]]
model.setInput(np.array([[1, 0]]), name='dense_input')
print('1 ^ 0 = ', model.forward(outputName='dense_4/Sigmoid'))

# 1 ^ 1 =  [[0.00406971]]
model.setInput(np.array([[1, 1]]), name='dense_input')
print('1 ^ 1 = ', model.forward(outputName='dense_4/Sigmoid'))