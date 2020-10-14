# coding=gbk
import numpy as np
import tensorflow as tf
import os
from PIL import Image

frozen_graph_path = '/workspace/hwzhu/tf2.0/tf_lite_model/mobilenet_v1_0.5_224/mobilenet_v1_0.5_224_frozen.pb'
inputs = ['input']
outputs = ['MobilenetV1/Predictions/Softmax']


# full uint8
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(frozen_graph_path, inputs, outputs)
converter.inference_type = tf.uint8
converter.quantized_input_stats = {'input':(127.0, 127.0)}
converter.default_ranges_stats = (0, 255)
#converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]  #is equal to 'converter.post_training_quantize = True'

tflite_model = converter.convert()

open("mobilenet_v1_0.5_224_Full_integer.tflite", "wb").write(tflite_model)

'''
# full float
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(frozen_graph_path, inputs, outputs)
converter.inference_type = tf.float32
tflite_model = converter.convert()

open("mobilenet_v1_0.5_224_float.tflite", "wb").write(tflite_model)
'''

# load TFLite model and then allocate tensor
# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.compat.v1.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# get input and output tensor
# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input_details:', input_details)
print('output_details:',output_details)
# using random data as input, then test TensorFlow Lite model
# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# function 'get_tensor()' will return a copy of a tensor
# using 'tensor()' to get a pointer that pointing to a tensor
# 函数 `get_tensor()` 会返回一份张量的拷贝。
# 使用 `tensor()` 获取指向张量的指针。
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# testing TensorFlow model with random data
# 使用随机数据作为输入测试 TensorFlow 模型。
'''
tf_results = model(tf.constant(input_data))

# compare results
# 对比结果。
for tf_result, tflite_result in zip(tf_results, tflite_results):
  print('tf_results:')
  print(tf_result)
  print('tflite_results:')
  print(tflite_result)
  # np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=0)

'''
for tflite_result in tflite_results:
  print('tflite_results:')
  print(tflite_result)
print('Finished!')
