# coding=gbk
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# load MobileNet tf.keras model
# ���� MobileNet tf.keras ģ�͡�
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
# print('model:', model)

# convert model
# ת��ģ�͡�
'''ֱ�ӽ�ģ��ת����liteģ�ͣ���������������'''
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

'''weight quantization'''
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#tflite_model = converter.convert()

'''Full integer quantization'''
# ���ɴ��������ݼ�
def get_representative_dataset_gen():
  path = './representative_dataset/'
  imgSet = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
  for name in imgSet:
    print(name)
    img = Image.open(name)
    img = np.array(img.resize((224,224)))
    #img = (img/255.0)
    img = np.array([img.astype('float32')])
    yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('/workspace/hwzhu/tf2.0/tf_lite_model/tf_keras_MobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
converter.representative_dataset = get_representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_types = [tf.compat.v1.lite.constants.QUANTIZED_UINT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()


'''Float16 quantization'''
'''
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
tflite_model = converter.convert()
'''
# save converted model
open("tf_keras_MobileNetV2.tflite", "wb").write(tflite_model)

# load TFLite model and then allocate tensor
# ���� TFLite ģ�Ͳ�����������tensor����
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# get input and output tensor
# ��ȡ��������������
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input_details:', input_details)
print('output_details:',output_details)
# using random data as input, then test TensorFlow Lite model
# ʹ�����������Ϊ������� TensorFlow Lite ģ�͡�
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# function 'get_tensor()' will return a copy of a tensor
# using 'tensor()' to get a pointer that pointing to a tensor
# ���� `get_tensor()` �᷵��һ�������Ŀ�����
# ʹ�� `tensor()` ��ȡָ��������ָ�롣
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# testing TensorFlow model with random data
# ʹ�����������Ϊ������� TensorFlow ģ�͡�
tf_results = model(tf.constant(input_data))

# compare results
# �ԱȽ����
for tf_result, tflite_result in zip(tf_results, tflite_results):
  print('tf_results:')
  print(tf_result)
  print('tflite_results:')
  print(tflite_result)
  # np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=0)
  
print('Finished!')
