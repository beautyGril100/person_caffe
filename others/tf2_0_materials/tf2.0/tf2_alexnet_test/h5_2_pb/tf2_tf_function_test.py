# coding=gbk
# ����ʵսtf.function auto_graph
# ��ͨ����ת��tensorflow�������Ż�������ٶ�
# ����ǩ����ͼ�ṹ
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
#import sklearn
#import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,tf,keras:
    print(module.__name__,module.__version__)
    
'''
2. tf.function
˵����
  ��ͨPython�﷨д�ĺ�����ת����tensorflowͼ����������ٶ�
  auto_graph ��tf.function�������Ļ���
'''

# 2.1 ��ͨ����
# tf.functin and autograph
def scaled_elu(z,scale=1.0,alpha=1.0):
    # z >= 0 ? scale *z :scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z,0.0)
    return scale * tf.where(is_positive,z,alpha * tf.nn.elu(z))

print('*********python function***********')
print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3.,-2.5])))

# 2.2 ת��ͼ�ṹ
# method 1
 # tf.function ת��ͼ�ṹ
scaled_elu_tf = tf.function(scaled_elu)
print('*********tf.function***************')
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3.,-.5])))

# python_function ��tensorflow����ת��python����
print('scaled_elu_tf.python_function is scaled_elu')

start_t = time.time()
scaled_elu(tf.random.normal((1000,1000)))
print('python func elapse: ', (time.time() - start_t) * 1000, 'ms')
start_t = time.time()
scaled_elu_tf(tf.random.normal((1000,1000)))
print('tf func elapse: ', (time.time() - start_t) * 1000, 'ms')

# method 2
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total
print(converge_to_2(20))

'''
# 2.2.2 tensorflow����
# չʾtensorflow����
def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display,Markdown
    display(Markdown('```python\n{}\n```'.format(code)))

display_tf_code(scaled_elu)
display_tf_code(converge_to_2)
'''

# 2.2.3 tf.Variable
# ע�� ���������ں�����
# tf.Variable
var = tf.Variable(0.)
@tf.function
def add_21():
    return var.assign_add(21) # +=
print(add_21())

@tf.function
def cube(z):
    return tf.pow(z,3)
print(cube(tf.constant([1.,2.,3.])))
print(cube(tf.constant([1 ,2 ,3])))

# �޶�����
# �����޶���������
@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')])
def cube(z):
    return tf.pow(z,3)
try:
    print(cube(tf.constant([1.,2.,3.])))
except ValueError as ex:
    print(ex)
print(cube(tf.constant([1 ,2 ,3])))

'''
input_signature =>> save_model
1. @tf.function py func -> tf graph
2. get_concrete_function -> add input signature -> SavedModel
'''
# @tf.function py func -> tf graph
# get_concrete_function -> add input signature -> SavedModel
cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None],tf.int32))
print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5],tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1,2,3])))

cube_func_int32.graph
# ����
cube_func_int32.graph.get_operations()

pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

print(list(pow_op.inputs))
print(list(pow_op.outputs))

cube_func_int32.graph.get_operation_by_name('x')
cube_func_int32.graph.get_tensor_by_name('x:0')
cube_func_int32.graph.as_graph_def()



  




