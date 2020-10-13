import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
from tensorflow.python.framework.ops import numpy_text

print('Tensorflow LSTM will be testing........')

cell = LSTMCell(1, state_is_tuple=False, forget_bias=0.0, activation='linear')
cell.build(input_shape=[1, 3])
cell._kernel.assign([[0., 1., 0., 0.],
                     [100., 0., 100., 0.],
                     [0., 0., 0., 100.],
                     [-10., 0., 10., -10.]])
inputs = tf.constant([[3.0, 1.0, 0.0],
                      [4.0, 1.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [1.0, 0.0, 1.0],
                      [3.0, -1.0, 0.0]])
inputArray = array_ops.split(value=inputs, num_or_size_splits=5, axis=0)
print('inputArray: ', inputArray)
states = tf.constant([[0.0, 1.0]])
cStr = 'c values: '
hStr = 'h values: '
for inputi in inputArray:
    _, new_state = cell.call(inputi, states)
    print('************output******************')
    print('new_state: ', new_state)
    c, h = array_ops.split(value=new_state, num_or_size_splits=2, axis=1)
    states = array_ops.concat([c, tf.ones_like(h)], 1)
    print('c: ', c)
    print('h: ', h)
    #cStr = cStr + numpy_text(c) + ' '
    #hStr = hStr + numpy_text(h) + ' '

print('cStr: ', cStr)
print('hStr: ', hStr)
    
