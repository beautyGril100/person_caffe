# coding=gbk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn.python.ops import lstm_ops
import cv2
import numpy as np

state_is_tuple_flag = False
custom_loop_flag = True
# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
train_iters = 10000
batch_size = 1  #128
display_step = 10

n_inputs = 196  #28
n_steps = 4  #28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weight
weights = {
  # (28, 128)
  'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
  # (128, 10)
  'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
} 

biases = {
  # (128, )
  'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
  # (10, )
  'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
'''
def RNN(X, weights, biases):
  # 形状变换成lstm可以训练的维度
  X = tf.reshape(X, [-1, n_inputs])  # (128 * 28, 28)
  X_in = tf.matmul(X, weights['in']) + biases['in']  # (128 * 28, 128)
  X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # (128, 28, 128)
  
  # cell
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
  # lstm cell is divided into two parts(c_state, m_state)
  _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
  
  outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
  
  
  # outputs
  # results = tf.matmul(states[1], weights['out']) + biases['out']
  # or
  outputs = tf.transpose(outputs, [1, 0, 2])
  results = tf.matmul(outputs[-1], weights['out']) + biases['out']
  
  return results
  
pred = RNN(x, weights, biases)
'''

# 形状变换成lstm可以训练的维度
X = tf.reshape(x, [-1, n_inputs])  # (128 * 28, 28)
#X_in = tf.matmul(X, weights['in']) + biases['in']  # (128 * 28, 128)
X_in = tf.matmul(X, weights['in'])
X_in = tf.nn.bias_add(X_in, biases['in'])
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # (128, 28, 128)
print('X_in.shape: ', X_in.shape)
# cell
if state_is_tuple_flag:
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
else:
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=False)
# lstm cell is divided into two parts(c_state, m_state)
if state_is_tuple_flag:
  _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
else:
  _init_state = tf.constant(value=0.0, dtype=tf.float32, shape=[1, n_hidden_units * 2])
  
print('_init_state: ', _init_state)

#
if custom_loop_flag: 
  inputArray = tf.split(value=X_in, num_or_size_splits=n_steps, axis=1)
  print('inputArray: ', inputArray)
  tmpOutputTensors = []
  for inputi in inputArray:
    #print('_init_state: ', _init_state)
    #print('b-inputi: ', inputi)
    inputi = tf.squeeze(inputi, axis=1)
    #print('a-inputi: ', inputi)
    new_h, new_state = lstm_cell.__call__(inputi, _init_state)
    
    #c, h = tf.split(value=new_state, num_or_size_splits=2, axis=0)
    #new_state = tf.concat([c, h], 0)
      
      
    _init_state = new_state
    #print('new_h: ', new_h)
    tmpOutputTensors.append(new_h)
  
  print('tmpOutputTensors: ', tmpOutputTensors)
  outputs = tf.concat(values=tmpOutputTensors, axis=0)
  outputs = tf.expand_dims(input=outputs, axis=0)
else:
  outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
  
print('outputs.shape: ', outputs.shape)

# outputs
# results = tf.matmul(states[1], weights['out']) + biases['out']
# or
outputs = tf.transpose(outputs, [1, 0, 2])
#pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
pred = tf.matmul(outputs[-1], weights['out'])
pred = tf.nn.bias_add(pred, biases['out'],name='Output_hwzhu')
print('pred.name: ', pred.name)
print('pred: ', pred)
prob = tf.nn.softmax(pred, axis=1, name='prob_hwzhu')
print('prob: ', prob)
  
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''
init = tf.initialize_all_variables()

# create a saver
saver = tf.train.Saver(tf.global_variables())


with tf.Session() as sess:
  sess.run(init)
  step = 0
  while step * batch_size < train_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    sess.run(train_op, feed_dict={x:batch_xs, y:batch_ys})
    step += 1
    if step % 1000 == 0:
      print('accuracy: ', sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
    if step % 5000 == 0:
      saver.save(sess, 'ckpt_model/'+'lstm_model', global_step=step)

'''
input_checkpoint = tf.train.latest_checkpoint('ckpt_model/')
output_graph = 'lstm_test.pb'
print('input_checkpoint: ', input_checkpoint)
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#graph = tf.compat.v1.get_default_graph()  # 获得默认的图
#input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

with tf.Session() as sess:
  saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
  
  for op in sess.graph.get_operations():
    print('os.type:', op.type, ', name:', op.name)
  
  img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img,(196, 4))
  img = np.expand_dims(img,axis=0)
  img = img.astype("float32")
  print('img.shape: ', img.shape)
  #inputs=tf.placeholder(dtype=tf.float32,shape=(1,4,196))
  #score = sess.run(prob,feed_dict={inputs:img})
  print('score: ', score)

  graph_def_removed_training_nodes = tf.graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
  
  output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
      sess=sess,
      input_graph_def=graph_def_removed_training_nodes, # input_graph_def=input_graph_def,  # 等于:sess.graph_def
      output_node_names=['prob_hwzhu_1'])  # 如果有多个输出节点，以逗号隔开

  with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
    f.write(output_graph_def.SerializeToString())  # 序列化输出
  # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

  