# coding=gbk
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

slim = contrib_slim

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def lstm_test(inputs, use_explicit_padding=False, scope=None):
    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    
    with tf.variable_scope(scope, 'LSTM_Test', [inputs]):
        with slim.arg_scope([slim.conv2d], padding=padding):
            net = inputs
            net = slim.conv2d(inputs=net, num_outputs=96, kernel_size=[3, 3], stride=1, padding=padding,activation_fn=None, scope='conv1')
            net = slim.avg_pool2d(inputs=net, kernel_size=[2, 2], stride=2, padding='VALID', scope='pool1')
            net = tf.nn.relu(features=net, name='relu1')
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=1, padding=padding, activation_fn=None, scope='conv2')
            net = slim.avg_pool2d(inputs=net, kernel_size=[2, 2], stride=2, padding='VALID', scope='pool2')
            net = tf.nn.relu(features=net, name='relu2')
            net = slim.conv2d(inputs=net, num_outputs=384, kernel_size=[3, 3], stride=1, padding=padding, activation_fn=tf.nn.relu, scope='conv3')
            #net = slim.dropout(inputs=net, keep_prob=0.8, scope='drop3')
            net = slim.conv2d(inputs=net, num_outputs=384, kernel_size=[3, 3], stride=1, padding=padding, activation_fn=tf.nn.relu, scope='conv4')
            #net = slim.dropout(inputs=net, keep_prob=0.8, scope='drop4')
            print('net-0: ', net)
            net = slim.avg_pool2d(inputs=net, kernel_size=[2, 7], stride=2, padding='VALID', scope='pool5')
            print('net: ', net)
            net = tf.reshape(tensor=net, shape=[1, 384, 3], name='reshape')
            net = tf.transpose(a=net, perm=[2, 0, 1], name='permuted_data')
            print('net.shape:',net.shape)
            n_hidden_units = 128
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=False, name='lstm1')
            _init_state = tf.constant(value=0.0, dtype=tf.float32, shape=[1, n_hidden_units * 2], name='lstm_initial_state')
            inputArray = tf.split(value=net, num_or_size_splits=net.shape[0], axis=0, name='split_inputs')
            print('inputArray: ', inputArray)
            tmpOutputTensors = []
            for inputi in inputArray:
                inputi = tf.squeeze(input=inputi, axis=1, name='squeeze_input')
                new_h, new_state = lstm_cell.__call__(inputi, _init_state)
                _init_state = new_state
                tmpOutputTensors.append(new_h)
            
            print('tmpOutputTensors: ', tmpOutputTensors)
            outputs = tf.concat(values=tmpOutputTensors, axis=1, name='concat_outputs')
            print('outputs: ', outputs)
            #net = tf.expand_dims(input=outputs, axis=0, name='expand_outputs_dim')
            net = outputs
            print('net-2: ', net)
            #net = tf.transpose(a=net, perm=[1, 2, 0], name='permuted_lstm')
            net = slim.fully_connected(inputs=net, num_outputs=256, activation_fn=tf.nn.relu, scope='fc6')
            #net = slim.dropout(inputs=net, keep_prob=0.8, scope='drop6')
            net = slim.fully_connected(inputs=net, num_outputs=128, activation_fn=tf.nn.relu, scope='fc7')
            #net = slim.dropout(inputs=net, keep_prob=0.8, scope='drop7')
            net = slim.fully_connected(inputs=net, num_outputs=10, activation_fn=None, scope='fc8')
            prob = tf.nn.softmax(logits=net, axis=1, name='prob')
            return prob, net
            
          
n_classes = 10
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])           
            
prob, net = lstm_test(x)
print('prob: ', prob)
print('net: ', net)

lr = 0.001
train_iters = 10000
batch_size = 1  #128
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.initialize_all_variables()

# create a saver
saver = tf.train.Saver(tf.global_variables())


with tf.Session() as sess:
  sess.run(init)
  step = 0
  while step * batch_size < train_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #print('batch_xs.shape: ', batch_xs.shape)
    batch_xs = batch_xs.reshape([batch_size, 28, 28, 1])
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
  img = cv2.resize(img,(28, 28))
  img = np.expand_dims(img,axis=0)
  img = img.astype("float32")
  print('b-img.shape: ', img.shape)
  img = img.transpose(1,2,0)
  print('a-img.shape: ', img.shape)
  img = np.expand_dims(img, axis=0)
  print('aa-img.shape: ', img.shape)
  inputs=tf.placeholder(dtype=tf.float32,shape=(1,28,28,1))
  prob, net = lstm_test(inputs)
  sess.run(tf.initialize_all_variables())
  score = sess.run(prob,feed_dict={inputs:img})
  print('score: ', score.shape)

  graph_def_removed_training_nodes = tf.graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
  
  output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
      sess=sess,
      input_graph_def=graph_def_removed_training_nodes, # input_graph_def=input_graph_def,  # 等于:sess.graph_def
      output_node_names=['LSTM_Test/Softmax'])  # 如果有多个输出节点，以逗号隔开

  with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
    f.write(output_graph_def.SerializeToString())  # 序列化输出
  # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
'''

            