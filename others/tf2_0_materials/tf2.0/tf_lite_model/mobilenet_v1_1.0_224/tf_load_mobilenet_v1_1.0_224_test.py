# coding=gbk
import tensorflow as tf
from mobilenet_v1 import mobilenet_v1,mobilenet_v1_arg_scope
import cv2
import os
import numpy as np
slim = tf.contrib.slim
CKPT = 'mobilenet_v1_1.0_224.ckpt' 
dir_path = 'test_images'

def build_model(inputs):   
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        logits, end_points = mobilenet_v1(inputs, is_training=False, depth_multiplier=1.0, num_classes=1001)
    scores = end_points['Predictions']
    print(scores)
    #取概率最大的5个类别及其对应概率
    output = tf.nn.top_k(scores, k=3, sorted=True)
    #indices为类别索引，values为概率值
    return output.indices,output.values

def load_model(sess):
    loader = tf.train.Saver()
    loader.restore(sess,CKPT)
 
def get_data(path_list,idx): 
    img_path = path_list[idx]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img,axis=0)
    img = (img/255.0-0.5)*2.0
    return img_path,img
def load_label():
    label=[]
    with open('labels_1001_mobilenet_quant_v1_224.txt','r',encoding='utf-8') as r:
        lines = r.readlines()
        for l in lines:
            l = l.strip()
            arr = l.split(',')
            label.append(arr[1])
    return label

inputs=tf.placeholder(dtype=tf.float32,shape=(1,224,224,3))
classes_tf,scores_tf = build_model(inputs) 
images_path =[dir_path+'/'+n for n in os.listdir(dir_path)]
label=load_label()
with tf.Session() as sess:
    load_model(sess)
    for i in range(len(images_path)):
        path,img = get_data(images_path,i)
        classes,scores = sess.run([classes_tf,scores_tf],feed_dict={inputs:img})
        print('\n识别',path,'结果如下：')
        for j in range(3):#top 3
            idx = classes[0][j]
            score=scores[0][j]
            print('\tNo.',j,'类别:',label[idx],'概率:',score) 
    
