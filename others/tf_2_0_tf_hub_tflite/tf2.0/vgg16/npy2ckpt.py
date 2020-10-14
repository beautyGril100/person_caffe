from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels


import argparse
import os

SAVE_DIR = './'

def get_arguments():
    '''
    Parse all the arguments provided from the command line
    
    Returns:
      A list of parsed arguments
    '''
    parser = argparse.ArgumentParser(description='NPY to CKPT converter.')
    parser.add_argument('npy_path', type=str, help='Path to the .npy file, which contains the weights.')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help='Where to save the converted .ckpt file.')
    
    return parser.parse_args()
    
def save(saver, sess, logdir):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    saver.save(sess, checkpoint_path, write_meta_graph=False)
    print('The weights have been converted to {}.'.format(checkpoint_path))

'''
def main():
    # Create the model.
    args = get_arguments()
    
    # Default image.
    # image_batch = tf.constant(0, tf.float32, shape=[1, 321, 321, 3])
    image_batch = tf.placeholder(tf.float32, [1, 224, 224, 3])
    
    # Create network
    net = vgg16.Vgg16()
    var_list = tf.global_variables()
    
    # Set up tf session and initialize variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Loading .npy weights.
        net.load(args.npy_path, sess)
        net.forward(image_batch)
        probability = sess.run(vgg.prob, feed_dict={images: img_ready})
        # Saver for converting the loaded weights into .ckpt
        saver = tf.train.Saver(var_list=var_list, write_version=1)
        save(saver, sess, args.save_dir)
        
if __name__ == '__main__':
    main()
'''


img_path = input('Input the path and image name: ')
img_ready = utils.load_image(img_path)

fig = plt.figure(u"Top-5 predict results")


var_list = tf.global_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    images    = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    
    vgg.forward(images)
    probability = sess.run(vgg.prob, feed_dict={images: img_ready})
    saver = tf.train.Saver(var_list=var_list, write_version=1)
    save(saver, sess, args.save_dir)
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)
    '''
    values = []
    bar_label = []
    for n, i in enumerate(top5):
        print("n:", n)
        print("i:", i)
        values.append(probability[0][i])
        bar_label.append(labels[i])
        print(i, ":", labels[i], "----", utils.percent(probability[0][i]))

    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probabilityit')
    ax.set_title(u'Top-5')
    for a, b in zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
    plt.show()
    '''

