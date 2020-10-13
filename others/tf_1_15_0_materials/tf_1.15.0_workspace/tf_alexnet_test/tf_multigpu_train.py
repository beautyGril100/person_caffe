import tf_alexnet as alexnet
import tensorflow as tf
from tensorflow.contrib import slim
import time
import tf_tool_util
import numpy as np

logger = tf_tool_util.logger

#only support 224 in this alexnet
tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')  #32
tf.app.flags.DEFINE_integer('num_readers', 8, '')  #16
tf.app.flags.DEFINE_float('learning_rate', 0.0005, '')
tf.app.flags.DEFINE_float('keep_prob', 0.5, '')
tf.app.flags.DEFINE_integer('max_steps', 5000, '')  #600000
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './classify/', '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')

import tf_readdata as readdata

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

def tower_loss(images, labels, reuse_variable=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variable):
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            images = readdata.mean_image_subtraction(images)
            outputs, end_points = alexnet.alexnet_v2(images, num_classes=FLAGS.num_classes)
            logger.debug(outputs.get_shape().as_list())
            logger.debug(labels.get_shape().as_list())
            assert outputs.get_shape().as_list()==labels.get_shape().as_list()
    model_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=labels, name='cross-entropy'))
    total_loss = tf.add_n([model_loss]+tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    if reuse_variable is None:
        tf.summary.scalar('cross_entropy', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss

#grad average for multi gpu
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_sum(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var=(grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3], name='input_images')
    gt_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='groundtruth_labels')

    keep_prob = tf.placeholder(tf.float32)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=1000, decay_rate=0.94, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    #split data for multi gpus
    input_image_splits = tf.split(input_images, len(gpus))
    gt_labels_splits = tf.split(gt_labels, len(gpus))

    tower_grads = []
    reuse_variable = None
    for i, gpuid in enumerate(gpus):
        with tf.device('/gpu:%d'%gpuid):
            with tf.name_scope('model_%d'%gpuid) as scope:
                each_image_split = input_image_splits[i]
                each_label_split = gt_labels_splits[i]
                total_loss, model_loss = tower_loss(each_image_split, each_label_split, reuse_variable)

                reuse_variable = True
                grads = optimizer.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    logger.debug('grads:'+str(type(grads)))
    logger.debug(grads)
    apply_gradient_opt = optimizer.apply_gradients(grads, global_step=global_step)


    summary_opt = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    # [variables_averages_op, apply_gradient_opt, batch_norm_updates_op]
    with tf.control_dependencies([variables_averages_op, apply_gradient_opt]):
        train_op = tf.no_op(name='train_op')
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            logger.info('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            logger.info(ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = readdata.get_batch(num_workers=FLAGS.num_readers,
                                            input_size=FLAGS.input_size,
                                            batch_size=FLAGS.batch_size_per_gpu*len(gpus))

        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            start_time = time.time()
            ml, tl, lr, _ = sess.run([model_loss, total_loss, learning_rate, apply_gradient_opt], feed_dict={input_images:data[0], gt_labels:data[1], keep_prob:FLAGS.keep_prob})
            duration = time.time()-start_time

            if np.isnan(tl):
                logger.error('Loss diverged, stop training')
                break

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size_per_gpu*len(gpus)
                examples_per_sec = num_examples_per_step/duration
                logger.info('step {:06d}, model_loss {:.4f}, total_loss {:.4f}, learning_rate {:.6f}, {:.2f} seconds/step, {:.2f} examples/second'.format(step, ml, tl, lr, duration, examples_per_sec))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                tl, summary_str, _ = sess.run([total_loss, summary_opt, apply_gradient_opt], feed_dict={input_images:data[0], gt_labels:data[1],  keep_prob:FLAGS.keep_prob})
                summary_writer.add_summary(summary_str, global_step=step)

if __name__=='__main__':
    tf.app.run()
