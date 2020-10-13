# coding=gbk
import tensorflow as tf
from tensorflow.contrib import slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc

def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
    """AlexNet version 2.
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224. To use in fully
          convolutional mode, set spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[11, 11], stride=4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=192, kernel_size=[5, 5], scope='conv2')
            net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(inputs=net, num_outputs=384, kernel_size=[3, 3], scope='conv3')
            net = slim.conv2d(inputs=net, num_outputs=384, kernel_size=[3, 3], scope='conv4')
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], scope='conv5')
            net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope='pool5')
            
            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(inputs=net, num_outputs=4096, kernel_size=[5, 5], padding='VALID', scope='fc6')
                net = slim.dropout(inputs=net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = slim.conv2d(inputs=net, num_outputs=4096, kernel_size=[1, 1], scope='fc7')
                net = slim.dropout(inputs=net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=[1, 1], activation_fn=None, normalizer_fn=None, biases_initializer=tf.zeros_initializer(), scope='fc8')
                
            # Convert end_points_collection into a end_point dict.
            print('end_points_collection: ', end_points_collection)
            tmp_collection = tf.get_collection(end_points_collection)
            print('tmp_collection: ', tmp_collection)
            print('tensor_aliases-tensor as follow: ')
            for tmp_tensor in tmp_collection:
                if hasattr(tmp_tensor, 'aliases'):
                    print('has aliases...')
                    aliases = tmp_tensor.aliases
                else:
                    if tmp_tensor.name[-2:] == ':0':
                        # Use op.name for tensor ending in :0
                        print('use tensor.op.name...')
                        aliases = [tmp_tensor.op.name]
                    else:
                        print('use tensor name...')
                        aliases = [tmp_tensor.name]
                        
                print(aliases, tmp_tensor)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            
            print('net: ', net)
            print('end_points: ', end_points)
            if spatial_squeeze:
                net = tf.squeeze(input=net, axis=[1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            
            return net, end_points

alexnet_v2.default_image_size = 224

if __name__ == '__main__':
    random_test_flag = False
    if random_test_flag:
        batch_size = 1
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = alexnet_v2(inputs)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            output = sess.run(logits)
            #print('output: ', output)
    else:
        with slim.arg_scope(alexnet_v2_arg_scope()):
            inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
            alexnet_v2(inputs=inputs)
