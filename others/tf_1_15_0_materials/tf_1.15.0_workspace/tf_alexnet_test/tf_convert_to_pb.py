import tensorflow as tf
import tf_alexnet as alexnet
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util
from tf_tool_util import logger
import tf_readdata as readdata

tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_string('checkpoint_path', './classify/', '')

FLAGS = tf.app.flags.FLAGS

def freeze_graph(input_checkpoint: str, output_node: str = "output/softmax", output_pb: str = './alexnet_test.pb'):
    '''
    :param input_checkpoint:
    :param output_node: output node name
    :param output_pb: save pb path
    :return:
    '''
    # output node name
    output_node_names = output_node
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name = 'input_images')

        input_images = readdata.mean_image_subtraction(input_images)  #no subtract mean value
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            outputs, _ = alexnet.alexnet_v2(input_images, num_classes=FLAGS.num_classes, is_training=False)
            logger.debug(outputs.get_shape().as_list())
            squeeze_outputs = tf.squeeze(outputs)
            logger.debug(tf.shape(squeeze_outputs))
            with tf.variable_scope( 'output'):
                probs = tf.nn.softmax(squeeze_outputs, name='softmax')
            logger.debug(tf.shape(probs))

            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print('input_checkpoint: ', input_checkpoint)
                ckpt_state = tf.train.get_checkpoint_state(input_checkpoint)
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                graph = tf.get_default_graph()
                input_graph_def = graph.as_graph_def()
                logger.debug(input_graph_def)
                output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                                                input_graph_def=input_graph_def,
                                                                                output_node_names=output_node_names.split(","))
    
                with tf.gfile.GFile(output_pb, "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                logger.debug("convert complete")

if __name__ == '__main__':
    input_checkpoint = 'classify/'
    freeze_graph(input_checkpoint)