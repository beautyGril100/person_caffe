import tensorflow as tf
import numpy as np
import cv2

class Predictor(object):
    def __init__(self, pbfile, config: tf.ConfigProto):
        '''
        pbfile: pb file
        config: like tf.ConfigProto(allow_soft_placement=True)
        '''
        self.sess = tf.Session(config=config)
        with open(pbfile, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
   
        self.sess.run(tf.global_variables_initializer())
        self.img=self.sess.graph.get_tensor_by_name('input_images:0')
        self.probs=self.sess.graph.get_tensor_by_name('output/softmax:0')

    def __resize_image(self, image, short_edge_length = 224, max_length = 224):
        '''
        resize image
        return: resized image
        '''

        # apply resize operate
        ret = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]

        return ret


    def predict(self, image, short_edge_length = 224, max_length = 224):
        '''
        image: input image, read from opencv(ndarray)
        '''
        orig_shape = image.shape[:2]
        resized_img = self.__resize_image(image, short_edge_length, max_length)
        probs =self.sess.run([self.probs], feed_dict={self.img: [resized_img]})
        return probs[0]

if __name__=='__main__':
    pbfile = './alexnet_test.pb'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    predictor = Predictor(pbfile, config)
    img = cv2.imread('./test/1.jpg')
    probs = predictor.predict(img)
    print(probs)