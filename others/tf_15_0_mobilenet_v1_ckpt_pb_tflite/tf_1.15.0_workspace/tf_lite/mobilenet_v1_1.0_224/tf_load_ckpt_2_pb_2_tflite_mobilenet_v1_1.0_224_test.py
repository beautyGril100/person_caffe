# coding=gbk
import tensorflow as tf
from tensorflow.lite.python import lite_constants
from mobilenet_v1 import mobilenet_v1,mobilenet_v1_arg_scope
import cv2
import os
import numpy as np
slim = tf.contrib.slim
CKPT = 'mobilenet_v1_1.0_224.ckpt' 
dir_path = 'test_images'

#sort tools for computing topK
def naive_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(-matrix, axis=axis)  #sort from big to small
    print('full_sort:',full_sort)
    topK_indices = full_sort.take(np.arange(K), axis=axis)
    print('topK_indices:', topK_indices)
    sorted_data = matrix[full_sort]
    print('sorted_data:',sorted_data)
    topK_values = sorted_data.take(np.arange(K), axis=axis)
    print('topK_values:', topK_values)
    return topK_indices, topK_values

'''
# Example
>>> dists = np.random.permutation(np.arange(30)).reshape(6, 5)
array([[17, 28,  1, 24, 23,  8],
       [ 9, 21,  3, 22,  4,  5],
       [19, 12, 26, 11, 13, 27],
       [10, 15, 18, 14,  7, 16],
       [ 0, 25, 29,  2,  6, 20]])
>>> naive_arg_topK(dists, 2, axis=0) # col
array([[4, 2, 0, 4, 1, 1],
       [1, 3, 1, 2, 4, 0]])
>>> naive_arg_topK(dists, 2, axis=1) # row
array([[2, 5],
       [2, 4],
       [3, 1],
       [4, 0],
       [0, 3]])
'''
def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

'''
# Example
>>> dists = np.random.permutation(np.arange(30)).reshape(6, 5)
array([[17, 28,  1, 24, 23,  8],
       [ 9, 21,  3, 22,  4,  5],
       [19, 12, 26, 11, 13, 27],
       [10, 15, 18, 14,  7, 16],
       [ 0, 25, 29,  2,  6, 20]])
>>> partition_arg_topK(dists, 2, axis=0)  # col
array([[4, 2, 0, 4, 1, 1],
       [1, 3, 1, 2, 4, 0]])
>>> partition_arg_topK(dists, 2, axis=1)  # row
array([[2, 5],
       [2, 4],
       [3, 1],
       [4, 0],
       [0, 3]])
'''

def build_model(inputs):   
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        logits, end_points = mobilenet_v1(inputs, is_training=False, depth_multiplier=1.0, num_classes=1001)
    scores = end_points['Predictions']
    print("scores:",scores)
    #取概率最大的5个类别及其对应概率
    output = tf.nn.top_k(scores, k=5, sorted=True)
    #indices为类别索引，values为概率值
    return output.indices,output.values

def load_model(sess):
    loader = tf.train.Saver()
    loader.restore(sess,CKPT)
 
def get_data(path_list,idx): 
    img_path = path_list[idx]
    img = cv2.imread(img_path)
    src_img = img.copy()  #used to show
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img,axis=0)
    uint8_img = img.copy()  #used to tflite uint8
    img = img.astype("float32")
    img = (img/255.0-0.5)*2.0
    return img_path,img,src_img, uint8_img
def load_label():
    label=[]
    with open('labels_1001_mobilenet_quant_v1_224.txt','r',encoding='utf-8') as r:
        lines = r.readlines()
        #print('lines:', lines)
        for l in lines:
            try:
                l = l.strip()
                arr = l.split(',')
                #print('arr:',arr)
                label.append(arr[0])
            except:
                continue
    print("label size:",len(label))
    return label

if __name__ == '__main__':
    inputs=tf.placeholder(dtype=tf.float32,shape=(1,224,224,3))
    classes_tf,scores_tf = build_model(inputs) 
    images_path =[dir_path+'/'+n for n in os.listdir(dir_path)]
    
    label=load_label()
    with tf.Session() as sess:
        load_model(sess)
        for i in range(len(images_path)):
            path,img,src_img,_ = get_data(images_path,i)
            classes,scores = sess.run([classes_tf,scores_tf],feed_dict={inputs:img})
            print('\n**********Infrence ckpt 识别',path,'结果如下：******************')
            for j in range(5):#top 5
                idx = classes[0][j]
                score=scores[0][j]
                print('\tNo.',j,'类别:',label[idx],'概率:',score) 
            #show
            label_text = "It's a %s, confidence is %f" % (label[classes[0][0]], scores[0][0])
            print('label_text:', label_text)
            show_ckpt = False
            if show_ckpt:
                cv2.putText(src_img, label_text, (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
                window_name = 'ckpt_resut'
                #cv2.namedWindow(window_name,0)
                #cv2.startWindowThread()
                cv2.imshow(window_name,src_img)
                #cv.imwrite("pb_result.jpg", result)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)        
        #ckpt converted to pb
        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        pb_model_path="custom_mobilenet_v1_1.0_224_frozen.pb"
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["MobilenetV1/Predictions/Softmax"])  #
        with tf.gfile.FastGFile(pb_model_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
            
    test_pb_model=True
    test_tflite_model=True
    
    if test_pb_model:
        input_image=tf.placeholder(tf.float32,(1,224,224,3))
 
        with open(pb_model_path,"rb") as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            out_result=tf.import_graph_def(graph_def,input_map={"Placeholder:0":input_image},return_elements=["MobilenetV1/Predictions/Softmax:0"])  #
            #取概率最大的5个类别及其对应概率
            output = tf.nn.top_k(out_result[0], k=5, sorted=True)
        sess=tf.Session()
        for i in range(len(images_path)):
            path,img, src_img,_ = get_data(images_path,i)
            result=sess.run(output,feed_dict={input_image:img})
            print("output:", result)
            #indices为类别索引，values为概率值
            print('\n**********Infrence pb 识别',path,'结果如下：******************')
            for j in range(5):#top 5
                idx = result.indices[0][j]
                score=result.values[0][j]
                print('\tNo.',j,'类别:',label[idx],'概率:',score)
        
            #show
            label_text = "It's a %s, confidence is %f" % (label[result.indices[0][0]], result.values[0][0])
            print('label_text:', label_text)
            show_pb = False
            if show_pb:
                cv2.putText(src_img, label_text, (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
                window_name = 'pb_resut'
                #cv2.namedWindow(window_name,0)
                #cv2.startWindowThread()
                cv2.imshow(window_name,src_img)
                #cv.imwrite("pb_result.jpg", result)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
    if test_tflite_model:
        full_float = True
        inputs = ['Placeholder']
        outputs = ['MobilenetV1/Predictions/Softmax']
        if full_float:# full float
            tflite_model_path = "custom_mobilenet_v1_1.0_224_frozen_float.tflite"
            converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_model_path, inputs, outputs)
            converter.inference_type = lite_constants.FLOAT
            tflite_model = converter.convert()
            open(tflite_model_path, "wb").write(tflite_model)
        else:# full uint8
            
            tflite_model_path = "custom_mobilenet_v1_1.0_224_frozen_integer.tflite"
            converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_model_path, inputs, outputs)
            converter.inference_type = lite_constants.QUANTIZED_UINT8
            converter.input_type=lite_constants.QUANTIZED_UINT8
            converter.quantized_input_stats = {'Placeholder':(127.5, 127.5) }#(mean, stddev)
            converter.default_ranges_stats = (-5.9, 100)
            #converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]  #is equal to 'converter.post_training_quantize = True'
            tflite_model = converter.convert()
            open(tflite_model_path, "wb").write(tflite_model) 
            
            

        interpreter=tf.lite.Interpreter(tflite_model_path)  #mobilenet_v1_1.0_224_quant.tflite
        interpreter.allocate_tensors()
 
        input_details=interpreter.get_input_details()
        input_scalefactor = input_details[0]["quantization"][0]
        input_zeroPoint = input_details[0]["quantization"][1]
        print('input_details:', input_details)
        print('input_scalefactor:', input_scalefactor, ', input_zeroPoint:', input_zeroPoint)
        output_details=interpreter.get_output_details()
        output_scalefactor = output_details[0]["quantization"][0]
        output_zeroPoint = output_details[0]["quantization"][1]
        print('output_details:', output_details)
        print('output_scalefactor:', output_scalefactor, ', output_zeroPoint:', output_zeroPoint)
 
        for i in range(len(images_path)):
            path,img, src_img,uint8_img = get_data(images_path,i)
            
            if full_float:# full float
                interpreter.set_tensor(input_details[0]["index"],img)
            else:  #full uint8
                interpreter.set_tensor(input_details[0]["index"],uint8_img)
                #print('uint8_img:',uint8_img)
                
            interpreter.invoke()
            output_data=interpreter.get_tensor(output_details[0]["index"])
            print('tflite_output_data:', output_data)
            print('tflite_output_data[0]:', output_data[0])
            result=output_data[0]
            
            if not full_float:
                print("uint8_result:", result)
                result = (result - output_zeroPoint) * output_scalefactor
            print("result:", result)
            
            #dump output data
            if len(result) > 0:
                savefile_folder = 'test_mobilenet_v1_1.0_224.txt'
                savefile = open(savefile_folder, 'w')
                for data_idx in range(len(result)):
                    savefile.write('%.06f'%(result[data_idx])+',')
                savefile.write('\n')
                savefile.close()
                print(path + '    OVER!!!')
            
            indices, values = naive_arg_topK(result, 5)
            #indices为类别索引，values为概率值
            print('\n**********Infrence tflite 识别',path,'结果如下：******************')
            for j in range(5):#top 5
                idx = indices[j]
                score=values[j]
                print('\tNo.',j,'类别:',label[idx],'概率:',score)    
            
            #show
            label_text = "It's a %s, confidence is %f" % (label[indices[0]], values[0])
            print('label_text:', label_text)
            show_tflite = False
            if show_tflite:
                cv2.putText(src_img, label_text, (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
                window_name = 'tflite_resut'
                #cv2.namedWindow(window_name,0)
                #cv2.startWindowThread()
                cv2.imshow(window_name,src_img)
                #cv.imwrite("pb_result.jpg", result)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
    
