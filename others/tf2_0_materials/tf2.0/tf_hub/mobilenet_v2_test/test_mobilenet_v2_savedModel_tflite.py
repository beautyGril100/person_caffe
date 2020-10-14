# coding=gbk
import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image

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

def get_data(path_list,idx): 
    img_path = path_list[idx]
    img = cv2.imread(img_path)
    src_img = img.copy()  #used to show
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img,axis=0)
    #uint8_img = img.copy()  #used to tflite uint8
    img = img.astype("float32")
    uint8_img = img.copy()  #used to tflite uint8
    img = (img/255.0-0.5)*2.0
    return img_path,img,src_img, uint8_img


if __name__ == '__main__':
    test_imageNet_flag = True
    if test_imageNet_flag:
        saved_model = "./keras_mobilenet_v2_classification_imageNet_saved_models/1590568434"
    else:
        saved_model = "./keras_mobilenet_v2_classification_flower_saved_models/1590571985"
    
    full_float_tflite_flag = True
    full_integer_tflite_flag = False
    full_float16_tflite_flag = False
    
    if full_float_tflite_flag:
        if test_imageNet_flag:
            full_float_tflite_file = 'tf_keras_mobilenet_v2_float.tflite'
        else:
            full_float_tflite_file = 'tf_keras_mobilenet_v2_classification_flower_float.tflite'
        #converter = tf.lite.TFLiteConverter.from_keras_model('keras_mobilenet_v2_classification_imageNet.h5')
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
        tflite_model = converter.convert()
        
        '''weight quantization'''
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        #tflite_model = converter.convert()
    
        open(full_float_tflite_file, "wb").write(tflite_model)
        
    if full_integer_tflite_flag:
        if test_imageNet_flag:
            full_integer_tflite_file = 'tf_keras_mobilenet_v2_integer.tflite'
        else:
            full_integer_tflite_file = 'tf_keras_mobilenet_v2_classification_flower_integer.tflite'
        # 生成代表性数据集
        def get_representative_dataset_gen():
            path = './representative_dataset/'
            imgSet = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for name in imgSet:
                print(name)
                img = Image.open(name)
                img = np.array(img.resize((224,224)))
                #img = (img/255.0)
                img = np.array([img.astype('float32')])
                yield [img]
        
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
        converter.representative_dataset = get_representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.INT8]
        #converter.inference_input_type = tf.uint8
        #converter.inference_output_type = tf.uint8
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.experimental_enable_mlir_converter = False
        tflite_model = converter.convert()
        open(full_integer_tflite_file, "wb").write(tflite_model)
    
    if full_float16_tflite_flag:
        if test_imageNet_flag:
            full_float16_tflite_file = 'tf_keras_mobilenet_v2_float16.tflite'
        else:
            full_float16_tflite_file = 'tf_keras_mobilenet_v2_classification_flower_float16.tflite'
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        tflite_model = converter.convert()
        open(full_float16_tflite_file, "wb").write(tflite_model)
    
    # load TFLite model and then allocate tensor
    # 加载 TFLite 模型并分配张量（tensor）。
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    all_tensor_details = interpreter.get_tensor_details()
    print('all_tensor_number:', len(all_tensor_details))
    for b_tmp_tensor in all_tensor_details:
        #b_tmp_tensor['dtype'] = np.float32
        print('b_tmp_tensor:',b_tmp_tensor)
    # get input and output tensor
    # 获取输入和输出张量。
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input_details:', input_details)
    print('output_details:',output_details)
    
    label=load_label()
    full_float = True
    show_tflite = True
    dir_path = 'test_images'
    images_path =[dir_path+'/'+n for n in os.listdir(dir_path)]
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
            output_scalefactor = output_details[0]["quantization"][0]
            output_zeroPoint = output_details[0]["quantization"][1]
            print('output_scalefactor:', output_scalefactor, ", output_zeroPoint:", output_zeroPoint)
            if output_scalefactor != 0:
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

