# coding=gbk
import os
import cv2
import time
import numpy as np
import collections
import tensorflow as tf
from tensorflow.contrib.lite.python import lite_constants

import sys
sys.path.append("/workspace/hwzhu/tf_1.12.0_workspace/models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

count = 0

def get_data(path_list,idx): 
    img_path = path_list[idx]
    img = cv2.imread(img_path)
    #src_img = img.copy()  #used to show
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(300,300))
    src_img = img.copy()  #used to show
    img = np.expand_dims(img,axis=0)
    uint8_img = img.copy()  #used to tflite uint8
    img = img.astype("float32")
    img = (img/255.0-0.5)*2.0
    return img_path,img,src_img, uint8_img

def get_ssd_results(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    max_boxes_number=20,
    min_score_thresh=.5):
    """Overlay labeled boxes on an image with formatted scores and label names.
  
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.
  
    Args:
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_number: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    print('min_score_thresh:', min_score_thresh)
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    class_list = []
    #box_to_index = collections.defaultdict(tuple)
    boxes_list = []
    scores_list = []
    if not max_boxes_number:
        max_boxes_number = boxes.shape[0]
        print('max_boxes_number:', max_boxes_number)
    for i in range(min(max_boxes_number, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            print('boxes[%d]:' % i, boxes[i])
            box = tuple(boxes[i].tolist())
            print('box:', box)
            display_str = ''
            print('classes[%d]:' % i, classes[i])
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'  #unknown class
                continue
            display_str = str(class_name)
            #display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            class_list.append(classes[i])  #display_str
            #box_to_index[i] = box
            boxes_list.append(box)
            scores_list.append(scores[i])
  
    # Draw all boxes onto image.
    #print('box_to_index:', box_to_index)
    print('boxes_list:', boxes_list)
    print('boxes as follows:')
    print('image.shape:', image.shape)
    file_name= 'test_tf_faster_rcnn_resnet50_coco_300'+'.txt'
    global count
    count = count + 1
    
    with open(file_name,'a') as ff:
        #for index, box in box_to_index.items():
        for index, box in enumerate(boxes_list):
            ymin, xmin, ymax, xmax = box
            im_height,im_width,im_channel = image.shape
            if use_normalized_coordinates:
                left, top, right, bottom = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
            else:
                left, top, right, bottom = (xmin, ymin, xmax, ymax)
            
            if(right < 0 or bottom < 0):
                continue
            if(left < 0):
                left = 0
            if(top < 0):
                top = 0
            
            ff.write('%d,%d,%f,%d,%d,%d,%d'%(count, class_list[index],scores_list[index],int(left), int(top), int(right), int(bottom)))
            ff.write('\n')     
            print(index,":", (int(left), int(top), int(right), int(bottom)), 'class:', class_list[index], 'score:', scores_list[index])  



def FetchOutput(sess):
    outputs = []
    for op in sess.graph.get_operations():
        if (op.type == "Softmax"):
            outputs.append(op.name.replace('import/', ''))
    if (len(outputs) == 0): 
        print("No outputs spotted.")
    else:
        print("Found ", len(outputs), " possible output(s): ")
        for i in range(len(outputs)):
            print(outputs[i])
    return outputs
    
def freeze_graph(input_checkpoint, output_graph):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    #graph = tf.compat.v1.get_default_graph()  # 获得默认的图
    #input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        
        for op in sess.graph.get_operations():
          print('os.type:', op.type, ', name:', op.name)

        graph_def_removed_training_nodes = tf.graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
        output_node_names = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']  #FetchOutput(sess)
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=graph_def_removed_training_nodes, # input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=['add'])  # 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        
        
if __name__ == '__main__':
    #modelpath="./model.ckpt"
    #freeze_graph(modelpath,"ckpt2frozenPb_test.pb")
    #print("From ckpt Convert to pb finish!")
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    #MODEL_NAME = './'
    #PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_inference_graph.pb')
    #print('PATH_TO_CKPT:', PATH_TO_CKPT)
    # List of the strings that is used to add correct label for each box.
    #PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'mscoco_label_map.pbtxt')
    #print('PATH_TO_LABELS:', PATH_TO_LABELS)
    
    NUM_CLASSES = 90
    
    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #print('label_map:',label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
                                                                
    #print('categories:', categories)
    category_index = label_map_util.create_category_index(categories)
    #print('category_index:',category_index)
    
    dir_path = '../test_image_data'
    images_path =[dir_path+'/'+n for n in os.listdir(dir_path)]
    
    test_pb_model = True
    test_tflite_model = False
    if test_pb_model:
        #Open video file
        #cap = cv2.VideoCapture("video1.mp4") 
        #cap = cv2.VideoCapture(0)
        #Load a frozen TF model 
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        with detection_graph.as_default():
             with tf.Session(graph=detection_graph) as sess:
                   #while (cap.isOpened()):
                      #for op in sess.graph.get_operations():
                          #print('os.type:', op.type, ', name:', op.name, 'inputs:',op.inputs, 'outputs:', op.outputs)
                      for i in range(len(images_path)):
                          path,img, src_img,uint8_img = get_data(images_path,i)
                          #ret, frame = cap.read()
                          start = time.time()
                          image_np = src_img
                          image_np_expanded = uint8_img  #np.expand_dims(image_np, axis=0)
                          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                          print('image_tensor:', image_tensor)
                          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                          scores = detection_graph.get_tensor_by_name('detection_scores:0')
                          classes = detection_graph.get_tensor_by_name('detection_classes:0')
                          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                          # Actual detection.
                          (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                            
                          #print('boxes:',boxes)
                          #print('boxes_squeeze:',np.squeeze(boxes))
                          #print('scores:',scores)
                          #print('scores_squeeze:',np.squeeze(scores))
                          #print('classes:',classes)
                          #print('classes_squeeze:',np.squeeze(classes).astype(np.int32))
                          #print('num_detections:',num_detections)
                          score_threshold = 0.5
                          
                          #post process
                          get_ssd_results(
                          src_img,
                          np.squeeze(boxes),
                          np.squeeze(classes).astype(np.int32),
                          np.squeeze(scores),
                          category_index,
                          use_normalized_coordinates=True,
                          max_boxes_number=None,
                          min_score_thresh=score_threshold)
                          
                          print(path + ' Over!!!')
                          
                          vis_util.visualize_boxes_and_labels_on_image_array(
                              image_np,
                              np.squeeze(boxes),
                              np.squeeze(classes).astype(np.int32),
                              np.squeeze(scores),
                              category_index,
                              use_normalized_coordinates=True,
                              line_thickness=2,
                              min_score_thresh=score_threshold)
                          end = time.time()
                          print('frame:', 1.0 / (end - start))
                          
                          cv2.imshow("capture", image_np)
                          cv2.waitKey(0)
         
        #cap.release()
        cv2.destroyAllWindows()
    if test_tflite_model:
        full_float = True
        
        inputs = ['image_tensor']
        outputs = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']
        if full_float:# full float
            tflite_model_path = "custom_faster_rcnn_resnet50_coco_frozen_float.tflite"
            converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(PATH_TO_CKPT, inputs, outputs, input_shapes={'image_tensor':[1, 300, 300, 3]})
            converter.inference_type = lite_constants.FLOAT
            tflite_model = converter.convert()
            open(tflite_model_path, "wb").write(tflite_model)
        else:# full uint8
            
            tflite_model_path = "custom_faster_rcnn_resnet50_coco_frozen_integer.tflite"
            converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(PATH_TO_CKPT, inputs, outputs, input_shapes={'image_tensor':[1, 300, 300, 3]})
            converter.inference_type = lite_constants.QUANTIZED_UINT8
            converter.input_type=lite_constants.QUANTIZED_UINT8
            converter.quantized_input_stats = {'Placeholder':(127.5, 127.5) }#(mean, stddev)
            converter.default_ranges_stats = (-5.9, 100)
            #converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]  #is equal to 'converter.post_training_quantize = True'
            tflite_model = converter.convert()
            open(tflite_model_path, "wb").write(tflite_model) 
        
        #tflite_model_path = '../coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'   

        interpreter=tf.contrib.lite.Interpreter(tflite_model_path)  #mobilenet_v1_1.0_224_quant.tflite
        interpreter.allocate_tensors()
 
        input_details=interpreter.get_input_details()
        input_scalefactor = input_details[0]["quantization"][0]
        input_zeroPoint = input_details[0]["quantization"][1]
        print('input_details:', input_details)
        print('input_scalefactor:', input_scalefactor, ', input_zeroPoint:', input_zeroPoint)
        output_details=interpreter.get_output_details()
        print('output_details:', output_details)
        output_scalefactor_box = output_details[0]["quantization"][0]
        output_zeroPoint_box = output_details[0]["quantization"][1]
        output_scalefactor_classes = output_details[1]["quantization"][0]
        output_zeroPoint_classes = output_details[1]["quantization"][1]
        output_scalefactor_scores = output_details[2]["quantization"][0]
        output_zeroPoint_scores = output_details[2]["quantization"][1]
        output_scalefactor_num_detections = output_details[3]["quantization"][0]
        output_zeroPoint_num_detections = output_details[3]["quantization"][1]
        print('output_scalefactor_box:', output_scalefactor_box, ', output_zeroPoint_box:', output_zeroPoint_box)
        print('output_scalefactor_classes:', output_scalefactor_classes, ', output_zeroPoint_classes:', output_zeroPoint_classes)
        print('output_scalefactor_scores:', output_scalefactor_scores, ', output_zeroPoint_scores:', output_zeroPoint_scores)
        print('output_scalefactor_num_detections:', output_scalefactor_num_detections, ', output_zeroPoint_num_detections:', output_zeroPoint_num_detections)


        for i in range(len(images_path)):
            path,img, src_img,uint8_img = get_data(images_path,i)
            
            start = time.time()
            if full_float:# full float
                interpreter.set_tensor(input_details[0]["index"],img)
            else:  #full uint8
                interpreter.set_tensor(input_details[0]["index"],uint8_img)
                #print('uint8_img:',uint8_img)
                
            interpreter.invoke()
            boxes=interpreter.get_tensor(output_details[0]["index"])
            classes=interpreter.get_tensor(output_details[1]["index"])
            scores=interpreter.get_tensor(output_details[2]["index"])
            num_detections=interpreter.get_tensor(output_details[3]["index"])
            print('boxes:', boxes)
            print('classes:', classes)
            print('scores:', scores)
            print('num_detections:', num_detections)
            score_threshold = 0.5
            
            #post process
            get_ssd_results(
            src_img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_number=None,
            min_score_thresh=score_threshold)
            
            print(path + ' Over!!!')
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                              src_img,
                              np.squeeze(boxes),
                              np.squeeze(classes).astype(np.int32),
                              np.squeeze(scores),
                              category_index,
                              use_normalized_coordinates=True,
                              line_thickness=2,
                              min_score_thresh=score_threshold)
            end = time.time()
            print('frame:', 1.0 / (end - start))
            cv2.imshow("capture", src_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()