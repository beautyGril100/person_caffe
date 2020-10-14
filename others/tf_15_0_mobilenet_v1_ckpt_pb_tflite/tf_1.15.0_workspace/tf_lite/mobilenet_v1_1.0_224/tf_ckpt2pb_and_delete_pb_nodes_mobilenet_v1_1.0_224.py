# coding=gbk
import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import tensor_shape_pb2

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
        output_node_names = FetchOutput(sess)
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=graph_def_removed_training_nodes, # input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)  # 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        
        
def delete_ops_from_pb_graph(input_pb_model_file_path, output_pb_model_file_path):
    with open(input_pb_model_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Delete nodes
    nodes = []
    
    
    need_add_placeholder_flag = True
    input_nodes_names = ['batch']
    ceil_nodes_names = [['batch'], ['MobilenetV1/Logits/AvgPool_1a/AvgPool']]
    floor_nodes_names = ['MobilenetV1/MobilenetV1/Conv2d_0/Conv2D', 'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D']
    need_deleted_nodes_names = ['batch/fifo_queue', 'batch/n']
    need_deleted_nodes_keywordsInNames = ['dropout']
    for node in graph_def.node:
        #*****************add input placeholder start***********************
        for input_node_name in input_nodes_names:
            if input_node_name == node.name:
                print('src_input_node_attr:')
                print(node)
                del node.input[:]
                node.attr.clear()
                node.op = 'Placeholder'
                node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
                node.attr['shape'].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1), \
                                                                                                                  tensor_shape_pb2.TensorShapeProto.Dim(size=224), \
                                                                                                                  tensor_shape_pb2.TensorShapeProto.Dim(size=224), \
                                                                                                                  tensor_shape_pb2.TensorShapeProto.Dim(size=3)])))
                print('modified_input_node_attr:')
                print(node)
                break
        #*****************add input placeholder end*************************         
        
        #*****************delete unused nodes start************************* 
        for ceil_node, floor_node in zip(ceil_nodes_names, floor_nodes_names):
            if floor_node == node.name:
                print('ceil_node:', ceil_node)
                print('floor_node:', floor_node)
                floor_node_inputs = []
                floor_node_inputs.extend(ceil_node)
                #print('1-floor_node_inputs:',floor_node_inputs)
                for input_tensor in node.input:
                    for delete_nodes_keyword in need_deleted_nodes_keywordsInNames:
                        if delete_nodes_keyword in input_tensor:
                            node.input.remove(input_tensor)
                            break
                
                for valid_input_tensor in node.input:
                    print('valid_input_tensor:', valid_input_tensor)
                    if valid_input_tensor not in ceil_node:
                        floor_node_inputs.append(valid_input_tensor) 
                    #print('2-floor_node_inputs:',floor_node_inputs) 
                
                #print('1-node.input:', node.input)
                del node.input[:]
                #print('2-node.input:', node.input)
                node.input.extend(floor_node_inputs)
                #print('3-node.input:', node.input)
                
                break
        
        valid_node = True
        for need_deleted_nodes_keyword in need_deleted_nodes_keywordsInNames:
            if need_deleted_nodes_keyword in node.name:
                valid_node = False
                print('drop node name:', node.name)
                break
                
        if node.name not in need_deleted_nodes_names and valid_node:
            nodes.append(node)
           
        #*****************delete unused nodes end*************************** 
        '''
        new_inputs = []
        if 'batch' == node.name:
            print('batch:',node)
            del node.input[:]
            node.attr.clear()
            node.op = 'Placeholder'
            node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            node.attr['shape'].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1), tensor_shape_pb2.TensorShapeProto.Dim(size=224), tensor_shape_pb2.TensorShapeProto.Dim(size=224), tensor_shape_pb2.TensorShapeProto.Dim(size=3)])))
            print('m:',node)
        
        if 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D' == node.name:
            print('node.name:', node)
        if 'MobilenetV1/Logits/AvgPool_1a/AvgPool' == node.name:
            new_inputs.append(node.name)
        if 'dropout' in node.name and 0 != len(new_inputs):
            for new_input in new_inputs:
                for drop_input in node.input:
                    if new_input == drop_input:
                        node.input.remove(drop_input)
        if 'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D' == node.name:
            print('*****src-node.input:', node.input)
            for input_tensor in node.input:
                if 'dropout' in input_tensor:
                    node.input.remove(input_tensor)
            for valid_input_tensor in node.input:
                print('valid_input_tensor:', valid_input_tensor)
                new_inputs.append(valid_input_tensor)   
            print('new_inputs:', new_inputs)
            print('type(node.input):', type(node.input))
            print('*****del-node.input:', node.input)
            del node.input[:]
            node.input.extend(new_inputs)
            print('*****new-node.input:', node.input)   
        if 'FIFOQueueV2' == node.op or node.op == 'QueueDequeueManyV2' or 'dropout' in node.name or 'batch/n' == node.name:
            print('Drop', node.name)
        else:
            nodes.append(node)
        '''

    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes)

    # Delete references to deleted nodes
    '''
    for node in mod_graph_def.node:
        inp_names = []
        for inp in node.input:
            if 'Neg' in inp:
                pass
            else:
                inp_names.append(inp)

        del node.input[:]
        node.input.extend(inp_names)
    '''
    
    with open(output_pb_model_file_path, 'wb') as f:
        f.write(mod_graph_def.SerializeToString())

if __name__ == '__main__':
    input_pb_model_file_path = 'ckpt2frozenPb_test.pb'
    modelpath="./mobilenet_v1_1.0_224.ckpt"
    freeze_graph(modelpath,input_pb_model_file_path)
    print("From ckpt Convert to pb finish!")
    
    output_pb_model_file_path = 'ckpt2frozenPb_test_deleted.pb'
    delete_ops_from_pb_graph(input_pb_model_file_path, output_pb_model_file_path)
    print("Modified pb file finish!")
    
    dir_path = 'test_images'
    images_path =[dir_path+'/'+n for n in os.listdir(dir_path)]
    
    label=load_label()
    
    test_pb_model = True
    if test_pb_model:
        input_image=tf.placeholder(tf.float32,(1,224,224,3))
    
        with open(output_pb_model_file_path,"rb") as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            out_result=tf.import_graph_def(graph_def,input_map={"batch:0":input_image},return_elements=["MobilenetV1/Predictions/Softmax:0"])  #
            #取概率最大的5个类别及其对应概率
            output = tf.nn.top_k(out_result[0], k=5, sorted=True)
        sess=tf.Session()
        for i in range(len(images_path)):
            path,img, src_img, uint8_img = get_data(images_path,i)
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