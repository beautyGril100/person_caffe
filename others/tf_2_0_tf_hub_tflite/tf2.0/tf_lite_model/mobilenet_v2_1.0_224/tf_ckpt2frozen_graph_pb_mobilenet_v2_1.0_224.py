# coding=gbk
import tensorflow as tf
import os
import cv2 as cv
import numpy as np

tf.compat.v1.disable_eager_execution()

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
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.compat.v1.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        
        for op in sess.graph.get_operations():
          print('os.type:', op.type, 'name:', op.name)
        #if (op.type == "Softmax"):
            #outputs.append(op.name.replace('import/', ''))
        #saver.restore(sess, os.path.expanduser("/workspace/hwzhu/tf_1.12.0_workspace/models_ckpt/resnet_v1_50/resnet_v1_50.ckpt"))
        graph_def_removed_training_nodes = tf.compat.v1.graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
        output_node_names = FetchOutput(sess)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=graph_def_removed_training_nodes, # input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)  # 如果有多个输出节点，以逗号隔开
 
        with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        
        #saver.restore(sess, os.path.expanduser("/workspace/hwzhu/tf_1.12.0_workspace/models_ckpt/resnet_v1_50/resnet_v1_50.ckpt"))
        #graph_def_removed_training_nodes = graph_util.remove_training_nodes(sess.graph_def, protected_nodes=None)
        #outputs = FetchOutput(sess)
        #graph_def_const = graph_util.convert_variables_to_constants(sess, graph_def_removed_training_nodes, outputs)
        tf.compat.v1.train.write_graph(output_graph_def, os.path.expanduser("./"), "deploy_hwzhu.pb", as_text=False)


if __name__ == '__main__':
    modelpath="./mobilenet_v2_1.0_224.ckpt"
    freeze_graph(modelpath,"frozen_hwzhu.pb")
    print("From ckpt Convert to pb finish!")

'''
    test_pb_model=False
    test_tflite_model=True
    read_cahnge_graph=False
    
    pb_model_path="mobilenet_v1_0.5_224_frozen.pb"
    tflite_model_path = "mobilenet_v1_0.5_224_float.tflite"
    #input_node_name="iamges"
    #output_node_name="MobilenetV1/Predictions/Softmax"
    
    src_img=cv.imread("dog.jpg")
    cv.imwrite("src.jpg",src_img)
    
    src_img=cv.resize(src_img,(224,224))
    src_img=cv.cvtColor(src_img,cv.COLOR_BGR2RGB)
    src_img=src_img/127.5-1
    src_img=src_img.astype("float32")
    
    src_img=src_img.reshape((1,224,224,3))
    
    if test_tflite_model:
        interpreter=tf.compat.v1.lite.Interpreter(tflite_model_path)
        interpreter.allocate_tensors()
 
        input_details=interpreter.get_input_details()
        print('input_details:', input_details)
        output_details=interpreter.get_output_details()
        print('output_details:', output_details)
 
        #src_img=src_img.astype("uint8")  #full integer tflite
        interpreter.set_tensor(input_details[0]["index"],src_img)
 
        interpreter.invoke()
        output_data=interpreter.get_tensor(output_details[0]["index"])
        print('tflite_output_data:', output_data)
        print('tflite_output_data[0]:', output_data[0])
        result=output_data[0]
 
        result=(result+1)*127.5
        result[result>255]=255
        result[result<0]=0
        result=result.astype(np.uint8)
        #cv.imshow("result",result)
        cv.imwrite("tflite_result.jpg", result)
        #cv.waitKey()
    if test_pb_model:
        input_image=tf.compat.v1.placeholder(tf.float32,(1,224,224,3))
 
        with open(pb_model_path,"rb") as f:
            graph_def=tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            out_result=tf.compat.v1.import_graph_def(graph_def,input_map={"input:0":input_image},return_elements=["MobilenetV1/Predictions/Softmax:0"])
        sess=tf.compat.v1.Session()
        result=sess.run(out_result,feed_dict={input_image:src_img})
        print('pb_results:', result)
        print('pb_results[0]:', result[0])
        print('pb_results[0][0]:', result[0][0])
 
        #result=result[0][0]
        #result=(result+1)*127.5
        #result[result>255]=255
        #result[result<0]=0
        #result=result.astype(np.uint8)
 
        #cv.imshow("pb_resut",result)
        #cv.imwrite("pb_result.jpg", result)
        #cv.waitKey()
'''