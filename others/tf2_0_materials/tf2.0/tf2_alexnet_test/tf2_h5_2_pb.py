# coding=gbk
import tensorflow as tf
import os
from functools import partial
import numpy as np
import shutil

print("tf.__version__:", tf.__version__)
# hdf5转pb模型
print('************************************ convert hdf5 to pb ***************************************')
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #调用模型前一定要执行该命令, 0:test, 1:train
tf1.disable_v2_behavior() #禁止tensorflow2.0的行为
#加载hdf5模型
output_folder = './saved_model'
hdf5_pb_model = tf.keras.models.load_model(output_folder + "/tf2_alexnet.h5")
def freeze_session(session,keep_var_names=None,output_names=None,clear_devices=True):
    graph = session.graph
    with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
#         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names",output_names)
        input_graph_def = graph.as_graph_def()
#         for node in input_graph_def.node:
#             print('node:', node.name)
        print("hdf5 %d node1" % len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#去掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("frozenGraph %d node1" % len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in hdf5_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder, "tf2_alexnet.pb", as_text=False)
tf1.train.write_graph(frozen_graph, output_folder, "tf2_alexnet.pbtxt", as_text=True)
print('************************************ convert hdf5 to pb Successfully!!!***************************************')