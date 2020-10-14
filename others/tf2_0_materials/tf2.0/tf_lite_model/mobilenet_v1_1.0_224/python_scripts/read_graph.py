# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import sys
from utils import sort_ops ,_get_ops_in_path
from GraphBuilder import GraphBuilder
from MergeLayers import merge_layers
from NodeObj import OPNode,TSNode

tf.compat.v1.disable_eager_execution()

def remove_identity_const(sorted_ops):
    ops=[]
    for op in sorted_ops:
        if op.type=='Const':
            continue
        elif op.type=='Identity':
            output = op.outputs[0]
            output.identity_from=op.inputs[0]
            sorted_ops.remove(op)
            # print('--->',output.identity_from)
        else:
            continue

    return sorted_ops
def read_graph_from_pb(tf_model_path ,input_names,output_name):  
    with open(tf_model_path, 'rb') as f:
        serialized = f.read() 
    tf.compat.v1.reset_default_graph()
    gdef = tf.compat.v1.GraphDef()
    gdef.ParseFromString(serialized) 
    with tf.Graph().as_default() as g:
        tf.import_graph_def(gdef, name='') 
    
    with tf.compat.v1.Session(graph=g) as sess: 
        OPS=get_ops_from_pb(g,input_names,output_name)
    return OPS
 
def remove_ops_before_inputs(inputs,ops):
    tensor_queue=inputs.copy()  
    visited_ts=set()
    invalid_ops=set()
    while len(tensor_queue)>0:
        ts = tensor_queue.pop(0) 
        if not ts.op in invalid_ops: 
            invalid_ops.add(ts.op)
            tensor_queue=tensor_queue+[inp for inp in  ts.op.inputs if not inp in visited_ts]  
        visited_ts.add(ts) 
    ops = [op for op in ops if not op in invalid_ops]
    ops = get_connected_ops(ops,inputs) 
    return ops
def get_connected_ops(ops_set,start_tensors):
    visited_ts = set()
    visited_ops=set()
    ts_queue=start_tensors 
    while len(ts_queue)>0:
        ts = ts_queue.pop(0) 
        if ts.op in ops_set:
            visited_ops.add(ts.op)
            ts_queue=ts_queue+[input for input in ts.op.inputs if not input in visited_ts]
        for op in ts.consumers():
            if op in ops_set:
                visited_ops.add(op) 
                ts_queue=ts_queue+[output for output in op.outputs if not output in visited_ts]
        visited_ts.add(ts)
    ops = [op for op in ops_set if op in visited_ops] 
    return ops
 
def ops_to_OPNodes(ops,inputs):

    ops_map = dict()
    ts_set=set()
    ts_map=dict()
   
    for op in ops:
        op_node = OPNode(op)
        ops_map[op]= op_node
        ts_set |=set([ts for ts in op.inputs])
        ts_set |=set([ts for ts in op.outputs])
    for ts in ts_set:
        ts_map[ts]=TSNode(ts,None) 
    for op,op_node in ops_map.items(): 
        inps=[]
        for inp in op.inputs:#�޸Ľڵ�����
            inp = ts_map[inp]
            inps.append(inp)
        op_node.inputs=inps
        outputs=[]
        for output in op.outputs:#�޸Ľڵ����
            output = ts_map[output]
            outputs.append(output)
        op_node.outputs=outputs
    for ts,ts_node in ts_map.items():
        consumers=[] 
        for op in ts.consumers():
            if op in ops:
                consumers.append(ops_map[op]) 
        ts_node.next_ops=consumers 
        ts_node.op = ops_map.get(ts.op,None)
        if ts_node.op==None:
            print('---->',ts_node.name)
    print(inputs)
    #��inputs��placeholder�滻
    replace_input=dict() 
    for input in inputs:#��inputӳ��placeholder 
        # if not input.op.type=='Placeholder':
        input_shape = input.get_shape()
        if input_shape==None:
            input_shape=[None,None,None,None]
        ph = tf.compat.v1.placeholder(input.dtype,input_shape)
        print(ph.get_shape())
        replace_input[input.name] = ph
        ph_node = OPNode(ph.op) 
        ops_map[ph.op]=ph_node
   
    for op,op_node in ops_map.items():
        new_inputs=[] 
        for input in op_node.inputs:
            input = replace_input.get(input.name,input) #placeholder output
            new_inputs.append(input) 
        op_node.inputs=new_inputs 
        
    return ops_map.values() 

def get_ops_from_inputs_outputs(graph, inputs,outputs):
    ops = graph.get_operations() 
    ops=remove_ops_before_inputs(inputs.copy(),ops)
    ops = ops_to_OPNodes(ops,inputs)
    return ops
def get_ops_from_pb(graph,input_names,output_name,save_ori_network=True):
    if save_ori_network:
        with open('ori_network.txt','w+') as w: 
            OPS=graph.get_operations()
            for op in OPS:
                txt = str([v.name for v in op.inputs])+'---->'+op.type+'--->'+str([v.name for v in op.outputs])
                w.write(txt+'\n') 
    inputs_tf = [graph.get_tensor_by_name(input_name) for input_name in input_names]
    output_tf =graph.get_tensor_by_name(output_name) 
    OPS =get_ops_from_inputs_outputs(graph, inputs_tf,[output_tf] ) 
    with open('network.txt','w+') as w: 
        for op in OPS:
            txt = str([v.name for v in op.inputs])+'---->'+op.type+'--->'+str([v.name for v in op.outputs])
            w.write(txt+'\n') 
    OPS = sort_ops(OPS)
    OPS = merge_layers(OPS)
    return OPS
def read_graph_from_ckpt(ckpt_path,input_names,output_name ):   
    saver = tf.compat.v1.train.import_meta_graph(ckpt_path+'.meta',clear_devices=True)
    graph = tf.compat.v1.get_default_graph()
    with tf.compat.v1.Session( graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) 
        saver.restore(sess,ckpt_path) 
        
        with open('output_ori_network.txt','w+') as w: 
            OPS=graph.get_operations()
            for op in OPS:
                txt = str([v.name for v in op.inputs])+'---->'+op.type+'--->'+str([v.name for v in op.outputs])
                w.write(txt+'\n') 
                
        output_tf =graph.get_tensor_by_name(output_name) 
        pb_graph = tf.compat.v1.graph_util.convert_variables_to_constants( sess, graph.as_graph_def(), [output_tf.op.name]) 
     
    with tf.Graph().as_default() as g:
        tf.import_graph_def(pb_graph, name='')  
    with tf.compat.v1.Session(graph=g) as sess:
        OPS=get_ops_from_pb(g,input_names,output_name)
    return OPS

def gen_graph(ops,html_dst):
    gb = GraphBuilder(html_dst) 
    for op in ops:
        if not len(op.outputs)>0:
            continue  
        if(op.type=='Placeholder'):
            continue
        gb.add_op(op )
    gb.build()
def print_graph(ops):
    
    for op in ops:
        output = op.outputs[0] 
        print(op.inputs,output)
def read_graph(model_path,input_names,output_name,html_dst):
    dir_path = os.path.dirname(html_dst)
    if len(dir_path)>0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if model_path.endswith('pb'):
        ops = read_graph_from_pb( model_path ,input_names,output_name)	
    else:
        ops = read_graph_from_ckpt(model_path ,input_names,output_name)
    
    print(dir_path)
    gen_graph(ops,html_dst)
if __name__=='__main__':
    #model_path = sys.argv[1]
    #input_names = sys.argv[2]
    #output_name = sys.argv[3]
    #html_dst = sys.argv[4]
    
    #model_path = '../mobilenet_v1_1.0_224_frozen.pb'
    #input_names = 'input:0'
    #output_name = 'MobilenetV1/Predictions/Reshape_1:0'
    #html_dst = 'dst.html'
    
    model_path = '../mobilenet_v1_1.0_224.ckpt'
    input_names = 'batch:0'
    output_name = 'MobilenetV1/Predictions/Reshape_1:0'
    html_dst = 'checkpoint.html'
    
    input_names=input_names.split(',')
    read_graph(model_path,input_names,output_name,html_dst)
  
# read_graph('../../mobilenet_v1_1.0_192.ckpt',['batch:0'],'MobilenetV1/Predictions/Reshape_1:0','output/html_dst3.html')
# read_graph( '../../mobilenet_v1_1.0_192_frozen.pb' ,['input:0'],'MobilenetV1/Predictions/Reshape_1:0','output/html_dst1.html')