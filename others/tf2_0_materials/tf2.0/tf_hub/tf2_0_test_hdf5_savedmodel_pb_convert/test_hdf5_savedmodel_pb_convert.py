# coding=gbk
'''
1��ѵ��ģ��
2������ģ�ͼ以ת����֤
2.1 hdf5תsaved model
2.2 saved modelתhdf5
2.3 ����ģ�;��Ȳ���
2.4 hdf5��savedģ��תtensorflow1.x pbģ��
2.5 ���ز�����pbģ��
�ܽ�
'''
'''
1��ѵ��ģ��
���ǽ���ѵ���õ�ģ�ͷֱ𱣴��hdf5��saved model��ʽ��Ȼ���������֮��Ļ���ת���Լ��ֱ�תtensorflow1.x��pb��ʽ�������У�
1.hdf5תsaved model,����֤ת�����saved model��ֱ�ӱ����saved model���޲����ԣ���С�����ȣ�
2.saved modelתhdf5,����֤ת�����hdf5��ֱ�ӱ����hdf5���޲����ԣ���С�����ȣ�
3.hdf5תpb,����֤ת�����pb��ֱ��ԭʼ�ĵ�hdf5���޲����ԣ���С�����ȣ�
4.saved modeתpb,����֤ת�����pb��ֱ��ԭʼ�ĵ�saved mode���޲����ԣ���С�����ȣ�
5.�Ա�hdf5��תpb��saved model��תpb������
'''
import tensorflow as tf
import os
from functools import partial
import numpy as np
import shutil

print("tf.__version__:", tf.__version__)

training_flag = True

output_folder="./test/hdf5_model"
output_folder1="./test/saved_model"
output_folder2="./test/pb_model"
    
if training_flag:
    batch_size=64
    epochs=6
    regularizer=1e-3
    total_train_samples=60000
    total_test_samples=10000
    
    
    for m in (output_folder,output_folder1,output_folder2):
        if os.path.exists(m):
            inc=input("The model(%s) saved path has exist,Do you want to delete and remake it?(y/n)"%m)
            while(inc.lower() not in ['y','n']):
                inc=input("The model saved path has exist,Do you want to delete and remake it?(y/n)")
            if inc.lower()=='y':
                shutil.rmtree(m)
                os.makedirs(m)
        elif not os.path.exists(m):
            os.makedirs(m)
    
    
    #ָ���Կ�
    physical_devices = tf.config.experimental.list_physical_devices('GPU')#�г����пɼ��Կ�
    print("All the available GPUs:\n",physical_devices)
    if physical_devices:
        gpu=physical_devices[0]#��ʾ��һ���Կ�
        tf.config.experimental.set_memory_growth(gpu, True)#������Ҫ�Զ������Դ�
        tf.config.experimental.set_visible_devices(gpu, 'GPU')#ֻѡ���һ��
    
    #׼������
    fashion_mnist=tf.keras.datasets.fashion_mnist
    (train_x,train_y),(test_x,test_y)=fashion_mnist.load_data()
    
    train_x,test_x = train_x[...,np.newaxis]/255.0,test_x[...,np.newaxis]/255.0
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
     
    train_ds=train_ds.shuffle(buffer_size=batch_size*10).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#����repeat��ִ��һ�ξ���
    
    #����ģ��
    l2 = tf.keras.regularizers.l2(regularizer)#����ģ�����򻯷���
    ini = tf.keras.initializers.he_normal()#���������ʼ������
    conv2d = partial(tf.keras.layers.Conv2D,activation='relu',padding='same',kernel_regularizer=l2,bias_regularizer=l2)
    fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)
    maxpool=tf.keras.layers.MaxPooling2D
    dropout=tf.keras.layers.Dropout
    def test_model():
        x_input = tf.keras.layers.Input(shape=(28,28,1),name='input_node')
        x = conv2d(128,(5,5))(x_input)
        x = maxpool((2,2))(x)
        x = conv2d(256,(5,5))(x)
        x = maxpool((2,2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = fc(128)(x)
        x_output=fc(10,activation=None,name='output_node')(x)
        model = tf.keras.models.Model(inputs=x_input,outputs=x_output) 
        return model
    model = test_model()
    print(model.summary())
    
    #����ģ�� -- ����ѵ������
    initial_learning_rate=0.01
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate,momentum=0.95)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=['accuracy','sparse_categorical_crossentropy']
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    
    #ѵ��ģ��
    H=model.fit(train_ds,epochs=6,
                steps_per_epoch=np.floor(len(train_x)/batch_size).astype(np.int32),
                validation_data=test_ds,
                validation_steps=np.ceil(len(test_x)/batch_size).astype(np.int32),
                verbose=1)
    
    #�ֱ𱣴����ָ�ʽ��ģ��
    model.save(filepath=os.path.join(output_folder,'hdf5_model.h5'),save_format='h5')
    model.save(filepath=output_folder1,save_format='tf')
    #����warning��Ϣ��tensorflow������������ͬ������
    
    #ѡȡ��һ����������Ϊ��������,����������ͬģ��֮��ľ���
    test_sample = train_x[0:1]
    test_y=train_y[0]
    out = model.predict(test_sample)
    print('**************************Training ******************************')
    print("probs:",out[0])
    print("true label:{} pred label:{}".format(test_y,np.argmax(out)))

#��������������������������������������������������2������ģ�ͼ以ת����֤��������������������������������������������������������������
# 2.1 hdf5תsaved model
tf.keras.backend.clear_session()
hdf5_model = tf.keras.models.load_model(output_folder + "/hdf5_model.h5")
hdf5_2_savedModel = "./test/hdf5_2_saved_model"
hdf5_model.save(hdf5_2_savedModel,save_format='tf')

# 2.2 saved modelתhdf5
# ��saved model���޷�ת����hdf5ģ�͵ģ����Ը��˸о���ѵ�������б���hdf5��ʽ��ģ�ͱȽϺ�
savedModel_2_hdf5_flag = False
if savedModel_2_hdf5_flag:
    tf.keras.backend.clear_session()
    saved_model = tf.keras.models.load_model(output_folder1)
    savedModel_2_hdf5 = './test/hdf5_model'
    saved_model.save(savedModel_2_hdf5 + "/saved2hdf5_model.h5",save_format='h5')

# 2.3 ����ģ�;��Ȳ���
# ��������ģ�͵ľ��ȣ�ԭʼhdf5ģ�ͣ�ԭʼsaved model��hdf5ת����saved model
origin_hdf5_model = tf.keras.models.load_model(output_folder + "/hdf5_model.h5")
origin_saved_model = tf.keras.models.load_model(output_folder1)
converted_saved_model = tf.keras.models.load_model(hdf5_2_savedModel)
out1 = origin_hdf5_model.predict(test_sample)
out2 = origin_saved_model.predict(test_sample)
out3 = converted_saved_model.predict(test_sample)
print('**************************origin_hdf5_model ******************************')
print("probs:",out1[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out1)))
print('**************************origin_saved_model ******************************')
print("probs:",out2[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out2)))
print('**************************origin_saved_model ******************************')
print("probs:",out3[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out3)))
np.testing.assert_array_almost_equal(out,out1)
np.testing.assert_array_almost_equal(out,out2)
np.testing.assert_array_almost_equal(out,out3)
print('Successfully!!! origin_hdf5_model == origin_saved_model == origin_saved_model')

# 2.4 hdf5��savedģ��תtensorflow1.x pbģ��
# hdf5תpbģ��
print('************************************ convert hdf5 to pb ***************************************')
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #����ģ��ǰһ��Ҫִ�и�����
tf1.disable_v2_behavior() #��ֹtensorflow2.0����Ϊ
#����hdf5ģ��
hdf5_pb_model = tf.keras.models.load_model(output_folder + "/hdf5_model.h5")
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
        print("hdf5 len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#ȥ���������޹ص�����
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("frozenGraph len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in hdf5_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "hdf5_2_pb.pb", as_text=False)
print('************************************ convert hdf5 to pb Successfully!!!***************************************')
# saved modelתpbģ��
print('************************************ convert savedModel to pb ***************************************')
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #����ģ��ǰһ��Ҫִ�и�����
tf1.disable_v2_behavior() #��ֹtensorflow2.0����Ϊ
#����hdf5ģ��
saved_pb_model = tf.keras.models.load_model(output_folder1)
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
        print("savedModel len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#ȥ���������޹ص�����
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("frozenGraph len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in saved_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "savedModel_2_pb.pb", as_text=False)

print('************************************ convert savedModel to pb Successfully!!!***************************************')
#ת�����saved modelתpb
print('************************************ convert hdf5_2_savedModel to pb ***************************************')
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #����ģ��ǰһ��Ҫִ�и�����
tf1.disable_v2_behavior() #��ֹtensorflow2.0����Ϊ
#����hdf5ģ��
hdf5_2_savedModel_pb_model = tf.keras.models.load_model(hdf5_2_savedModel)
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
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#�Ƶ��������޹ص�����
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in hdf5_2_savedModel_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "hdf5_2_savedModel_2_pb.pb", as_text=False)
print('************************************ convert hdf5_2_savedModel to pb Successfully!!!***************************************')
# 2.5 ���ز�����pbģ��
print('************************************ Method 1 - Testing hdf5_2_pb.pb ***************************************')
def load_graph(file_path):
    with tf1.gfile.GFile(file_path,'rb') as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf1.Graph().as_default() as graph:
        tf1.import_graph_def(graph_def,input_map = None,return_elements = None,name = "",op_dict = None,producer_op_list = None)
    graph_nodes = [n for n in graph_def.node]
    return graph,graph_nodes

graph,graph_nodes = load_graph(output_folder2 + "/hdf5_2_pb.pb")
print("hdf5_2_pb num nodes:",len(graph_nodes))
for node in graph_nodes:
    print('node:', node.name) 

input_node = graph.get_tensor_by_name('input_node:0')
output = graph.get_tensor_by_name('output_node/BiasAdd:0')

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.25# �趨GPUʹ��ռ��
config.gpu_options.visible_device_list = '0'  # '0,1'
config.allow_soft_placement = True
config.log_device_placement = False

with tf1.Session(config=config,graph=graph) as sess:
        logits = sess.run(output, feed_dict = {input_node:test_sample})
print("hdf5_2_pb logits:",logits)
np.testing.assert_array_almost_equal(out,logits)
print('************************************ Method 1 - Testing hdf5_2_pb.pb Successfully!!!***************************************')

print('************************************ Method 2 - Testing hdf5_2_pb.pb ***************************************')
# ����һ�ֵ���hdf5ת����pb(����tensorflow2.x�е���tensorflow1.xת��pb)
tf1.reset_default_graph()
tf1.enable_v2_behavior()#tensorflow2.x�е���tensorflow1.x��������Ҫ����tensorflow2.x������
tf.keras.backend.clear_session()
def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf1.import_graph_def(graph_def, name="")
    wrapped_import = tf1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


with open(output_folder2 + "/hdf5_2_pb.pb",'rb') as f:
    graph_def = tf1.GraphDef()
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        print("node.name",node.name)

model_func = wrap_frozen_graph(
    graph_def, inputs='input_node:0',
    outputs='output_node/BiasAdd:0')

o=model_func(tf.constant(test_sample,dtype=tf.float32))

print(o)

np.testing.assert_array_almost_equal(out,o.numpy())

print('************************************ Method 2 - Testing hdf5_2_pb.pb Successfully!!!***************************************')
test_savedModel_2_pb_flag = False
if test_savedModel_2_pb_flag:
    print('************************************ Testing savedModel_2_pb.pb ***************************************')
    #graph,graph_nodes = load_graph(output_folder2 + "/savedModel_2_pb.pb")
    graph,graph_nodes = load_graph(output_folder2 + "/hdf5_2_savedModel_2_pb.pb")
    print("num nodes",len(graph_nodes))
    for node in graph_nodes:
        print('node:', node.name)
    
    input_node = graph.get_tensor_by_name('input_1:0')
    output = graph.get_tensor_by_name('output_node/bias:0')
    
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.25# �趨GPUʹ��ռ��
    config.gpu_options.visible_device_list = '0'  # '0,1'
    config.allow_soft_placement = True
    config.log_device_placement = False
    
    with tf1.Session(config=config,graph=graph) as sess:
            logits = sess.run(output, feed_dict = {input_node:test_sample})
    print("savedModel_2_pb logits:",logits)
    np.testing.assert_array_almost_equal(out,logits)
    print('************************************ Testing savedModel_2_pb.pb Successfully!!!***************************************')


'''
�ܽ�
1.tensorflow2.x�����hdf5ģ�Ϳ���ת��tensorflow1.x��pb ,Ҳ����ת��tensorflow2.x saved model
2.saved model����ת��pb ,����ת�����޷�ʹ��
3.saved model������ת����hdf5ģ��
4.��tensorflow2.x�п���ʹ��tensorflow1.x��tensorflow2.x���﷨������pb���Ӷ�ѡ��ͬ�İ汾
5.tensorflow2.xѵ����ģ�� ��ת����pb��ֻ����tensorflow1.14��1.15�����á�
�����������Ժ����ֻ����hdf5ģ�ͣ���������ʹ��tensorflow2.x��ѵ��ģ�ͣ����Ҫ��tensorflow2.x�Ƽ��ĸ�ʽ���Ͱ�hdf5ת����save_model���ã����Ҫ�þɵ�tensorflow1.x�汾�����԰�hdf5ת����pb���á�
'''
