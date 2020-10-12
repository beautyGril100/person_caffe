import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/lfwang/ssd_new/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

#f=open('ssd_result_caffe.txt','w')
caffe.set_mode_cpu()

net_file_0= './ruiwei/mobilenet_ssd_ruiwei_320_570/deploy.prototxt'  
caffe_model_0='./ruiwei/mobilenet_ssd_ruiwei_320_570/deploy.caffemodel'  

net_file_1= './ruiwei/mobilenet_ssd_ruiwei_320_570/deploy_merge.prototxt'  
caffe_model_1='./ruiwei/mobilenet_ssd_ruiwei_320_570/deploy_merge.caffemodel'  

if not os.path.exists(caffe_model_0):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()


def convert_net():
    net0 = caffe.Net(net_file_0,caffe_model_0,caffe.TEST)
    net1 = caffe.Net(net_file_1,caffe.TEST)
    for i in list(net0.params):
        print i
        if 'conv4_3_mbox_loc_frontalface' ==  i:
            #set_trace()
            net1.params['conv4_3_mbox_loc_conf_union_frontalface'][0].data[:8,:,:,:] = net0.params[i][0].data[:,:,:,:]
            #for j in range(1, len(net0.params[i])):
            #    net1.params['conv4_3_mbox_loc_conf_union_frontalface'][j].data[...] = net0.params[i][j].data[...]
        elif 'conv4_3_mbox_conf_frontalface' ==  i:    
            #set_trace()
            net1.params['conv4_3_mbox_loc_conf_union_frontalface'][0].data[8:,:,:,:] = net0.params[i][0].data[:,:,:,:]
            #for j in range(1, len(net0.params[i])):
            #    net1.params['conv4_3_mbox_loc_conf_union_frontalface'][j].data[...] = net0.params[i][j].data[...]       
        elif 'conv3_3_mbox_loc_frontalface' ==  i:
            #set_trace()
            net1.params['conv3_3_mbox_loc_conf_union_frontalface'][0].data[:8,:,:,:] = net0.params[i][0].data[:,:,:,:]
            #for j in range(1, len(net0.params[i])):
            #    net1.params['conv4_3_mbox_loc_conf_union_frontalface'][j].data[...] = net0.params[i][j].data[...]
        elif 'conv3_3_mbox_conf_frontalface' ==  i:    
            #set_trace()
            net1.params['conv3_3_mbox_loc_conf_union_frontalface'][0].data[8:,:,:,:] = net0.params[i][0].data[:,:,:,:]
            #for j in range(1, len(net0.params[i])):
            #    net1.params['conv4_3_mbox_loc_conf_union_frontalface'][j].data[...] = net0.params[i][j].data[...]       
        else:
            for j in range(len(net1.params[i])):
                net1.params[i][j].data[...] = net0.params[i][j].data[...]
    net1.save(caffe_model_1)
    print('123123')  
    
    
convert_net()