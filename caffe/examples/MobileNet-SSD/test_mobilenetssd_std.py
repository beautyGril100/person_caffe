import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/lfwang/ssd_new/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

#f=open('ssd_result_caffe.txt','w')
caffe.set_mode_cpu()

net_file= './mobilenetStd/deploy.prototxt'  
caffe_model='./mobilenetStd/deploy.caffemodel'  
test_dir = "images_jd"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

net_result_1 = 'net_result_1.txt'
net_result_2 = 'net_result_2.txt'
 
 
CLASSES = ('__background__',
           'jd_market', 'xiuzi')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img =img
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    
    for layer_name, param in net.params.items():
      print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


    for layer_name, blob in net.blobs.items():
      print layer_name + '\t' + str(blob.data.shape)
      blob_1d = np.reshape(blob.data,(1,-1))
      layer_name_modify = layer_name.replace('/','_')
      np.savetxt('mobilenetStd/temp/'+layer_name_modify+'.txt',blob_1d,fmt="%.5f", delimiter="\n")
    
   # mbox_priorbox = out['mbox_priorbox']
    
    detection_out = out['detection_out']
    
    
   # mbox_priorbox_1d = np.reshape(mbox_priorbox,(1,-1))
    detection_out_1d = np.reshape(detection_out,(1,-1))
    
    
  
    #np.savetxt(net_result_1,mbox_priorbox_1d,fmt="%.5f", delimiter="\n")
    np.savetxt(net_result_2,detection_out_1d,fmt="%.5f", delimiter="\n")

    return True


for index in range(0,1):
    img_path = "ruiwei/0.jpg"
    
   # f.write('image '+str(index)+ '\r\n')
    
    detect(img_path)

#for f in os.listdir(test_dir):
#    if detect(test_dir + "/" + f) == False:
#       break
