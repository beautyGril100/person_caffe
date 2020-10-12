import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/lfwang/ssd_new/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

f=open('ssd_result_caffe.txt','w')
caffe.set_mode_cpu()

net_file= 'MSSD_jd/mobilenet_ssd.prototxt'  
caffe_model='MSSD_jd/mobilenet_ssd.caffemodel'  
test_dir = "images_jd"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
 
 
CLASSES = ('__background__',
           'jd_market', 'xiuzi')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
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
    box, conf, cls = postprocess(origimg, out)
    
    
    for layer_name, param in net.params.items():
      print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


    for layer_name, blob in net.blobs.items():
      print layer_name + '\t' + str(blob.data.shape)
      blob_1d = np.reshape(blob.data,(1,-1))
      layer_name_modify = layer_name.replace('/','_')
      np.savetxt('Moblienet_SSD_JD/temp/'+layer_name_modify+'.txt',blob_1d,fmt="%.5f", delimiter="\n")

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
       string_conf = "%.5f" % conf[i]
       f.write(str(string_conf)+' ')
       f.write(str(box[i][0])+' '+str(box[i][1])+' '+str(box[i][2])+' '+str(box[i][3]) + '\r\n')
   # cv2.imshow("SSD", origimg)
 
    #k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    #if k == 27 : return False
    #return True


for index in range(0,1):
    img_path = "ruiwei/0.jpg"
    
   # f.write('image '+str(index)+ '\r\n')
    
    detect(img_path)

#for f in os.listdir(test_dir):
#    if detect(test_dir + "/" + f) == False:
#       break
