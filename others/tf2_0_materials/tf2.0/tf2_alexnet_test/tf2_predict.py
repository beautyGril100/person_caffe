# coding=gbk
import tensorflow as tf
from tf2_alexnet_model import AlexNet_v1, AlexNet_v2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import os

im_height = 224
im_width = 224

# load image
img = Image.open("test/tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# scaling pixel value to (0-1)
img = np.array(img) / 255.

# Add the image to a batch where it's the only one sample
img = (np.expand_dims(img, 0))  # 利用expand_dims来扩充batch维度

# read class_indice
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

only_use_weights = True
if only_use_weights:
    model = AlexNet_v1(class_num=5, is_training=False)

    model.load_weights("./save_weights/myAlex.ckpt")
    #model.load_weights("./save_weights/myAlex.h5")
else:
    model = tf.keras.models.load_model("./saved_model/tf2_alexnet.h5", compile=False)

result = np.squeeze(model(img, training=False))
#result = np.squeeze(model.predict(img, verbose=0))

# saved model
model.save(filepath=os.path.join('./saved_model','tf2_alexnet.h5'),save_format='h5')
# 预测之后用squeeze将batch维度压缩，得到每个类别的概率值
predict_class = np.argmax(result)
print(class_indict[str(predict_class)], result[predict_class])  # 输出图像类别和该类别的概率值
plt.show()