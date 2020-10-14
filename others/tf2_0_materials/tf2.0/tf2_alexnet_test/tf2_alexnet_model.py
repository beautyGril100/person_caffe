# coding=gbk
from tensorflow.keras import layers, models, Model, Sequential

def AlexNet_v1(im_height=224, im_width=224, class_num=1000, is_training=True):
    # tensorflow中的tensor通道排序是NHWC
    # 使用函数形式构建模型，必须加上输入层
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(padding=((1, 2), (1, 2)))(input_image)              # ((top_pad, bottom_pad), (left_pad, right_pad)),output(None, 227, 227, 3)
    x = layers.Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation="relu")(x)       # output(None, 55, 55, 48)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)                              # output(None, 27, 27, 48)
    x = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(x)                              # output(None, 13, 13, 128)
    x = layers.Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(x)                              # output(None, 6, 6, 128)
    x = layers.Flatten()(x)                         # output(None, 6*6*128)
    if is_training:
        x = layers.Dropout(rate=0.2)(x)                      # 为下一层的Dense层加入dropout
    x = layers.Dense(units=2048, activation="relu")(x)    # output(None, 2048)
    if is_training:
        x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dense(units=class_num)(x)                  # output(None, 5), 在这里是可以用layers.Dense(10, activation="softmax")，从而省略后面的softmax层
    predict = layers.Softmax()(x)
    
    model = models.Model(inputs=input_image, outputs=predict)
    return model

class AlexNet_v2(Model):
    def __init__(self, class_num=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
              layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
              layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
              layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
              layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
              layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
              layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
              layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
              layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
              layers.MaxPool2D(pool_size=3, strides=2)])                              # output(None, 6, 6, 128)
              
        self.flatten = layers.Flatten()
        '''
        self.dropout1 = layers.Dropout(0.2)
        self.dense1 = layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
        self.dropout2 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(128, activation="relu"),                                   # output(None, 2048)
        self.dense3 = layers.Dense(class_num),                                                # output(None, 5)
        self.softmax = layers.Softmax()
        '''
        self.classifier = Sequential([
              layers.Dropout(0.2),
              layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
              layers.Dropout(0.2),
              layers.Dense(128, activation="relu"),                                   # output(None, 2048)
              layers.Dense(class_num),                                                # output(None, 5)
              layers.Softmax()
              ])
        
    def call(self, inputs, training=True):
        x = self.features(inputs)
        x = self.flatten(x)
        
        '''
        if training:
            x = self.dropout1(x, training=training)
        print('x.shape: ', x.shape)
        x = self.dense1(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        '''
        
        x = self.classifier(x)
        return x
        
if __name__ == '__main__':
    print('Tensorflow v2 create alexnet model!')
        