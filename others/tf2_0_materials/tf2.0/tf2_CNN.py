import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        print("before expand_dims train_data's shape:")
        print(self.train_data.shape)
        print("before expand_dims test_data's shape:")
        print(self.test_data.shape)
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        print("after expand_dims train_data's shape:")
        print(self.train_data.shape)
        print("train_label's shape")
        print(self.train_label.shape)
        print("after expand_dims test_data's shape:")
        print(self.test_data.shape)
        print("test_label's shape:")
        print(self.test_label.shape)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

# mnist_loader = MNISTLoader()

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64, ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)          # [batch_size, 28, 28, 32]
        x = self.pool1(x)               # [batch_size, 14, 14, 32]
        x = self.conv2(x)               # [batch_size, 14, 14, 64]
        x = self.pool2(x)               # [batch_size, 7, 7, 64]
        x = self.flatten(x)             # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)              # [batch_size, 1024]
        x = self.dense2(x)              # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


if __name__ == '__main__':
    nume_epochs = 5      # 5
    batch_size = 50      # 50
    learning_rate = 0.001
    
    model = CNN()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    num_batches = int(data_loader.num_train_data // batch_size * nume_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            '''
            loss = tf.keras.losses.categorical_crossentropy(
                y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
                y_pred=y_pred
            )
            '''
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    
    print("start evaluate the model ...")
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index:end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())





