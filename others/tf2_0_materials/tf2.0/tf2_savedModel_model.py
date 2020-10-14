import tensorflow as tf
import numpy as np
from tf2_CNN import MNISTLoader
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CNN_TEST(tf.keras.Model):
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

    @tf.function
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


nume_epochs = 1      # 5
batch_size = 50      # 50
learning_rate = 0.001
'''
model = CNN_TEST()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * nume_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        
        # loss = tf.keras.losses.categorical_crossentropy(
        #    y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
        #    y_pred=y_pred
        #)
        
        loss = tf.reduce_mean(loss)
        print("savedModel batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

tf.saved_model.save(model, "saved/2")
'''

print("start evaluate the model ...")
batch_size = 50      # 50
model = tf.saved_model.load("saved/2")
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.call(data_loader.test_data[start_index:end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())





