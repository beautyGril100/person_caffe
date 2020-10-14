import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("hello, tensorflow 2.0")
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)
print(c)

# scalar
random_float = tf.random.uniform(shape=())

# vector
zero_vector = tf.zeros(shape=(2,))

# matrix
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# tensor's attribute: shape, dtype, numpy()
print("scalar tensor random_float's attribue:")
print(random_float.shape)  # random_float's shape,
print(random_float.dtype)  # random_float's value's data type
print(random_float.numpy)  # random_float's value
print(random_float)

print("vector tensor zero_vector's attribue:")
print(zero_vector.shape)  # zero_vector's shape,
print(zero_vector.dtype)  # zero_vector's value's data type
print(zero_vector.numpy)  # zero_vector's value
print(zero_vector)

print("matrix tensor A's attribue:")
print(A.shape)  # A's shape,
print(A.dtype)  # A's value's data type
print(A.numpy)  # A's value
print(A)

C = tf.add(A, B)
print("C = A + B :")
print(C)

D = tf.matmul(A, B)
print("D = A * B :")
print(D)

# deviate
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
    y_grad = tape.gradient(y, x)
    print("Gradient:")
    print([y, y_grad])

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=[[1.], [1]])
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    w_grad, b_grad = tape.gradient(L, [w, b])
    print("case1:L, w_grad, b_grad:")
    print([L.numpy(), w_grad.numpy(), b_grad.numpy()])


b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    w_grad, b_grad = tape.gradient(L, [w, b])
    print("case2:L, w_grad, b_grad:")
    print([L.numpy(), w_grad.numpy(), b_grad.numpy()])

# Linear Regression
print("case1:numpy implement:")
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# normalization
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

a, b = 0, 0
num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # compute gradients by manual
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # update parameters
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print("a = %f, b = %f" % (a, b))

print("case2:tensorflow implement")
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
print(a.numpy, b.numpy)

print("case3:tensorflow.keras's model and layers:")
X_mu = 2013
X_sigma = 2017 - 2013
X = tf.constant([[(2013 - X_mu)/X_sigma],
                 [(2014 - X_mu)/X_sigma],
                 [(2015 - X_mu)/X_sigma],
                 [(2016 - X_mu)/X_sigma],
                 [(2017 - X_mu)/X_sigma]])
y_mu = 12000
y_sigma = 17500 - 12000
y = tf.constant([[(12000 - y_mu) / y_sigma],
               [(14000 - y_mu) / y_sigma],
               [(15000 - y_mu) / y_sigma],
               [(16500 - y_mu) / y_sigma],
               [(17500 - y_mu) / y_sigma]])


#X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    def call(self,input):
        output = self.dense(input)
        return output

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
num_epoch = 10000
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)

