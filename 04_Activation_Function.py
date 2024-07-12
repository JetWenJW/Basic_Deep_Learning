import tensorflow as tf
from tensorflow.keras import activations
import numpy as np
import matplotlib.pyplot as plt


# ReLU(Rectified linear unit)
# set x = -10 ~ 10 for test
x = np.linspace(-10, 10, 21)
x_tf = tf.constant(x, dtype = tf.float32)

# ReLU
y = activations.relu(x_tf).numpy()

# Plot 
plt.plot(x, y)
plt.show

# ReLU 參數變換
x = np.linspace(-10, 10,21)
x_tf = tf.constant(x, dtype = tf.float32)
y = activations.relu(x_tf, threshold = 5).numpy()

plt.plot(x, y)
plt.show()

x = np.linspace(-10, 10,21)
x_tf = tf.constant(x, dtype = tf.float32)
y = activations.relu(x_tf, max_value = 5).numpy()

plt.plot(x, y)
plt.show()

# x = np.linspace(-10, 10,21)
# x_tf = tf.constant(x, dtype = tf.float32)
# y = activations.relu(x_tf, alpha = 0.5).numpy()
# print(y)
# plt.plot(x, y)
# plt.show()

# Sigmoid
x = np.linspace(-10, 10, 21)
x_tf = tf.constant(x, dtype = tf.float32)
y = activations.sigmoid(x_tf).numpy()

plt.axvline(-4, color = 'r')
plt.axvline(4, color = 'r')

plt.plot(x, y)
plt.show()

# 使用NumPy計算Sigmoid
x = np.linspace(-10, 10, 21)
y = 1 / (1 + np.e ** (-x))

plt.plot(x, y)
plt.show()


# tanh Function
x = np.linspace(-10, 10, 21)
x_tf = tf.constant(x, dtype = tf.float32)

# tanh
y = activations.tanh(x_tf).numpy()

plt.axvline(-3, color = 'r')
plt.axvline(3, color = 'r')

# Softmax
# activations.softmax輸入須為2維資料。
x = np.random.uniform(1, 10, 40).reshape(10, 4)
print('Input: \n', x)
x_tf = tf.constant(x, dtype = tf.float32)

# Softmax
y = activations.softmax(x_tf).numpy()
print('Total: ', np.round(np.sum(y, axis = 1)))

# 使用NumPy計算Softmax
x = np.random.uniform(1, 10, 40)

# Softmax
y = np.e ** (x) / tf.reduce_sum(np.e ** (x))
print(sum(y))





