import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 隨機梯度下降法 (Stochastic Gradient Descent, SGD)
opt_sgd = tf.keras.optimizers.SGD(learning_rate=0.1)  # 設定學習率為 0.1 的 SGD 優化器
var = tf.Variable(1.0)  # 定義初始變數

# 定義損失函數 (loss function)
loss_fn = lambda: (var ** 2) / 2.0

# 使用 SGD 進行優化
for i in range(51):
    with tf.GradientTape() as tape:  # 使用 GradientTape 來記錄計算過程
        loss_value = loss_fn()  # 計算損失值
    grads = tape.gradient(loss_value, [var])  # 計算損失相對於變數的梯度
    opt_sgd.apply_gradients(zip(grads, [var]))  # 將計算出的梯度應用到變數上
    if i % 10 == 0 and i > 0:
        print(f'優化的步驟: {i}, 變數: {var.numpy()}')

# 優化三次測試隨機梯度下降法的動能
opt_sgd_momentum = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)  # 設定學習率為 0.1 並使用動能為 0.9 的 SGD 優化器
var = tf.Variable(1.0)  # 重新定義初始變數

val0 = var.value()
print(f'val0: {val0}')

# 定義損失函數
loss_fn = lambda: (var ** 2) / 2.0

# 第一次優化
with tf.GradientTape() as tape:  # 使用 GradientTape 來記錄計算過程
    loss_value = loss_fn()  # 計算損失值
grads = tape.gradient(loss_value, [var])  # 計算損失相對於變數的梯度
opt_sgd_momentum.apply_gradients(zip(grads, [var]))  # 將計算出的梯度應用到變數上
val1 = var.value()
print(f'優化的步驟: 1, val1: {val1}, 變化值: {val0 - val1}')

# 第二次優化
with tf.GradientTape() as tape:
    loss_value = loss_fn()
grads = tape.gradient(loss_value, [var])
opt_sgd_momentum.apply_gradients(zip(grads, [var]))
val2 = var.value()
print(f'優化的步驟: 2, val2: {val2}, 變化值: {val1 - val2}')

# 第三次優化
with tf.GradientTape() as tape:
    loss_value = loss_fn()
grads = tape.gradient(loss_value, [var])
opt_sgd_momentum.apply_gradients(zip(grads, [var]))
val3 = var.value()
print(f'優化的步驟: 3, val3: {val3}, 變化值: {val2 - val3}')

# 使用 Adam 優化器
opt_adam = tf.keras.optimizers.Adam(learning_rate=0.1)  # 設定學習率為 0.1 的 Adam 優化器
var = tf.Variable(1.0)  # 重新定義初始變數
loss_fn = lambda: (var ** 2) / 2.0  # 定義損失函數

# 使用 Adam 進行優化
for i in range(11):
    with tf.GradientTape() as tape:
        loss_value = loss_fn()
    grads = tape.gradient(loss_value, [var])
    opt_adam.apply_gradients(zip(grads, [var]))
    if i % 2 == 0 and i > 0:
        print(f'優化的步驟: {i}, 變數: {var.numpy()}')
