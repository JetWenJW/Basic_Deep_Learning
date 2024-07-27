import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 隨機梯度下降法 (Stochastic Gradient Descent, SGD)
opt_sgd = tf.keras.optimizers.SGD(learning_rate=0.1)    # 創建 SGD 優化器實例，學習率設定為 0.1
var = tf.Variable(1.0)                                  # 定義一個 TensorFlow 變數，初始值為 1.0

# 定義損失函數 (loss function)
loss_fn = lambda: (var ** 2) / 2.0  # 定義損失函數：(變數平方) / 2

# 使用 SGD 進行優化
for i in range(51):                                         # 循環 51 次，進行 51 步優化
    with tf.GradientTape() as tape:                         # 使用 GradientTape 來記錄梯度計算過程
        loss_value = loss_fn()                              # 計算損失值
    grads = tape.gradient(loss_value, [var])                # 計算損失對變數的梯度
    opt_sgd.apply_gradients(zip(grads, [var]))              # 將梯度應用到變數上進行更新
    if i % 10 == 0 and i > 0:                               # 每 10 步打印一次變數值
        print(f'優化的步驟: {i}, 變數: {var.numpy()}')      # 打印當前優化步驟和變數值

# 優化三次測試隨機梯度下降法的動能
opt_sgd_momentum = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)     # 創建 SGD 優化器，加入動能項，學習率為 0.1，動能為 0.9
var = tf.Variable(1.0)                                                          # 重新定義一個 TensorFlow 變數，初始值為 1.0

val0 = var.value()          # 獲取初始變數值
print(f'val0: {val0}')      # 打印初始變數值

# 定義損失函數
loss_fn = lambda: (var ** 2) / 2.0  # 定義損失函數：(變數平方) / 2

# 第一次優化
with tf.GradientTape() as tape:                                 # 使用 GradientTape 來記錄梯度計算過程
    loss_value = loss_fn()                                      # 計算損失值
grads = tape.gradient(loss_value, [var])                        # 計算損失對變數的梯度
opt_sgd_momentum.apply_gradients(zip(grads, [var]))             # 將梯度應用到變數上進行更新
val1 = var.value()                                              # 獲取更新後的變數值
print(f'優化的步驟: 1, val1: {val1}, 變化值: {val0 - val1}')    # 打印步驟、變數值及變化值

# 第二次優化
with tf.GradientTape() as tape:                                     # 使用 GradientTape 來記錄梯度計算過程
    loss_value = loss_fn()                                          # 計算損失值
grads = tape.gradient(loss_value, [var])                            # 計算損失對變數的梯度
opt_sgd_momentum.apply_gradients(zip(grads, [var]))                 # 將梯度應用到變數上進行更新
val2 = var.value()                                                  # 獲取更新後的變數值
print(f'優化的步驟: 2, val2: {val2}, 變化值: {val1 - val2}')        # 打印步驟、變數值及變化值

# 第三次優化
with tf.GradientTape() as tape:                                 # 使用 GradientTape 來記錄梯度計算過程
    loss_value = loss_fn()                                      # 計算損失值
grads = tape.gradient(loss_value, [var])                        # 計算損失對變數的梯度
opt_sgd_momentum.apply_gradients(zip(grads, [var]))             # 將梯度應用到變數上進行更新
val3 = var.value()                                              # 獲取更新後的變數值
print(f'優化的步驟: 3, val3: {val3}, 變化值: {val2 - val3}')    # 打印步驟、變數值及變化值

# 使用 Adam 優化器
opt_adam = tf.keras.optimizers.Adam(learning_rate=0.1)          # 創建 Adam 優化器，學習率設定為 0.1
var = tf.Variable(1.0)                                          # 重新定義一個 TensorFlow 變數，初始值為 1.0
loss_fn = lambda: (var ** 2) / 2.0                              # 定義損失函數：(變數平方) / 2

# 使用 Adam 進行優化
for i in range(11):                                             # 循環 11 次，進行 11 步優化
    with tf.GradientTape() as tape:                             # 使用 GradientTape 來記錄梯度計算過程
        loss_value = loss_fn()                                  # 計算損失值
    grads = tape.gradient(loss_value, [var])                    # 計算損失對變數的梯度
    opt_adam.apply_gradients(zip(grads, [var]))                 # 將梯度應用到變數上進行更新
    if i % 2 == 0 and i > 0:                                    # 每 2 步打印一次變數值
        print(f'Adam, 優化的步驟: {i}, 變數: {var.numpy()}')          # 打印當前優化步驟和變數值
