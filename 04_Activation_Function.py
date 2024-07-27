import tensorflow as tf
from tensorflow.keras import activations
import numpy as np
import matplotlib.pyplot as plt

# ReLU(Rectified linear unit)
# 設定 x 的範圍為 -10 到 10，總共 21 個點
x = np.linspace(-10, 10, 21)
x_tf = tf.constant(x, dtype=tf.float32)     # 將 x 轉換為 TensorFlow 的常數張量

# ReLU 激活函數
y = activations.relu(x_tf).numpy()          # 使用 ReLU 激活函數並轉換為 numpy 格式

# 繪製 ReLU 函數圖形
plt.plot(x, y)  # 繪製 x 和 y 的圖形
plt.show()      # 顯示圖形
print('-' * 20, "Part 1", '-' * 20)

# ReLU 參數變換
x = np.linspace(-10, 10, 21)                        # 設定 x 的範圍為 -10 到 10，總共 21 個點
x_tf = tf.constant(x, dtype = tf.float32)           # 將 x 轉換為 TensorFlow 的常數張量
y = activations.relu(x_tf, threshold = 5).numpy()   # 使用 ReLU 激活函數並設定 threshold 為 5

plt.plot(x, y)  # 繪製 x 和 y 的圖形
plt.show()      # 顯示圖形
print('-' * 20, "Part 2", '-' * 20)

x = np.linspace(-10, 10, 21)                        # 設定 x 的範圍為 -10 到 10，總共 21 個點
x_tf = tf.constant(x, dtype = tf.float32)           # 將 x 轉換為 TensorFlow 的常數張量
y = activations.relu(x_tf, max_value = 5).numpy()   # 使用 ReLU 激活函數並設定 max_value 為 5

plt.plot(x, y)  # 繪製 x 和 y 的圖形
plt.show()      # 顯示圖形
print('-' * 20, "Part 3", '-' * 20)

x = np.linspace(-10, 10, 21)                                # 設定 x 的範圍為 -10 到 10，總共 21 個點
x_tf = tf.constant(x, dtype=tf.float32)                     # 將 x 轉換為 TensorFlow 的常數張量
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.5)           # 使用 Leaky ReLU 激活函數並設定 alpha 為 0.5
y = leaky_relu(x_tf).numpy()                                # 將 x 傳入 Leaky ReLU 函數並轉換為 numpy 格式
print(y)                                # 打印 y 的值
plt.plot(x, y)                          # 繪製 x 和 y 的圖形
plt.show()                              # 顯示圖形
print('-' * 20, "Part 4", '-' * 20)

# Sigmoid 激活函數
x = np.linspace(-10, 10, 21)                    # 設定 x 的範圍為 -10 到 10，總共 21 個點
x_tf = tf.constant(x, dtype=tf.float32)         # 將 x 轉換為 TensorFlow 的常數張量
y = activations.sigmoid(x_tf).numpy()           # 使用 Sigmoid 激活函數並轉換為 numpy 格式

plt.axvline(-4, color='r')                      # 在 x = -4 處添加一條紅色垂直線
plt.axvline(4, color='r')                       # 在 x = 4 處添加一條紅色垂直線

plt.plot(x, y)  # 繪製 x 和 y 的圖形
plt.show()      # 顯示圖形
print('-' * 20, "Part 5", '-' * 20)

# 使用 NumPy 計算 Sigmoid
x = np.linspace(-10, 10, 21)    # 設定 x 的範圍為 -10 到 10，總共 21 個點
y = 1 / (1 + np.e ** (-x))      # 使用 NumPy 計算 Sigmoid 函數

plt.plot(x, y)  # 繪製 x 和 y 的圖形
plt.show()      # 顯示圖形
print('-' * 20, "Part 6", '-' * 20)

# tanh 函數
x = np.linspace(-10, 10, 21)                # 設定 x 的範圍為 -10 到 10，總共 21 個點
x_tf = tf.constant(x, dtype = tf.float32)   # 將 x 轉換為 TensorFlow 的常數張量

# 使用 tanh 激活函數
y = activations.tanh(x_tf).numpy()          # 使用 tanh 激活函數並轉換為 numpy 格式

plt.axvline(-3, color = 'r')      # 在 x = -3 處添加一條紅色垂直線
plt.axvline(3, color = 'r')       # 在 x = 3 處添加一條紅色垂直線

# 繪製 tanh 函數圖形
plt.plot(x, y)      # 繪製 x 和 y 的圖形
plt.show()          # 顯示圖形
print('-' * 20, "Part 7", '-' * 20)

# Softmax 激活函數
# activations.softmax 的輸入須為 2 維資料
x = np.random.uniform(1, 10, 40).reshape(10, 4)     # 生成隨機數組並重塑為 10x4 的形狀
print('Input: \n', x)                               # 打印輸入數據
x_tf = tf.constant(x, dtype = tf.float32)           # 將 x 轉換為 TensorFlow 的常數張量

# 使用 Softmax 激活函數
y = activations.softmax(x_tf).numpy()               # 使用 Softmax 激活函數並轉換為 numpy 格式
print('Total: ', np.round(np.sum(y, axis = 1)))     # 打印每行 Softmax 後的總和，應該接近 1

# 使用 NumPy 計算 Softmax
x = np.random.uniform(1, 10, 40)  # 生成隨機數組

# Softmax 計算
y = np.e ** (x) / np.sum(np.e ** (x))   # 使用 NumPy 計算 Softmax
print(sum(y))                           # 打印 Softmax 後的總和，應該接近 1
print('-' * 20, "Part 8", '-' * 20)
