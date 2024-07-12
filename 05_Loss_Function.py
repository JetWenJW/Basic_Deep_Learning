import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
import matplotlib.pyplot as plt
import math

# 二進位交叉熵損失函數
y_true = tf.constant([[0., 1.], [0., 0.]])  # 實際值
y_pred = tf.constant([[0.6, 0.4], [0.4, 0.6]])  # 預測值

# 計算 Binary Cross Entropy
bce = tf.keras.losses.BinaryCrossentropy()
print(f'Binary Cross Entropy: {bce(y_true, y_pred).numpy()}')

# 二進位交叉熵的手動驗證
manual_bce = ((0 - math.log(1 - 0.6) - math.log(0.4)) + (0 - math.log(1 - 0.6) - math.log(0.6))) / 4
print(f'Manual Binary Cross Entropy: {manual_bce}')

# Sigmoid 函數
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

# 自定義的二進位交叉熵函數
def BCE(output, target):
    n = len(output)
    total_value = 0

    output = list(map(sigmoid, output))
    print(output)

    for i in range(n):
        total_value += (target[i] * math.log(output[i]) + (1 - target[i]) * math.log(1 - output[i]))
    total_value *= -1 / n
    return total_value

y_pred = [-1, -2, -3, 1, 2, 3]
y_true = [0, 1, 0, 0, 0, 1]

print(f'Custom Binary Cross Entropy: {BCE(y_pred, y_true)}')

print("-" * 40)

# 類別交叉熵損失函數
y_true = tf.constant([[0, 1, 0], [0, 0, 1]])  # 實際值
y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])  # 預測值

cce = tf.keras.losses.CategoricalCrossentropy()
print(f'Categorical Cross Entropy: {cce(y_true, y_pred).numpy()}')

# 稀疏類別交叉熵損失函數
y_true = tf.constant([1, 2])  # 實際值
y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])  # 預測值

scce = tf.keras.losses.SparseCategoricalCrossentropy()
print(f'Sparse Categorical Cross Entropy: {scce(y_true, y_pred).numpy()}')

# 均方誤差 (Mean Squared Error) 損失函數
y_true = tf.constant([[0., 1.], [0., 0.]])  # 實際值
y_pred = tf.constant([[1., 1.], [1., 0.]])  # 預測值

mse = tf.keras.losses.MeanSquaredError()
print(f'Mean Squared Error: {mse(y_true, y_pred).numpy()}')

# 樣本類別的權重比例
print(f'Mean Squared Error with sample weight: {mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()}')

# 取總和，即 SSE，而非 MSE
mse_sum = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
print(f'Sum of Squared Errors: {mse_sum(y_true, y_pred).numpy()}')

# Hinge Loss
y_true = tf.constant([[0., 1.], [0., 0.]])
y_pred = tf.constant([[0.6, 0.4], [0.4, 0.6]])

hinge_loss = tf.keras.losses.Hinge()
print(f'Hinge Loss: {hinge_loss(y_true, y_pred).numpy()}')

# 手動驗證 Hinge Loss
manual_hinge_loss = (max(1 - (-1) * 0.6, 0) + max(1 - 1 * 0.4, 0) +
                     max(1 - (-1) * 0.4, 0) + max(1 - (-1) * 0.6, 0)) / 4
print(f'Manual Hinge Loss: {manual_hinge_loss}')

# 自定義損失函數
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

model.compile(optimizer='adam', loss=my_loss_fn)

print("模型已編譯，使用自定義損失函數。")
