import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
import matplotlib.pyplot as plt
import math

# 二進位交叉熵損失函數
# 設定實際值和預測值
y_true = tf.constant([[0., 1.], [0., 0.]])              # 實際值，代表兩個樣本的標籤
y_pred = tf.constant([[0.6, 0.4], [0.4, 0.6]])          # 預測值，模型對兩個樣本的預測概率

# 計算 Binary Cross Entropy（BCELoss）
bce = tf.keras.losses.BinaryCrossentropy()                      # 創建 BinaryCrossentropy 損失函數實例
print(f'Binary Cross Entropy: {bce(y_true, y_pred).numpy()}')   # 計算並打印 BCELoss 值

# 二進位交叉熵的手動驗證
# 計算每個元素的交叉熵
manual_bce = ((0 - math.log(1 - 0.6) - math.log(0.4)) + (0 - math.log(1 - 0.6) - math.log(0.6))) / 4

# 計算公式：平均每個樣本的交叉熵損失
print(f'Manual Binary Cross Entropy: {manual_bce}')  # 打印手動計算的 BCELoss 值

# Sigmoid 函數
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))  # Sigmoid 函數公式：1 / (1 + exp(-x))

# 自定義的二進位交叉熵函數
def BCE(output, target):
    n = len(output)                         # 輸入數據的數量
    total_value = 0                         # 初始化總損失值

    output = list(map(sigmoid, output))     # 使用 sigmoid 函數轉換預測值
    print(output)                           # 打印轉換後的預測值

    for i in range(n):
        total_value += (target[i] * math.log(output[i]) + (1 - target[i]) * math.log(1 - output[i]))  # 計算每個樣本的損失
    total_value *= -1 / n                   # 計算平均損失並取負值
    return total_value                      # 返回自定義的 BCELoss 值

y_pred = [-1, -2, -3, 1, 2, 3]  # 自定義預測值
y_true = [0, 1, 0, 0, 0, 1]     # 自定義實際值

print(f'Custom Binary Cross Entropy: {BCE(y_pred, y_true)}')  # 打印自定義計算的 BCELoss 值

print("-" * 40)  # 打印分隔線

# 類別交叉熵損失函數
y_true = tf.constant([[0, 1, 0], [0, 0, 1]])                # 實際值，為 one-hot 編碼格式
y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])    # 預測值，模型對每個類別的預測概率

cce = tf.keras.losses.CategoricalCrossentropy()                     # 創建 CategoricalCrossentropy 損失函數實例
print(f'Categorical Cross Entropy: {cce(y_true, y_pred).numpy()}')  # 計算並打印 CategoricalCrossentropy 值

# 稀疏類別交叉熵損失函數
y_true = tf.constant([1, 2])                                # 實際值，為每個樣本的類別索引
y_pred = tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])    # 預測值，模型對每個類別的預測概率

scce = tf.keras.losses.SparseCategoricalCrossentropy()                      # 創建 SparseCategoricalCrossentropy 損失函數實例
print(f'Sparse Categorical Cross Entropy: {scce(y_true, y_pred).numpy()}')  # 計算並打印 SparseCategoricalCrossentropy 值

# 均方誤差 (Mean Squared Error) 損失函數
y_true = tf.constant([[0., 1.], [0., 0.]])  # 實際值
y_pred = tf.constant([[1., 1.], [1., 0.]])  # 預測值

mse = tf.keras.losses.MeanSquaredError()                        # 創建 MeanSquaredError 損失函數實例
print(f'Mean Squared Error: {mse(y_true, y_pred).numpy()}')     # 計算並打印 Mean Squared Error 值

# 樣本類別的權重比例
print(f'Mean Squared Error with sample weight: {mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()}')  
# 計算 Mean Squared Error，並考慮每個樣本的權重

# 取總和，即 SSE，而非 MSE
mse_sum = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)     # 創建 MeanSquaredError 損失函數實例，並設定 reduction 為 SUM
print(f'Sum of Squared Errors: {mse_sum(y_true, y_pred).numpy()}')                      # 計算並打印 SSE 值

# Hinge Loss
y_true = tf.constant([[0., 1.], [0., 0.]])  # 實際值
y_pred = tf.constant([[0.6, 0.4], [0.4, 0.6]])  # 預測值

hinge_loss = tf.keras.losses.Hinge()  # 創建 Hinge 損失函數實例
print(f'Hinge Loss: {hinge_loss(y_true, y_pred).numpy()}')  # 計算並打印 Hinge Loss 值

# 手動驗證 Hinge Loss
manual_hinge_loss = (max(1 - (-1) * 0.6, 0) + max(1 - 1 * 0.4, 0) +
                     max(1 - (-1) * 0.4, 0) + max(1 - (-1) * 0.6, 0)) / 4
# 計算公式：平均每個樣本的 Hinge Loss
print(f'Manual Hinge Loss: {manual_hinge_loss}')  # 打印手動計算的 Hinge Loss 值

# 自定義損失函數
model = tf.keras.models.Sequential([                            # 創建一個 Sequential 模型
    tf.keras.layers.Flatten(input_shape=(28, 28)),              # 第一層：將 28x28 的輸入展平為一維向量
    tf.keras.layers.Dense(128, activation='relu'),              # 第二層：全連接層，有 128 個神經元，激活函數為 ReLU
    tf.keras.layers.Dropout(0.2),                               # 第三層：Dropout 層，丟棄 20% 的神經元
    tf.keras.layers.Dense(10, activation='softmax')             # 第四層：全連接層，有 10 個神經元，激活函數為 Softmax
])

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)             # 計算預測值與實際值之間的平方差
    return tf.reduce_mean(squared_difference, axis=-1)          # 返回平方差的平均值作為損失函數值

model.compile(optimizer='adam', loss=my_loss_fn)                # 編譯模型，使用自定義損失函數

print("模型已編譯，使用自定義損失函數。")                        # 打印模型編譯完成的提示
