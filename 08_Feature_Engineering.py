import tensorflow as tf  # 引入 TensorFlow 函式庫

# One-hot Encoding 示例
# 將數字 [0, 1, 2, 3] 編碼成一個具有 9 個類別的 one-hot 編碼矩陣
tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes = 9)

# 載入 MNIST 數據集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 下載並載入訓練和測試數據

# 特徵縮放，使用常態化(Normalization)
# 將像素值從 [0, 255] 範圍縮放到 [0, 1] 範圍
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# 將標籤轉換為 one-hot 編碼
# 將數字標籤轉換為對應的 one-hot 編碼形式
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 創建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),    # 將每個 28x28 的圖像展平為一維向量
    tf.keras.layers.Dense(256, activation = 'relu'),    # 添加一個全連接層，具有 256 個神經元，激活函數為 ReLU
    tf.keras.layers.Dropout(0.2),                       # 添加 Dropout 層以防止過擬合，丟棄率為 20%
    tf.keras.layers.Dense(10, activation = 'softmax')   # 添加輸出層，具有 10 個神經元（對應 10 個類別），激活函數為 Softmax
])

# 設定模型的優化器、損失函數和評估指標
model.compile(optimizer = 'adam',                       # 使用 Adam 優化器
              loss = 'categorical_crossentropy',        # 使用 categorical_crossentropy 損失函數（適用於多類別分類）
              metrics = ['accuracy'])                   # 設定評估指標為準確度

# 訓練模型
history = model.fit(x_train_norm, y_train, epochs = 5, validation_split = 0.2)  # 訓練模型 5 個 epoch，並使用 20% 的訓練數據進行驗證

# 評估模型
score = model.evaluate(x_test_norm, y_test, verbose = 0)                        # 在測試數據上評估模型，設定 verbose = 0 以禁止輸出詳細信息

# 輸出模型的評估結果
for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')                         # 打印每個評估指標的值，保留 4 位小數

import numpy as np                                  # 引入 NumPy 函式庫
import tensorflow as tf                             # 引入 TensorFlow 函式庫
from tensorflow.keras.layers import Normalization   # 從 Keras 模組中引入 Normalization 層

# 測試數據
data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7]])  # 創建一個包含三個樣本的測試數據

# 創建 Normalization 層
layer = Normalization()  # 初始化 Normalization 層

# 訓練 Normalization 層
layer.adapt(data)  # 使用測試數據訓練 Normalization 層，以計算數據的均值和標準差

# 標準化數據
normalized_data = layer(data)  # 對測試數據進行標準化處理

# 打印結果
print(f"平均數: {normalized_data.numpy().mean():.2f}")  # 打印標準化後數據的均值，保留 2 位小數
print(f"標準差: {normalized_data.numpy().std():.2f}")   # 打印標準化後數據的標準差，保留 2 位小數

print(normalized_data)  # 顯示標準化後的數據
