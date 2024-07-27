import tensorflow as tf  # 載入 TensorFlow 庫

# Step 1. Load Data
cifar10 = tf.keras.datasets.cifar10  # 從 Keras 中載入 CIFAR-10 數據集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 分別載入訓練集和測試集

# 輸出數據集的形狀 (訓練數據、訓練標籤、測試數據、測試標籤)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Step 2. Data Clean (There is no need to do so)
# 此步驟在此數據集上不需要進行數據清理

# Step 3. Feature Engineering (Feature as between 0 ~ 1)
x_train_norm, x_test_norm = x_train / 255, x_test / 255     # 將圖像數據正規化到 0 到 1 之間
x_train_norm[0]                                             # 顯示第一張訓練圖像的正規化數據

# Step 4. Data Split (Train & Validation)
# 此步驟在此數據集上不需要進行額外的數據分割，因為使用了 `validation_split` 參數

# Step 5. Create Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),     # 卷積層，具有 32 個濾波器，3x3 的卷積核，ReLU 激活函數，輸入圖像大小為 32x32x3
    tf.keras.layers.MaxPooling2D((2, 2)),                                               # 最大池化層，池化窗口大小為 2x2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),                              # 卷積層，具有 64 個濾波器，3x3 的卷積核，ReLU 激活函數
    tf.keras.layers.MaxPooling2D((2, 2)),                                               # 最大池化層，池化窗口大小為 2x2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),                              # 卷積層，具有 64 個濾波器，3x3 的卷積核，ReLU 激活函數
    tf.keras.layers.Flatten(),                                                          # 將 3D 輸出展平為 1D
    tf.keras.layers.Dense(64, activation='relu'),                                       # 全連接層，具有 64 個神經元，ReLU 激活函數
    tf.keras.layers.Dense(10, activation = 'softmax')                                   # 輸出層，具有 10 個神經元，用於 10 個類別的預測
])

# Step 6. Set Optimizer
model.compile(
    optimizer='adam',                                                       # 使用 Adam 優化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # 使用稀疏類別交叉熵作為損失函數，`from_logits=True` 表示輸出層的值是 logits
    metrics=['accuracy']                                                    # 評估指標為準確率
)

# Train The Model
history = model.fit(x_train_norm, y_train, epochs=10, validation_split=0.2)     # 訓練模型，進行 10 次訓練週期，每個 epoch 使用 20% 的數據作為驗證集

# Step 7. Score Model
score = model.evaluate(x_test_norm, y_test, verbose=0)                          # 在測試集上評估模型性能

# 輸出模型在測試集上的每個評估指標
for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')

# 繪製訓練過程中的準確度變化
import matplotlib.pyplot as plt  # 載入 Matplotlib 用於繪圖

plt.plot(history.history['accuracy'], label='accuracy')             # 繪製訓練準確度
plt.plot(history.history['val_accuracy'], label='val_accuracy')     # 繪製驗證準確度
plt.xlabel('Epoch')                                                 # 設置 x 軸標籤
plt.ylabel('Accuracy')                                              # 設置 y 軸標籤
plt.ylim([0.5, 1])                                                  # 設置 y 軸範圍
plt.legend(loc='lower right')                                       # 顯示圖例
plt.show()                                                          # 顯示圖形

# 顯示模型摘要
model.summary()  # 輸出模型的詳細結構和參數數量
