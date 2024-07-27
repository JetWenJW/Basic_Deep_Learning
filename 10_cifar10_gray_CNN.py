import tensorflow as tf             # 載入 TensorFlow 庫
import matplotlib.pyplot as plt     # 載入 Matplotlib 用於繪圖

# 載入 CIFAR-10 數據集
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 輸出數據集形狀 (訓練數據、測試數據、標籤)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 將 RGB 圖像轉換為灰度圖像
x_train = tf.image.rgb_to_grayscale(x_train)  # 將訓練數據從 RGB 轉換為灰度
x_test = tf.image.rgb_to_grayscale(x_test)    # 將測試數據從 RGB 轉換為灰度
# 輸出轉換後的數據形狀
print(x_train.shape, y_test.shape)

# 將數據類型轉換為 float32
x_train = tf.cast(x_train, tf.float32)  # 將訓練數據類型轉換為 float32
x_test = tf.cast(x_test, tf.float32)    # 將測試數據類型轉換為 float32

# 正規化數據，將像素值縮放到 0 到 1 之間
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 1)),       # 將 32x32x1 的灰度圖像展平為一維數據
    tf.keras.layers.Dense(128, activation='relu'),          # 全連接層，具有 128 個神經元，使用 ReLU 激活函數
    tf.keras.layers.Dropout(0.2),                           # Dropout 層，隨機丟棄 20% 的神經元以防止過擬合
    tf.keras.layers.Dense(10, activation='softmax')         # 輸出層，具有 10 個神經元，使用 Softmax 激活函數以進行分類
])

# 設置優化器和損失函數
model.compile(
    optimizer='adam',                                                       # 使用 Adam 優化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # 使用稀疏類別交叉熵作為損失函數
    metrics=['accuracy']                                                    # 評估指標為準確率
)

# 訓練模型
history = model.fit(x_train_norm, y_train, epochs=5, validation_split=0.2)  # 訓練模型，進行 5 次訓練週期，每個 epoch 使用 20% 的數據作為驗證集

# 評估模型
score = model.evaluate(x_test_norm, y_test, verbose=0)  # 在測試集上評估模型性能

# 輸出模型在測試集上的每個評估指標
for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')

# 繪製訓練過程中的準確度變化
plt.plot(history.history['accuracy'], label='accuracy')             # 繪製訓練準確度
plt.plot(history.history['val_accuracy'], label='val_accuracy')     # 繪製驗證準確度
plt.xlabel('Epoch')                                                 # 設置 x 軸標籤
plt.ylabel('Accuracy')                                              # 設置 y 軸標籤
plt.legend(loc='lower right')                                       # 顯示圖例
plt.show()                                                          # 顯示圖形

# 顯示模型摘要
model.summary()  # 輸出模型的詳細結構和參數數量
