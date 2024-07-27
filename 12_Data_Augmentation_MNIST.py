import tensorflow as tf  # 載入 TensorFlow 庫

# Step 1. Load Data
mnist = tf.keras.datasets.mnist  # 從 Keras 中載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 分別載入訓練集和測試集

# Create CNN Model
from tensorflow.keras import layers     # 從 Keras 中載入層
import numpy as np                      # 載入 numpy 用於數據處理

input_shape = (28, 28, 1)               # 定義輸入圖像的形狀 (28x28 像素，1 個通道)
x_train = np.expand_dims(x_train, -1)   # 為訓練數據擴展維度，新增通道維度
x_test = np.expand_dims(x_test, -1)     # 為測試數據擴展維度，新增通道維度

# CNN Model
model = tf.keras.models.Sequential([                                # 建立一個 Sequential 模型
    tf.keras.Input(shape=input_shape),                              # 輸入層，指定輸入形狀
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),       # 卷積層，具有 32 個濾波器，3x3 的卷積核，ReLU 激活函數
    layers.MaxPooling2D(pool_size=(2, 2)),                          # 最大池化層，池化窗口大小為 2x2
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),       # 卷積層，具有 64 個濾波器，3x3 的卷積核，ReLU 激活函數
    layers.MaxPooling2D(pool_size=(2, 2)),                          # 最大池化層，池化窗口大小為 2x2
    layers.Flatten(),                                               # 將 3D 輸出展平為 1D
    layers.Dropout(0.2),                                            # Dropout 層，隨機丟棄 20% 的神經元以防止過擬合
    layers.Dense(10, activation="softmax"),                         # 全連接層，具有 10 個神經元，使用 Softmax 激活函數進行分類
])

# Set Optimizer
model.compile(
    optimizer='adam',                           # 使用 Adam 優化器
    loss='sparse_categorical_crossentropy',     # 使用稀疏類別交叉熵作為損失函數
    metrics=['accuracy']                        # 評估指標為準確率
)

# Data Augmentation
# Set Parameter
batch_size = 1000   # 設定批次大小
epochs = 5          # 設定訓練週期數

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,           # 將圖像數據正規化到 0 到 1 之間
    rotation_range=10,          # 隨機旋轉圖像，範圍為 10 度
    zoom_range=0.1,             # 隨機縮放圖像，縮放範圍為 10%
    width_shift_range=0.1,      # 隨機水平平移圖像，平移範圍為 10%
    height_shift_range=0.1      # 隨機垂直平移圖像，平移範圍為 10%
)

datagen.fit(x_train)                                                        # 擬合訓練數據
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),                  # 使用數據增強生成器進行訓練
    epochs = epochs,                                                        # 設定訓練週期數
    validation_data=datagen.flow(x_test, y_test, batch_size=batch_size),    # 使用數據增強生成器進行驗證
    verbose = 1,                                                            # 訓練過程中顯示進度
    steps_per_epoch=x_train.shape[0] // batch_size                          # 每個 epoch 的步數
)

# Score Model
score = model.evaluate(x_test, y_test, verbose=0)  # 在測試集上評估模型性能

# 輸出模型在測試集上的每個評估指標
for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')

# 使用小畫家，繪製 0~9，實際測試看看
from skimage import io                      # 從 skimage 載入 io 模塊用於讀取影像
from skimage.transform import resize        # 從 skimage 載入 resize 模塊用於調整影像大小
import numpy as np                          # 載入 numpy 用於數據處理

# 讀取影像並轉為單色
uploaded_file = './myDigits/9.png'                  # 指定待測試影像的路徑
image1 = io.imread(uploaded_file, as_gray=True)     # 讀取影像並轉為灰度圖像

# 縮為 (28, 28) 大小的影像
image_resized = resize(image1, (28, 28), anti_aliasing=True)    # 調整影像大小為 28x28 像素
X1 = image_resized.reshape(1, 28, 28, 1)                        # 將影像數據重塑為模型輸入形狀 (1, 28, 28, 1)

# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = np.abs(1 - X1)  # 將顏色反轉，因為 MNIST 圖像為白底黑字

# 預測
predictions = np.argmax(model.predict(X1), axis=-1)     # 使用模型進行預測，取預測結果中最大值的索引
print(predictions)                                      # 輸出預測結果
