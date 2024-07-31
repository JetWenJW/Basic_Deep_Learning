import tensorflow as tf  # 載入 TensorFlow 庫

mnist = tf.keras.datasets.mnist  # 載入 MNIST 數據集

# 載入 MNIST 手寫阿拉伯數字資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 特徵縮放，將像素值歸一化到 0 到 1 之間
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # 將 28x28 的圖像展平為一維數據
    tf.keras.layers.Dense(256, activation='relu'),      # 全連接層，具有 256 個神經元，使用 ReLU 激活函數
    tf.keras.layers.Dropout(0.2),                       # Dropout 層，隨機丟棄 20% 的神經元以防止過擬合
    tf.keras.layers.Dense(10, activation='softmax')     # 輸出層，具有 10 個神經元，使用 Softmax 激活函數以進行分類
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',                         # 使用 Adam 優化器
              loss='sparse_categorical_crossentropy',   # 使用稀疏類別交叉熵作為損失函數
              metrics=['accuracy'])                     # 評估指標為準確率

# 模型訓練，進行 5 次訓練週期，每個 epoch 使用 20% 的數據作為驗證集
history = model.fit(x_train_norm, y_train, epochs = 5, validation_split = 0.2)

# 評分(Score Model)，在測試集上評估模型性能
score = model.evaluate(x_test_norm, y_test, verbose = 0)

# 輸出模型在測試集上的每個評估指標
for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')

print("-" * 10, "Train Done~", "-" * 10)

# 儲存模型為 .keras 格式
model.save('my_model.keras')
print("-" * 10, "Save my_model.keras Done~", "-" * 10)

# 載入模型
model2 = tf.keras.models.load_model('my_model.keras')

# 評分(Score Model)，在測試集上評估載入後的模型性能
score = model2.evaluate(x_test_norm, y_test, verbose = 0)

# 輸出載入後模型在測試集上的每個評估指標
for i, x in enumerate(score):
    print(f'{model2.metrics_names[i]}: {score[i]:.4f}')

# 模型比較，檢查兩個模型預測結果是否一致
import numpy as np

# 比較兩個模型的預測結果
np.testing.assert_allclose(
    model.predict(x_test_norm), model2.predict(x_test_norm)
)

# 儲存模型為 .h5 格式
model.save('my_model.h5')
print("-" * 10, "Save my_model.h5 Done~", "-" * 10)

# 載入模型
model3 = tf.keras.models.load_model('my_model.h5')

# 取得模型結構
config = model.get_config()

# 使用從模型結構創建新的 Sequential 模型
new_model = tf.keras.Sequential.from_config(config)

# 儲存模型結構為 JSON 格式
json_config = model.to_json()

# 從 JSON 格式的模型結構創建新的模型
new_model = tf.keras.models.model_from_json(json_config)

# 取得模型權重
weights = model.get_weights()
print(weights)

# 設定新的模型權重
new_model.set_weights(weights)

# 設定優化器、損失函數和效能衡量指標
new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 評估新的模型性能
score = new_model.evaluate(x_test_norm, y_test, verbose=0)
print(score)
print("-" * 10, "Json Done~", "-" * 10)

# 使用 Keras 註冊自定義層，以便保存和載入模型時能識別這個層
@tf.keras.utils.register_keras_serializable()
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    # 建立層的權重
    def build(self, input_shape):
        self.w = self.add_weight(
            shape = (input_shape[-1], self.units),
            initializer = "random_normal",
            trainable = True,
        )
        self.b = self.add_weight(
            shape = (self.units,), 
            initializer = "random_normal", 
            trainable = True
        )

    # 定義層的前向計算過程
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    # 取得層的配置
    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config

# 自定義激活函數
def custom_activation(x):
    return tf.nn.tanh(x) ** 2

# 使用 Custom Layer 和 Custom Activation 建立模型
inputs = tf.keras.Input((32,))                              # 定義輸入層，具有 32 個特徵
x = CustomLayer(32)(inputs)                                 # 添加 CustomLayer 層，輸出 32 個單元
outputs = tf.keras.layers.Activation(custom_activation)(x)  # 添加自定義激活函數
model = tf.keras.Model(inputs, outputs)                     # 創建模型

# 取得模型結構
config = model.get_config()

# 使用註冊的 Custom Layer 創建新模型
custom_objects = {"CustomLayer": CustomLayer, "custom_activation": custom_activation}
with tf.keras.utils.custom_object_scope(custom_objects):    # 在 custom_object_scope 範圍內
    new_model = tf.keras.Model.from_config(config)          # 根據配置創建新模型

# 模型權重存檔
model.save_weights('my_h5_model.weights.h5')                # 將模型權重保存到文件中

# 載入模型權重
model.load_weights('my_h5_model.weights.h5')                # 從文件中載入模型權重
