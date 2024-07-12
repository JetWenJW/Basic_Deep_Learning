import tensorflow as tf

mnist = tf.keras.datasets.mnist

# 載入 MNIST 手寫阿拉伯數字資料
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型訓練
history = model.fit(x_train_norm, y_train, epochs=5, validation_split=0.2)

# 評分(Score Model)
score=model.evaluate(x_test_norm, y_test, verbose=0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]:.4f}')


print("-" * 10, "Train Done~", "-" * 10)

# Save Model
model.save('my_model.keras')
print("-" * 10, "Save my_model.keras Done~", "-" * 10)

# Load Model
model2 = tf.keras.models.load_model('my_model.keras')

# Score Model
score = model2.evaluate(x_test_norm, y_test, verbose=0)

for i, x in enumerate(score):
    print(f'{model2.metrics_names[i]}: {score[i]:.4f}')

# 模型比較
import numpy as np

# 比較，若結果不同，會出現錯誤
np.testing.assert_allclose(
    model.predict(x_test_norm), model2.predict(x_test_norm)
)


# Save(Keras h5)
model.save('my_model.h5')
print("-" * 10, "Save my_model.h5 Done~", "-" * 10)

# Load Model
model3 = tf.keras.models.load_model('my_model.h5')

# 取得模型結構
config = model.get_config()
# Load Sequential Model
new_model = tf.keras.Sequential.from_config(config)

# Save(json Format)
json_config = model.to_json()

# Load Model
new_model = tf.keras.models.model_from_json(json_config)

# 取得模型權重
weights = model.get_weights()
weights

# Setting 模型權重
new_model.set_weights(weights)

# Set Optimizer
new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
# Predict
score = new_model.evaluate(x_test_norm, y_test, verbose=0)
score


# 取得模型結構時，Custom Layer 需註冊
@tf.keras.utils.register_keras_serializable()
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config
    
def custom_activation(x):
    return tf.nn.tanh(x) ** 2


# Make a Modle with Custom_Layer & Custom_Activation
inputs = tf.keras.Input((32,))
x = CustomLayer(32)(inputs)
outputs = tf.keras.layers.Activation(custom_activation)(x)
model = tf.keras.Model(inputs, outputs)

# Retrieve The Config
config = model.get_config()

# Custom Layer need to be registed
custom_objects = {"CustomLayer": CustomLayer, "custom_activation": custom_activation}
with tf.keras.utils.custom_object_scope(custom_objects):
    new_model = tf.keras.Model.from_config(config)


# 模型權重存檔，有 Custom Layer 會出現錯誤
model.save_weights('my_h5_model.weights.h5')

# 載入模型權重檔
model.load_weights('my_h5_model.weights.h5')
