# One Hot Encodeing
import tensorflow as tf

# One-hot Encodeing
tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes = 9)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# Setting Optimizer, Loss, Metrics
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Model Trainning
history = model.fit(x_train_norm, y_train, epochs = 5, validation_split = 0.2)

# Score Model
score = model.evaluate(x_test_norm, y_test, verbose = 0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization

# 測試數據
data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7]])

# 創建 Normalization 層
layer = Normalization()

# 訓練Normalization層
layer.adapt(data)

# 標準化數據
normalized_data = layer(data)

# 打印結果
print(f"平均數: {normalized_data.numpy().mean():.2f}")
print(f"標準差: {normalized_data.numpy().std():.2f}")

normalized_data
















