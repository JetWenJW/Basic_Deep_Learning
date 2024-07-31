import tensorflow as tf
from tensorflow.keras import layers

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),                          # 第一層：將輸入的28x28圖像展平成一維向量
    tf.keras.layers.Dense(128, activation = 'relu', name="layer1"),           # 第二層：全連接層，128個神經元，激活函數為ReLU，命名為layer1
    tf.keras.layers.Dropout(0.2),                                             # 第三層：Dropout層，丟棄20%的神經元
    tf.keras.layers.Dense(10, activation = 'softmax', name="layer2")          # 第四層：全連接層，10個神經元，激活函數為softmax，命名為layer2
])

# 設定優化器
model.compile(optimizer = 'adam',                         # 使用Adam優化器
              loss = 'sparse_categorical_crossentropy',   # 損失函數為sparse_categorical_crossentropy
              metrics = ['accuracy']                      # 評估指標為準確率
              )

# 顯示模型摘要
model.summary()  # 打印模型摘要

# 計算第一層 Dense 層的參數個數
# 設定模型的輸入和輸出
feature_extractor = tf.keras.Model(
    inputs=model.inputs,                            # 設定輸入
    outputs=model.get_layer(name="layer1").output,  # 設定輸出為layer1的輸出
)

# 呼叫 feature_extractor 獲取輸出
x = tf.ones((1, 28, 28))            # 定義一個形狀為1x28x28的全1張量作為輸入
features = feature_extractor(x)     # 獲取特徵
print(features.shape)               # 顯示特徵形狀

# 計算第一層 Dense 層的參數個數
parameter_count = (28 * 28) * features.shape[1] + features.shape[1]     # 計算參數個數：輸入神經元數量 * 輸出神經元數量 + 偏置
print(f'參數(parameter)個數： {parameter_count}')                       # 打印參數個數
print("-" * 20)  # 打印分隔線

# 計算第二層 Dense 層的參數個數
# 設定模型的輸入和輸出
feature_extractor = tf.keras.Model(
    inputs=model.inputs,                            # 設定輸入
    outputs=model.get_layer(name="layer2").output,  # 設定輸出為layer2的輸出
)

# 呼叫 feature_extractor 獲取輸出
x = tf.ones((1, 28, 28))            # 定義一個形狀為1x28x28的全1張量作為輸入
features = feature_extractor(x)     # 獲取特徵
print(features.shape)               # 顯示特徵形狀

# 計算第二層 Dense 層的參數個數
parameter_count = (128) * features.shape[1] + features.shape[1]     # 計算參數個數：輸入神經元數量 * 輸出神經元數量 + 偏置
print(f'參數(parameter)個數： {parameter_count}')                   # 打印參數個數
print("-" * 20)  # 打印分隔線
