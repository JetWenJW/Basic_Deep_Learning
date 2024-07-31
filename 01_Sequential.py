import tensorflow as tf

# 建立模型

# 建立模型語法1: 使用Sequential API建立模型，並設置各層
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),            # 將輸入的28x28圖像展平成一維向量
    tf.keras.layers.Dense(128, activation = 'relu'),            # 全連接層，128個神經元，激活函數為ReLU
    tf.keras.layers.Dropout(0.2),                               # 隨機丟棄20%的神經元，防止過擬合
    tf.keras.layers.Dense(10, activation = 'softmax')           # 輸出層，10個神經元，激活函數為softmax（適用於分類）
])

# 建立模型語法2: 使用Sequential API建立模型，並設置各層（將input_shape拿掉，以model參數設定）
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),                                  # 將輸入展平成一維向量
    tf.keras.layers.Dense(128, activation = 'relu'),            # 全連接層，128個神經元，激活函數為ReLU
    tf.keras.layers.Dropout(0.2),                               # 隨機丟棄20%的神經元，防止過擬合
    tf.keras.layers.Dense(10, activation = 'softmax')           # 輸出層，10個神經元，激活函數為softmax（適用於分類）
])
x = tf.keras.layers.Input(shape = (28, 28))     # 定義輸入層，形狀為28x28
y = model(x)                                    # 使用模型來處理輸入


print("-" * 20, "Part 1", '-' * 20)


# 建立模型語法3: 可以直接串連神經層
layer1 = tf.keras.layers.Dense(2, activation = 'relu', name = "layer1")     # 全連接層，2個神經元，激活函數為ReLU
layer2 = tf.keras.layers.Dense(3, activation = 'relu', name = "layer2")     # 全連接層，3個神經元，激活函數為ReLU
layer3 = tf.keras.layers.Dense(4, name = "layer3")                          # 全連接層，4個神經元，無激活函數

# 呼叫神經層處理測試輸入
x = tf.ones((3, 3))             # 定義一個3x3的全1矩陣作為輸入
y = layer3(layer2(layer1(x)))   # 串連神經層處理輸入
print("-" * 20, "Part 2", '-' * 20)


# 臨時加減神經層
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),        # 將輸入展平成一維向量
    tf.keras.layers.Dense(128, activation = 'relu'),        # 全連接層，128個神經元，激活函數為ReLU
    tf.keras.layers.Dropout(0.2),                           # 隨機丟棄20%的神經元，防止過擬合
    tf.keras.layers.Dense(10, activation = 'softmax')       # 輸出層，10個神經元，激活函數為softmax（適用於分類）
])

# 刪除一個神經層
model.pop()                             # 刪除模型中的最後一層
print(f'神經層數{len(model.layers)}')   # 打印模型中剩餘的神經層數
print(model.layers)                     # 顯示模型中的神經層

print("-" * 20, "Part 3", '-' * 20)

# 添加一個神經層
model.add(tf.keras.layers.Dense(10))    # 向模型中添加一個全連接層，10個神經元
print(f'神經層數{len(model.layers)}')   # 打印模型中更新後的神經層數
model.layers                            # 顯示模型中的神經層
print("-" * 20, "Part 4", '-' * 20)


# 獲取模型及神經層的資訊
layer1 = tf.keras.layers.Dense(2, activation = "relu", name = "layer1", input_shape = (28, 28))     # 定義第一個全連接層
layer2 = tf.keras.layers.Dense(3, activation = "relu", name = "layer2")                             # 定義第二個全連接層
layer3 = tf.keras.layers.Dense(4, name = "layer3")                                                  # 定義第三個全連接層

# 創建模型
model = tf.keras.models.Sequential([
    layer1, 
    layer2, 
    layer3
])

# 讀取模型權重
print(f'神經層參數類別總數:{len(model.weights)}')        # 打印模型中參數的總數
print(model.weights)                                    # 顯示模型中所有權重
print(f'{layer2.name}: {layer2.weights}')               # 打印第二個神經層的權重

# 獲取模型摘要
model.summary()  # 打印模型摘要
print("-" * 20, "Part 5", '-' * 20)

# 添加神經層並顯示模型摘要，方便調試
from tensorflow.keras import layers
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape = (250, 250, 3)))                    # 添加輸入層，形狀為250x250的RGB圖像
model.add(layers.Conv2D(32, 5, strides = 2, activation = "relu"))   # 添加卷積層，32個過濾器，大小為5x5，步幅為2，激活函數為ReLU
model.add(layers.Conv2D(32, 3, activation = "relu"))                # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
model.add(layers.MaxPooling2D(3))                                   # 添加最大池化層，大小為3x3

# 顯示模型摘要
model.summary()  # 打印模型摘要
print("-" * 20, "Part 6", '-' * 20)

model.add(layers.Conv2D(32, 3, activation = "relu"))            # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
model.add(layers.Conv2D(32, 3, activation = "relu"))            # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
model.add(layers.MaxPooling2D(3))                               # 添加最大池化層，大小為3x3
model.add(layers.Conv2D(32, 3, activation = "relu"))            # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
model.add(layers.Conv2D(32, 3, activation = "relu"))            # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
model.add(layers.MaxPooling2D(pool_size = (2, 2)))              # 添加最大池化層，大小為2x2

model.summary()  # 打印模型摘要
print("-" * 20, "Part 7", '-' * 20)

# GlobalMaxPooling2D都會將其縮減為單個值。
# 這通常用於卷積網絡的末端，
# 將特徵圖轉換為一維向量，以便進行分類或其他任務。
model.add(layers.GlobalMaxPooling2D())  # 添加全局最大池化層
model.add(layers.Dense(10))             # 添加全連接層，10個神經元
model.summary()  # 打印模型摘要
print("-" * 20, "Part 8", '-' * 20)



# 設置模型
initial_model = tf.keras.Sequential([
    tf.keras.Input(shape = (250, 250, 3)),                      # 定義輸入層，形狀為250x250的RGB圖像
    layers.Conv2D(32, 5, strides = 2, activation = "relu"),     # 添加卷積層，32個過濾器，大小為5x5，步幅為2，激活函數為ReLU
    layers.Conv2D(32, 3, activation = "relu"),                  # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
    layers.Conv2D(32, 3, activation = "relu"),                  # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
])

# 設置模型的輸入/輸出
feature_extractor = tf.keras.Model(
    inputs = initial_model.inputs,                                  # 設定輸入
    outputs = [layer.output for layer in initial_model.layers],     # 設定輸出為每個層的輸出
)

# 呼叫feature_extractor獲取輸出
x = tf.ones((1, 250, 250, 3))               # 定義一個形狀為1x250x250x3的全1張量作為輸入
features = feature_extractor(x)             # 獲取特徵
print(features)                             # 顯示特徵
print("-" * 20, "Part 9", '-' * 20)

# 設置模型
initial_model = tf.keras.Sequential([
    tf.keras.Input(shape = (250, 250, 3)),                                      # 定義輸入層，形狀為250x250的RGB圖像
    layers.Conv2D(32, 5, strides = 2, activation = "relu"),                     # 添加卷積層，32個過濾器，大小為5x5，步幅為2，激活函數為ReLU
    layers.Conv2D(32, 3, activation = "relu", name = "my_intermediate_layer"),  # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU，並命名
    layers.Conv2D(32, 3, activation = "relu")                                   # 添加卷積層，32個過濾器，大小為3x3，激活函數為ReLU
])

# 設置模型的輸入/輸出
feature_extractor = tf.keras.Model(
    inputs = initial_model.inputs,                                              # 設定輸入
    outputs = initial_model.get_layer(name = "my_intermediate_layer").output,   # 設定輸出為中間層的輸出
)

# 呼叫feature_extractor獲取輸出
x = tf.ones((1, 250, 250, 3))                   # 定義一個形狀為1x250x250x3的全1張量作為輸入
features = feature_extractor(x)                 # 獲取特徵
print(features)                                 # 顯示特徵
print("-" * 20, "Part 10", '-' * 20)
