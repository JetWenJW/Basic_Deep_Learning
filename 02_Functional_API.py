import tensorflow as tf
from tensorflow.keras import layers
import matplotlib as plt

# 使用 Functional API

# 建立第一層
InputTensor = layers.Input(shape = (100,))            # 定義輸入張量，形狀為100

# H1 緊接在 InputTensor 之後
H1 = layers.Dense(10, activation = 'relu')(InputTensor)     # 定義全連接層，10個神經元，激活函數為ReLU

# H2 緊接在 H1 之後
H2 = layers.Dense(20, activation='relu')(H1)                # 定義全連接層，20個神經元，激活函數為ReLU

# Output 緊接在 H2 之後
Output = layers.Dense(1, activation='softmax')(H2)          # 定義輸出層，1個神經元，激活函數為softmax

# 建立模型（必須標記輸入和輸出）
model = tf.keras.Model(inputs=InputTensor, outputs=Output)  # 創建模型，指定輸入和輸出

# 顯示模型摘要
model.summary()  # 打印模型摘要

# 設定變數
num_tags = 12           # 標籤數量
num_words = 10000       # 單詞數量
num_departments = 4     # 部門數量

# 建立第一層 InputTensor
title_input = tf.keras.Input(shape = (None,), name = "title")       # 定義標題輸入，形狀為不定長度的向量
body_input = tf.keras.Input(shape = (None,), name = "body")         # 定義正文輸入，形狀為不定長度的向量
tags_input = tf.keras.Input(shape = (num_tags,), name = "tags")     # 定義標籤輸入，形狀為標籤數量

# 建立第二層
title_features = layers.Embedding(num_words, 64)(title_input)   # 將標題輸入嵌入到64維向量
body_features = layers.Embedding(num_words, 64)(body_input)     # 將正文輸入嵌入到64維向量

# 建立第三層
title_features = layers.LSTM(128)(title_features)   # 將標題特徵輸入LSTM層，輸出128維向量
body_features = layers.LSTM(32)(body_features)      # 將正文特徵輸入LSTM層，輸出32維向量

# 合併所有上述的神經層
x = layers.concatenate([title_features, body_features, tags_input])  # 將標題特徵、正文特徵和標籤輸入合併

# 建立第四層並連接到 x
priority_pred = layers.Dense(1, name="priority")(x)                     # 定義優先級預測層，輸出1個值
department_pred = layers.Dense(num_departments, name="department")(x)   # 定義部門預測層，輸出部門數量的值

# 建立模型（必須標記輸入和輸出）
model = tf.keras.Model(
    inputs=[title_input, body_input, tags_input],   # 指定輸入
    outputs=[priority_pred, department_pred],       # 指定輸出
)

# 繪製模型結構圖
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", 
                          show_shapes=True)     # 繪製模型並保存為圖像，顯示形狀

# 顯示模型摘要
model.summary()  # 打印模型摘要
