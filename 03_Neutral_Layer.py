import tensorflow as tf
from tensorflow.keras import layers

# Create Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu', name = "layer1"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax', name = "layer2")
])

# Set Optimizer
model.compile(optimizer = 'adam', 
              loss = 'sparse_categotical_crossentropy',
              metrics = ['accuracy']
              )

# Display Model Sunmmary
model.summary()


# 第一層 Dense 參數個數計算
# Set model Input/Output
feature_extractor = tf.keras.Model(
    inputs = model.inputs,
    outputs = model.get_layer(name = "layer1").output,
)

# Call feature_extractor to get the Output
x = tf.ones((1, 28, 28))
features = feature_extractor(x)
features.shape

# First Dense
parameter_count = (28 * 28) * features.shape[1] + features.shape[1]
print(f'參數(parameter)個數： {parameter_count}')
print("-" * 20)

# 第二層 Dense 參數個數計算
# Setting Model Input/Output
feature_extractor = tf.keras.Model(
    inputs = model.inputs,
    outputs = model.get_layer(name = "layer2").output,
)

# Call feature_extractor to get the Output
x = tf.ones((1, 28, 28))
features = feature_extractor(x)
features.shape

# First Dense
parameter_count = (128) * features.shape[1] + features.shape[1]
print(f'參數(parameter)個數： {parameter_count}')






