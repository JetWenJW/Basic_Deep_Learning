import tensorflow as tf


# 建立模型

# Create Model Syntax 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# Create Model Syntax 2 (將input_shape拿掉，以model參數設定)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

x = tf.keras.layers.Input(shape = (28, 28))
y = model(x)

# Create Model Syntax 3 (可以直接串連神經層)
layer1 = tf.keras.layers.Dense(2, activation = 'relu', name = "layer1")
layer2 = tf.keras.layers.Dense(3, activation = 'relu', name = "layer2")
layer3 = tf.keras.layers.Dense(4, name = "layer3")

# Call Layers on test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))

# 臨時加減神經層
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# Delete one layer
model.pop()
print(f'神經層數{len(model.layers)}')
model.layers

# Add One Layer
model.add(tf.keras.layers.Dense(10))
print(f'神經層數{len(model.layers)}')
model.layers

# Get the informations of Model & 神經層
layer1 = tf.keras.layers.Dense(2, activation = "relu", name = "layer1", input_shape = (28, 28))
layer2 = tf.keras.layers.Dense(3, activation = "relu", name = "layer2")
layer3 = tf.keras.layers.Dense(4, name = "layer3")

# Create model
model = tf.keras.models.Sequential([
    layer1, 
    layer2, 
    layer3
])

# Read Model Weight
print(f'神經層參數類別總數:{len(model.weights)}')
model.weights
print(f'{layer2.name}: {layer2.weights}')

# Get summary
model.summary()

# Add Neutral Layer & Display model summary at the same time
# Much easier to Debug
from tensorflow.keras import layers
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape = (250, 250, 3))) # 250x250 RGB Images
model.add(layers.Conv2D(32, 5, strides = 2, activation = "relu"))
model.add(layers.Conv2D(32, 3, activation = "relu"))
model.add(layers.MaxPooling2D(3))

# Display Model Summary
model.summary()

model.add(layers.Conv2D(32, 3, activation = "relu"))
model.add(layers.Conv2D(32, 3, activation = "relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation = "relu"))
model.add(layers.Conv2D(32, 3, activation = "relu"))
model.add(layers.MaxPooling2D(2))

model.summary()

model.add(layers.GlobalMaxPooling2D())

model.add(layers.Dense(10))

# Setting of Model
initial_model = tf.keras.Sequential([
    tf.keras.Input(shape = (250, 250, 3)),
    layers.Conv2D(32, 5, strides = 2, activation = "relu"),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.Conv2D(32, 3, activation = "relu"),
])

# Setting Model's Input/Output
feature_extractor = tf.keras.Model(
    inputs = initial_model.inputs,
    outputs = [layer.output for layer in initial_model.layers],    
)

# Call feature_extractor get the output
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
features


# Setting Model
initial_model = tf.keras.Sequential([
    tf.keras.Input(shape = (250, 250, 3)),
    layers.Conv2D(32, 5, strides = 2, activation = "relu"),
    layers.Conv2D(32, 3, activation = "relu", name = "my_intermediate_layer"),
    layers.Conv2D(32, 3, activation = "relu")
])

# Set the Input/Output of Model
feature_extractor = tf.keras.Model(
    inputs = initial_model.inputs,
    outputs = initial_model.get_layer(name = "my_intermediate_layer").output,
)

# Call feature_extractor get the Output
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
features
