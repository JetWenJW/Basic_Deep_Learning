import tensorflow as tf
from tensorflow.keras import layers
import matplotlib as plt

# Functional API
# Create First Layer
InputTensor = layers.Input(shape=(100,))

# H1 follow InputTensor behind
H1 = layers.Dense(10, activation='relu')(InputTensor)

# H2 Follow H1 Behind
H2 = layers.Dense(20, activation='relu')(H1)

# Output follow H2 Behind
Output = layers.Dense(1, activation='softmax')(H2)

# Create Model(Must index Input/Output)
model = tf.keras.Model(inputs=InputTensor, outputs=Output)

# Display Summary of the Model
model.summary()

# Setting Variable
num_tags = 12
num_words = 10000
num_departments = 4

# Create First Layer InputTensor
title_input = tf.keras.Input(shape=(None,), name="title")
body_input = tf.keras.Input(shape=(None,), name="body")
tags_input = tf.keras.Input(shape=(num_tags,), name="tags")

# Create Second Layer
title_features = layers.Embedding(num_words, 64)(title_input)
body_features = layers.Embedding(num_words, 64)(body_input)

# Create Third Layer
title_features = layers.LSTM(128)(title_features)
body_features = layers.LSTM(32)(body_features)

# Merge all Neutral layers above
x = layers.concatenate([title_features, body_features, tags_input])

# Create 4th Layer & connect to x
priority_pred = layers.Dense(1, name="priority")(x)
department_pred = layers.Dense(num_departments, name="department")(x)

# Create model(must index Input/Output)
model = tf.keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)

# Plot the Model
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", 
                          show_shapes=True)

model.summary()
