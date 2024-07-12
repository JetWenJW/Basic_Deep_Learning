import tensorflow as tf


# Step 1. Load Data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Step 2.Data Clean(There is no need to do so)
# Step 3.Feature Engineering(Feature as between 0 ~ 1)
x_train_norm, x_test_norm = x_train / 255, x_test / 255
x_train_norm[0]

# Step 4. Data Split (Train & Validation)
# There is no need to do so

# Step 5. Create Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10)
])

# Step 6. Set Optimizer
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

# Train The Model
history = model.fit(x_train_norm, y_train, epochs = 10, validation_split = 0.2)

# Step 7. Score Model
score = model.evaluate(x_test_norm, y_test, verbose = 0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label ='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')

model.summary()

