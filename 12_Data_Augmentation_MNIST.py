import tensorflow as tf
 
# Step 1. Load Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create CNN Model
from tensorflow.keras import layers
import numpy as np

input_shape = (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape = input_shape),
    layers.Conv2D(32, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(10, activation = "softmax"),
])

# Set Optimizer
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Data Augmentation
# Set Parameter
batch_size = 1000
epochs = 5

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

datagen.fit(x_train)
history = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), 
    epochs = epochs,
    validation_data = datagen.flow(x_test, y_test, batch_size = batch_size), verbose = 1,
    steps_per_epoch = x_train.shape[0] // batch_size
)

# Score Model
score = model.evaluate(x_test, y_test, verbose = 0)

for i, x in enumerate(score):
    print(f'{model.metrics_names[i]}: {score[i]: .4f}')

# 使用小畫家，繪製 0~9，實際測試看看
from skimage import io
from skimage.transform import resize
import numpy as np

# 讀取影像並轉為單色
uploaded_file = './myDigits/9.png'
image1 = io.imread(uploaded_file, as_gray = True)

# 縮為 (28, 28) 大小的影像
image_resized = resize(image1, (28, 28), anti_aliasing = True)    
X1 = image_resized.reshape(1,28, 28, 1) #/ 255

# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = np.abs(1-X1)

# 預測
predictions = np.argmax(model.predict(X1), axis = -1)
print(predictions)














