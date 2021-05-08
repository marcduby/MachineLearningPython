
# imports
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist 

# hyperparameters
batch_size = 64
weight_decay = 0.001
learning_rate = 0.001
num_epochs = 5

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("got train data {} and labels {}".format(X_train.shape, y_train.shape))

# flatten the inputs
X_train = X_train.astype('float32') / 255.0        # the -1 says to leave that colmns as is
X_test = X_test.astype('float32') / 255.0

# build the model
inputs = keras.Input(shape = (28, 28, 1))           # 64x64 with 1 grayscale channel
x = layers.Conv2D(
    filters = 32,
    kernel_size = 3,
    padding = 'same', 
    kernel_regularizer=regularizers.l2(weight_decay),
)(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(
    filters = 64,
    kernel_size = 3,
    padding = 'same', 
    kernel_regularizer=regularizers.l2(weight_decay),
)(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

# train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2)
model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)


