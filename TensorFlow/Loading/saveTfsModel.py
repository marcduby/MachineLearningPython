
# imports
import os

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist

class LinearModel(keras.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input_layer = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.input_layer(input_tensor))
        x = self.output_layer(x)
        return x   

# constants
dir_saved = "/home/javaprog/Data/Personal/TensorFlow/Test/"
num_epochs = 10
batch_size = 32

# get the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data
X_train = X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28*28).astype("float32") / 255.0

# create the model
model = LinearModel()
model.load_weights(dir_saved)
print("loading model weights from {}".format(dir_saved))
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(),
    metrics = ['accuracy']
)

# train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2)

# save the model
model.save_weights(dir_saved)
print("model saved at {}".format(dir_saved))

