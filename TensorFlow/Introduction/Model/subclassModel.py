
# imports
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist 

# hyperparameters
batch_size = 64
weight_decay = 0.001
learning_rate = 0.001
num_epochs = 3

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("got train data {} and labels {}".format(X_train.shape, y_train.shape))

# flatten the inputs
# X_train = X_train.astype('float32') / 255.0        # the -1 says to leave that colmns as is
# X_test = X_test.astype('float32') / 255.0

# TODO - find way to do reshape in model itself
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0        # the -1 says to leave that colmns as is
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# define cnn block
class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv01 = layers.Conv2D(out_channels, kernel_size, padding='same', activation='relu')
        self.batch01 = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv01(input_tensor)
        x = self.batch01(x, training=training)
        print("shape of tensor is {}".format(x.shape))
        # x = tf.nn.relu(x)
        return x

class ImageModel(keras.Model):
    def __init__(self):
        super(ImageModel, self).__init__()
        # self.input_layer = keras.Input(shape = (28, 28, 1)) 
        self.conv01 = CNNBlock(32)
        self.conv02 = CNNBlock(64)
        self.conv03 = CNNBlock(128)
        self.flatten = layers.Flatten()
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, input_tensor, training=False):
        # x = self.input_layer(input_tensor)
        x = self.conv01(input_tensor)
        x = self.conv02(x)
        # x = self.conv03(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

# compile the model
model = ImageModel()
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy'],
)

# train and evaluate the model
model.fit(X_train, y_train, batch_size = batch_size, epochs=num_epochs, verbose=2)
model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)



