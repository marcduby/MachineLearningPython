# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("got train data {} and labels {}".format(X_train.shape, y_train.shape))

# flatten the inputs
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0        # the -1 says to leave that colmns as is
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
# X_train = X_train.astype('float32') / 255.0        # the -1 says to leave that colmns as is
# X_test = X_test.astype('float32') / 255.0

# create sequential model (easy but not flexible, can only have 1 input and one output)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    # sparse categorical means that labels are an integer for the type
    # for one hot encodings, use categorica; cross entropy
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),      # set logits to true if not output activation 
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

# print the model
# Note: can only summary the model if have input shape or after fit (have sent data to the model)
print("model: \n{}".format(model.summary()))

# fit the model
model.fit(X_train, y_train, batch_size=32,  epochs=4, verbose=2)

# evaluate the model
model.evaluate(X_test, y_test, batch_size=32, verbose=2)

