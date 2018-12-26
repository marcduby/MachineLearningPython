# from video https://www.youtube.com/watch?v=FtKUj0icUz4&t=960s

# imports
import tensorflow as tf
from tensorflow import keras
import numpy as np

# get the mnist fashion data (70k images)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize pixel values from 0-255 to 0 to 1
# TODO - see how it works w/o normalizing
train_images = train_images / 255.0
test_images = test_images / 255.0


# define network
# 3 sequential layers
# 1st layer takes 28 square value (flattens the matrix into array)
# 2nd layer 128 neuron
#  activation function for each neuron, RELU is if x > 0, return x else return 0 (filters out negative)
# 3rd layers with 10 neurons, outputs array of 10, each a probalility for that label (10 labels)
#  activation softmax
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

# compile NN with defined loss and optimizer
# specific types if problems are best with certain loss function and optimizers
# read papers, or trial and error
# TODO - test other optimizers, loss
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

# train the model
model.fit(train_images, train_labels, epochs = 30)


# test the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)


# for my own 28x28 grayscale image
# predictions = model.predict(my_images)

# predict on the test set
predictions = model.predict(test_images)

# verify for specific image on index
index = 41
# 27 woks, 42 index test wrong 6 vs 3, so try more epochs to see if ti gets it right with > accuracy (only 90% accuracy with 5 epochs)
print(test_labels[index])
print(np.argmax(predictions[index]))