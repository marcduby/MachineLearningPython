
# imports
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 

print("got tf version {}".format(tf.__version__))
print("got numpy version {}".format(np.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# constants
list_size = 20
number_epochs = 5

# create data
X = [x for x in range(list_size)]
y = [2 * x - 1 for x in range(list_size)]
print("got features {}".format(X))
print("got target {}".format(y))

# create numpy features
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)
print("got features shape {}".format(X.shape))

# build the network
layer0 = Dense(units=1, input_shape=[1], activation=tf.nn.relu)
model = Sequential([layer0])
model.compile(optimizer='sgd', loss='mean_squared_error')

# fit the model
model.fit(X, y, epochs=number_epochs)

# print the weights
print("got the dense layer weights of: {}".format(layer0.get_weights()))