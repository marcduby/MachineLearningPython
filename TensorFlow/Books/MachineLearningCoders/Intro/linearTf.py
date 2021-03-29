
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
number_epochs = 500

# create data
z = range(-3, 6, 2)
X = [float(x) for x in z]
y = [float(2 * x - 1) for x in z]
print("got features {}".format(X))
print("got target {}".format(y))

# create numpy features
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

# debug
# X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
print("got features {}".format(X))
print("got target {}".format(y))
print("got features shape {}".format(X.shape))

# build the network
layer0 = Dense(units=1, input_shape=[1])
model = Sequential([layer0])
model.compile(optimizer='sgd', loss='mean_squared_error')

# fit the model
model.fit(X, y, epochs=number_epochs)

# print the weights
print("got the dense layer weights of: {}".format(layer0.get_weights()))

# predict
X_test = np.array([20.0, 11.0], dtype=float)
y_predict = model.predict(X_test)
print("got predict of type {} and data {}".format(type(y_predict), y_predict))
