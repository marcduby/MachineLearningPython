

# imports
import tensorflow as tf 

print("got TF version {}".format(tf.__version__))

# constants
number_epochs = 50
callback_level = 0.93

# add callback class
class TfCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > callback_level):
            print("\nreached callback level {}, ending training".format(callback_level))
            self.model.stop_training=True
# load the data
data = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = data.load_data()
print("got train features of shape {} and labels of shape {}".format(X_train.shape, y_train.shape))
print("got test features of shape {} and labels of shape {}".format(X_test.shape, y_test.shape))

# normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# build the model
lin = tf.keras.layers.Flatten(input_shape=(28, 28))
l1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
lout = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.models.Sequential([lin, l1, lout])

# add in the optimizer/loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=number_epochs, callbacks=[TfCallback()])

# evaluate the model
print("testing model")
model.evaluate(X_test, y_test)

