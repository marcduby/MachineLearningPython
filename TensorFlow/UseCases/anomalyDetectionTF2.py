
# imports
from operator import mod
import pandas as pd 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras import layers


# model
class AnomalyDetectorModel(Model):
    def __init__(self):
        super(AnomalyDetectorModel, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(140, activation='sigmoid'),
            ]
        )

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_model():
    model = AnomalyDetectorModel()
    model.compile(optimizer='adam', loss='mae')

    # return
    return model

# methods
def get_anomaly_model():
    pass

# constants
file_ekg = "/home/javaprog/Data/Personal/TensorFlow/EKG/ecg.csv"

# load the ekg data
df_ekg = pd.read_csv(file_ekg, header=None)
raw_data = df_ekg.values
print(df_ekg.head())

# get the labels
labels = raw_data[:,-1]
features = raw_data[:,0:-1]
print("the features have size {} and labels have shape {}".format(features.shape, labels.shape))

# normalize the feature data
# scaler = preprocessing.StandardScaler()
scaler = preprocessing.MinMaxScaler()
print(features[0,])
features = scaler.fit_transform(features)
print(features[0,])

# divide data into regular and irregular EKGs
regular_data = features[labels == 1]
regular_labels = labels[labels == 1]
irregular_data = features[labels == 1]
print("the regular features have shape {} and irregular EKG features have shape {}".format(regular_data.shape, irregular_data.shape))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(regular_data, regular_labels, test_size=.15, random_state=42)
print("the regular train have shape {} and regular test have shape {}".format(X_train.shape, X_test.shape))

# cast to float32
X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)

# train the model
model = get_model()
history = model.fit(X_train, X_train, epochs=25, batch_size=64, shuffle=True, validation_data = (X_test, X_test))


# TODO - try using 


# 1 - NOTES
# w/o using cast to float32 and using standardscaler and normal ecg data for validation
# Epoch 25/25
# 39/39 [==============================] - 0s 1ms/step - loss: 0.4407 - val_loss: 0.4404

# 2 - NOTES
# switching from StandardScaler (-1 tp 1 with 0 mean) to MinMaxSClaer (0 to 1, proportional) gave betyter training
# Epoch 25/25
# 39/39 [==============================] - 0s 1ms/step - loss: 0.0284 - val_loss: 0.0292


