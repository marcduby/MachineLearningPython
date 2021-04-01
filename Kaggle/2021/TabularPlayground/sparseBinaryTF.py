
# TODO
# - add callback array (https://stackoverflow.com/questions/62630291/callback-claim-that-val-loss-did-not-improve-while-it-clearly-show-that-it-did)
# - add semi supervised
# - add dropout
# - read overfitting (https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)


# imports
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
from category_encoders import BinaryEncoder
from sklearn.metrics import accuracy_score
import numpy as np

print("got pandas version {}".format(pd.__version__))
print("got tensorflow version {}".format(tf.__version__))

import sys
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
sys.path.insert(0, dir_code + 'MachineLearningPython/Libraries')
from preprocessLib import resample_dataset
from tfModelLib import tf_pseudo_sample_fit

# constants
num_epochs = 50
timestr = time.strftime("%Y%m%d-%H%M%S")
home_dir = "/Users/mduby/Data"
home_dir = "/home/javaprog/Data/Personal"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
test_file = home_dir + "/Kaggle/202103tabularPlayground/test.csv"
submission_file = home_dir + "/Kaggle/202103tabularPlayground/Submissions/" + timestr + "-{}-{}-submit.csv"
random_state = 23

# get the model
def get_model():
    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # return
    return model

# read the data
df_train = pd.read_csv(train_file)
df_train.info()
df_test = pd.read_csv(test_file)

# resample the training dataset
df_train = resample_dataset(df_train, up=True)

# get the column names
columns = list(df_train.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        unique = df_train[col].unique()
        print("col {} got {} values".format(col, len(unique)))

# categorical columns
categorical = [cat for cat in columns if 'cat' in cat]
print("categorical columns {}".format(categorical))

# drop the id and target columns
X = df_train.drop(['id', 'target'], axis=1)
y = df_train['target']
X_submit = df_test.drop(['id'], axis=1)
id_series_submit = df_test['id']
print("got train feature shape {}/{} and submit feature shape {}".format(X.shape, y.shape, X_submit.shape))

# binary encode the features
encoder = BinaryEncoder(cols=categorical)
X = encoder.fit_transform(X)
X_submit = encoder.transform(X_submit)
print("after encoding, got train feature shape {}/{} and submit feature shape {}".format(X.shape, y.shape, X_submit.shape))

# convert to numpy arrays
# X = X.to_numpy()
# y = y.to_numpy()
# convert to tensors
# dataset_train = tf.data.Dataset.from_tensor_slices((X.values, y.values))

# fit the model
model = get_model()
# model.fit(dataset_train, epochs=num_epochs)
# model.fit(X, y, epochs=num_epochs)
model = tf_pseudo_sample_fit(get_model, X, y, X_submit, 75, False)

# get the scores
# y_test = model.predict(X)
# print("got prediction \n{}".format(y_test))
# y_test = np.where(y_test > 0.5, 1, 0)
# print("got prediction \n{}".format(y_test))
evaluation = model.evaluate(X, y, batch_size=128)
model_score = evaluation[1]
print("got model score {}".format(model_score))

# read in the test set and predict
print("predict test file {}".format(test_file))
y_submit = model.predict(X_submit)
y_submit = np.where(y_submit > 0.5, 1, 0)
print("got id shape {} and prediction shape {}".format(id_series_submit.shape, y_submit.shape))
y_submit = tf.squeeze(y_submit)
print("got id shape {} and prediction shape {}".format(id_series_submit.shape, y_submit.shape))
submit_df = pd.concat([id_series_submit, pd.Series(y_submit, name='target')], axis=1)
print("results dataframe type {} \n{}".format(type(submit_df), submit_df.head(20)))

# write out submission file
submission_file = submission_file.format(model.__class__.__name__, str(model_score))
submit_df.to_csv(submission_file, index=False)
print("write submission file {}".format(submission_file))
