
# imports
import pandas as pd 
import sklearn 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB          # 81.2 percent
from sklearn.linear_model import RidgeClassifier    # 83.5 percent
from sklearn.linear_model import SGDClassifier      # 76 percent
from sklearn.ensemble import RandomForestClassifier # 99 percent
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time

print("got pandas version {}".format(pd.__version__))
print("got sklearn version {}".format(sklearn.__version__))

def get_accuracy_scores(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, y_pred)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print("for model {} got train accuracy score {} and test accuracy score {}".format(model.__class__.__name__, train_score, test_score))

def get_f1_score(model, X_test, y_test):
    ''' method to print the model f1 score '''
    y_test_pred = model.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='micro')
    print("for model: {} got f1 score {}".format(model.__class__.__name__, score))

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

def encode(encoder, df, column_list):
    for col in column_list:
        if 'cat' in col:
            df[col] = encoder.transform(df[col])

    return df

def fit_then_predict(model, X_train, y_train, X_test):
    ''' will fit on train, then predict test, then fit again on train/test combination '''
    # fit on train
    model.fit(X_train, y_train)

    # predict test
    y_test = model.predict(X_test)

    # combine train and test
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test])

    # train on combination
    model.fit(X_combined, y_combined)

    # return 
    return model


# constants
timestr = time.strftime("%Y%m%d-%H%M%S")
train_file = "/home/javaprog/Data/Personal/Kaggle/202103tabularPlayground/train.csv"
train_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/train.csv"
test_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/test.csv"
submission_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/Submissions/" + timestr + "-{}-submit.csv"
random_state = 23

# read the data
df = pd.read_csv(train_file)
df.info()
test_df = pd.read_csv(test_file)

# get the column names
columns = list(df.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
# for col in columns:
#     if 'cat' in col:
#         unique = df[col].unique()
#         print("col {} got {} values: {}".format(col, len(unique), unique))

# build the label encoder
encoder = LabelEncoder()
encoded_values = set()
for col in columns:
    if 'cat' in col:
        unique = df[col].unique()
        # print("col {} got {} values: {}".format(col, len(unique), unique))
        encoded_values.update(unique)
        unique = test_df[col].unique()
        encoded_values.update(unique)
print("got label unique values {}".format(encoded_values))
encoder.fit(list(encoded_values))

# df = pd.get_dummies(df, prefix=columns)
# df = pd.get_dummies(df, prefix=['cat12'])
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# one hot encode the data
X = encode(encoder, X, columns)

# train the classifier
model = RandomForestClassifier()
model.fit(X, y)

# get the feature importances
feature_importance = model.feature_importances_
importance = sorted(zip(feature_importance, columns), reverse=True)
[print("{} - {}".format(i, row)) for i, row in enumerate(importance)]

