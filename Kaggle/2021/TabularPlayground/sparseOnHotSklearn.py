
# imports
import pandas as pd 
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import RidgeClassifier
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

def one_hot(X_train, X_test):
    ''' combines the train and test DF and one hots the combined df '''
    # add column to dataframes
    X_train['split'] = 'train'
    X_test['split'] = 'test'

    # combine data frames
    X_combined = pd.concat([X_train, X_test], axis=0)

    # one hot
    
    # split data frames

    # return

# constants
timestr = time.strftime("%Y%m%d-%H%M%S")
train_file = "/home/javaprog/Data/Personal/Kaggle/202103tabularPlayground/train.csv"
train_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/train.csv"
test_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/test.csv"
submission_file = "/Users/mduby/Data/Kaggle/202103tabularPlayground/" + timestr + "-submit.csv"
random_state = 23

# read the data
df = pd.read_csv(train_file)
df.info()

# get the column names
columns = list(df.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        unique = df[col].unique()
        print("col {} got {} values: {}".format(col, len(unique), unique))


# df = pd.get_dummies(df, prefix=columns)
# df = pd.get_dummies(df, prefix=['cat12'])
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# one hot encode the data
encoder = OneHotEncoder()
X = encoder.fit_transform(X)
# X = pd.get_dummies(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
print("got train features {} with target {}".format(X_train.shape, y_train.shape))
print("got test features {} with target {}".format(X_test.shape, y_test.shape))

# train a bayes classifier
model = RidgeClassifier()
model.fit(X_train, y_train)

# get the scores
get_f1_score(model, X_test, y_test)
get_accuracy_scores(model, X_train, y_train, X_test, y_test)
# cv_score = get_cross(model, X, y)
# print("got CV score {}".format(cv_score))

# fit the data on all data

# read in the test set and predict
test_df = pd.read_csv(test_file)
print("read test file {}".format(test_file))
X_submit = test_df.drop(['id'], axis=1)
X_submit = encoder.transform(X_submit)
submit_df = test_df['id']
y_submit = model.predict(X_submit)
submit_df['target'] = y_submit

# write out submission file
submit_df.to_csv(submission_file)
print("write submission file {}".format(submission_file))
