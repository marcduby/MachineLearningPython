
# imports
import pandas as pd 
import sklearn 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import time
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from category_encoders import BinaryEncoder 

import sys
dir_code = "/Users/mduby/Code/WorkspacePython/"
sys.path.insert(0, dir_code + 'MachineLearningPython/Libraries')
from preprocessLib import resample_dataset, pseudo_sample_fit


print("got pandas version {}".format(pd.__version__))
print("got sklearn version {}".format(sklearn.__version__))

def get_accuracy_scores(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, y_pred)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print("for model {} got train accuracy score {} and test accuracy score {}".format(model.__class__.__name__, train_score, test_score))
    return test_score

def get_f1_score(model, X_test, y_test):
    ''' method to print the model f1 score '''
    y_test_pred = model.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='micro')
    print("for model: {} got f1 score {}".format(model.__class__.__name__, score))
    return score

def get_model_score(model, X_test, y_test):
    model_score = model.score(X_test, y_test)
    print("for model {} got score: {}".format(model.__class__.__name__, model_score))
    return model_score

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)


# constants
timestr = time.strftime("%Y%m%d-%H%M%S")
home_dir = "/home/javaprog/Data/Personal"
home_dir = "/Users/mduby/Data"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
test_file = home_dir + "/Kaggle/202103tabularPlayground/test.csv"
submission_file = home_dir + "/Kaggle/202103tabularPlayground/Submissions/" + timestr + "-{}-{}-submit.csv"
random_state = 23

# set of classiffiers
model_set = [
    # "knn": KNeighborsClassifier(2),
    # "QDA": QuadraticDiscriminantAnalysis(), 
    # "ncentroid": NearestCentroid(),
    # "passiveAgressive": PassiveAggressiveClassifier(),
    # ("perceptron", Perceptron()),
    # ("logistic", LogisticRegression()),
    # ("nn", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 50))),
    # ("sgd", SGDClassifier(alpha=0.0001, penalty='elasticnet')),
    # ("ridge", RidgeClassifier()),
    ("random", RandomForestClassifier()),
    ("bernoulli", BernoulliNB(alpha=0.1))
]

# read the data
df_train = pd.read_csv(train_file)
df_train.info()
df_test = pd.read_csv(test_file)

# resample the training dataset
df_train = resample_dataset(df_train, up=False)

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

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
print("got train features {} with target {}".format(X_train.shape, y_train.shape))
print("got test features {} with target {}".format(X_test.shape, y_test.shape))

# use voting classifier
model = VotingClassifier(model_set, voting='soft', verbose=True)
print("\nfitting model: voting soft - {}".format(model.__class__.__name__))
pipeline = Pipeline([
    ('binary_enc', BinaryEncoder(cols=categorical)),
    ('model', model)
])
# pipeline.fit(X_train, y_train)

# pseudo label train the model
model = pseudo_sample_fit(pipeline, X, y, X_submit)

# get the scores
# f1_score = get_f1_score(model, X_test, y_test)
model_score = get_model_score(pipeline, X_test, y_test)
# cv_score = get_cross(model, X, y)
# print("got CV score {}".format(cv_score))

# fit the data on all data
pipeline.fit(X, y)

# read in the test set and predict
print("predict test file {}".format(test_file))
submit_series = df_test['id']
y_submit = pipeline.predict(X_submit)
submit_df = pd.concat([submit_series, pd.Series(y_submit, name='target')], axis=1)
print("results dataframe type {} \n{}".format(type(submit_df), submit_df.head(5)))

# write out submission file
submission_file = submission_file.format(model.__class__.__name__, str(model_score))
submit_df.to_csv(submission_file, index=False)
print("write submission file {}".format(submission_file))


