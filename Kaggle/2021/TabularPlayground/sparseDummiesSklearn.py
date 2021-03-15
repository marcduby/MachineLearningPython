
# imports
import pandas as pd 
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
    return score

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

def one_hot(X_train, X_test):
    ''' combines the train and test DF and one hots the combined df '''
    # add column to dataframes
    X_train['split'] = 'train'
    X_test['split'] = 'test'

    # combine data frames
    X_combined = pd.concat([X_train, X_test], axis=0)

    # get categorical column list
    categorical = [cat for cat in columns if 'cat' in cat]

    # one hot
    X_combined = pd.get_dummies(X_combined, columns=categorical)

    # split data frames
    X_rtrain = X_combined[X_combined['split'] == 'train']
    X_rtest = X_combined[X_combined['split'] == 'test']

    # drop extra column
    X_rtrain = X_rtrain.drop(['split'], axis=1)
    X_rtest = X_rtest.drop(['split'], axis=1)

    # return
    return X_rtrain, X_rtest

# constants
timestr = time.strftime("%Y%m%d-%H%M%S")
home_dir = "/Users/mduby/Data"
home_dir = "/home/javaprog/Data/Personal"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
train_file = home_dir + "/Kaggle/202103tabularPlayground/train.csv"
test_file = home_dir + "/Kaggle/202103tabularPlayground/test.csv"
submission_file = home_dir + "/Kaggle/202103tabularPlayground/Submissions/" + timestr + "-{}-{}-submit.csv"
random_state = 23

# set of classiffiers
model_set = {
    # "knn": KNeighborsClassifier(2),
    # "QDA": QuadraticDiscriminantAnalysis(), 
    # "ncentroid": NearestCentroid(),
    # "passiveAgressive": PassiveAggressiveClassifier(),
    # "perceptron": Perceptron(),
    "sgd": SGDClassifier(alpha=0.0001, penalty='elasticnet'),
    # "bernoulli": BernoulliNB(alpha=0.1)
}

# read the data
df_train = pd.read_csv(train_file)
df_train.info()
df_test = pd.read_csv(test_file)

# get the column names
columns = list(df_train.columns)
print("got data columns {}".format(columns))

# loop through columns, get unique values
for col in columns:
    if 'cat' in col:
        unique = df_train[col].unique()
        print("col {} got {} values: {}".format(col, len(unique), unique))

# df = pd.get_dummies(df, prefix=columns)
# df = pd.get_dummies(df, prefix=['cat12'])
X = df_train.drop(['target'], axis=1)
y = df_train['target']

# one hot encode the data
X, X_submit = one_hot(X, df_test)
X = X.drop(['id'], axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
print("got train features {} with target {}".format(X_train.shape, y_train.shape))
print("got test features {} with target {}".format(X_test.shape, y_test.shape))

# loop through all models
for name, model in model_set.items():
    # train a bayes classifier
    print("\nfitting model: {} - {}".format(name, model.__class__.__name__))
    model.fit(X_train, y_train)

    # get the scores
    f1_score = get_f1_score(model, X_test, y_test)
    get_accuracy_scores(model, X_train, y_train, X_test, y_test)
    # cv_score = get_cross(model, X, y)
    # print("got CV score {}".format(cv_score))

    # fit the data on all data
    model.fit(X, y)

    # read in the test set and predict
    print("predict test file {}".format(test_file))
    submit_series = X_submit['id']
    X_submit = X_submit.drop(['id'], axis=1)
    y_submit = model.predict(X_submit)
    submit_df = pd.concat([submit_series, pd.Series(y_submit, name='target')], axis=1)
    print("results dataframe type {} \n{}".format(type(submit_df), submit_df.head(5)))

    # write out submission file
    submission_file = submission_file.format(model.__class__.__name__, str(f1_score))
    submit_df.to_csv(submission_file, index=False)
    print("write submission file {}".format(submission_file))
