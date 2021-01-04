# lazy predict library that will go through all sklearn models

# imports
import pandas as pd
import sklearn 
from sklearn.datasets import load_wine
import lazypredict
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

print("using sklearn version {}".format(sklearn.__version__))
print("using pandas version {}".format(pd.__version__))

# get the data
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

print("got wine data of shape {} and target data of shape {}".format(X.shape, y.shape))

# get the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

print("the train data is of type {} and shape {}".format(type(X_train), X_train.shape))
print("the test data is of type {} and shape {}".format(type(X_test), X_test.shape))

# create the classifier
lazy_classifier = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = lazy_classifier.fit(X_train, X_test, y_train, y_test)

# print what we got
print("got models of type {} and shape {}".format(type(models), models.shape))
print("got predictions of type {} and shape {}".format(type(predictions), predictions.shape))

# print the models
print("the model list is: {}".format(models.head(30)))


