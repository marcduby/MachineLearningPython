# reference page 157 of sklearn book

# imports
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

def evaluate_model(X, y, random, scaler=False):
    ''' method to evaluate a new SGD classifier model '''
    # scale the data if requested
    if scaler:
        print("scaling the data with {}".format(scaler))
        X = scaler.fit_transform(X)
    else:
        print("not scaling the data")

    # splt the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random)

    # create the model
    sgd_model = SGDClassifier(max_iter=100, random_state=random)

    # fit the model
    sgd_model.fit(X_train, y_train)

    # predict test
    y_test_pred = sgd_model.predict(X_test)

    # evaluate
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print("got accuracy for SGD scaler {} of {}".format(scaler, accuracy))

    # cross validate and publish
    cross_validation = cross_val_score(sgd_model, X, y, cv=20)
    cross_mean = np.mean(cross_validation)
    print("got cross validation score for SGD scaler {} of {}\n".format(scaler, cross_mean))

if __name__ == "__main__":
    # constants 
    random = 23

    # load the data
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # fit without scaling
    evaluate_model(X, y, random, None)

    # fit with standard scaling
    evaluate_model(X, y, random, StandardScaler())

    # fit with max abs scaling
    evaluate_model(X, y, random, MaxAbsScaler())

    # fit with min max scaling
    evaluate_model(X, y, random, MinMaxScaler())

    # fit with uniform distrib scaling scaling
    evaluate_model(X, y, random, QuantileTransformer(n_quantiles=100))

    # fit with gaussian distrib scaling scaling
    evaluate_model(X, y, random, PowerTransformer())


