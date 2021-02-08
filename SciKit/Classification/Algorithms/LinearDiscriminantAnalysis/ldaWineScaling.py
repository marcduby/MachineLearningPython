# reference page 157 of sklearn book

# imports
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

def evaluate_model(X, y, random, scaler=False):
    ''' method to evaluate a new LDA model '''
    # scale the data if requested
    if scaler:
        print("scaling the data with scaler {}".format(scaler))
        X = scaler.fit_transform(X)
    else:
        print("not scaling the data")

    # splt the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random)

    # create the model
    lda_model = LinearDiscriminantAnalysis()

    # fit the model
    lda_model.fit(X_train, y_train)

    # predict test
    y_test_pred = lda_model.predict(X_test)

    # evaluate
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print("got accuracy for LDA with scaler {} of {}".format(scaler, accuracy))

    # cross validate and publish
    cross_validation = cross_val_score(lda_model, X, y, cv=20)
    cross_mean = np.mean(cross_validation)
    print("got cross validation score for LDA with scaler {} of {}\n".format(scaler, cross_mean))

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


