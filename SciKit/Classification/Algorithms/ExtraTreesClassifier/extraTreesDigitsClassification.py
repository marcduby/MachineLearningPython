# reference page 175 of sklearn book

# imports
import numpy as np
import humanfriendly as hf
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_digits


def get_train_test_scores(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("model: {} with train: {} and test: {}".format(model.__class__.__name__, train_score, test_score))

def print_time(note, start):
    end = time.perf_counter()
    print("{} in: {}\n".format(note, hf.format_timespan(end-start, detailed=True)))

# constants
random_state = 23

if __name__ == "__main__":
    # load the data
    # digits_data = load_digits()
    # X = digits_data.data
    # y = digits_data.target
    X = np.load("../../../../Datasets/Books/HandsOnScikitLearnForML/X_mnist.npy")
    y = np.load("../../../../Datasets/Books/HandsOnScikitLearnForML/y_mnist.npy")
    print("got X shape {} and y shape {}".format(X.shape, y.shape))

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, shuffle=True)
    print("got train X shape {} and y shape {}".format(X_train.shape, y_train.shape))

    # get smaller dataset for the tuning
    X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, random_state=random_state, shuffle=True, train_size=0.2)
    print("got tuning X shape {} and y shape {}".format(X_train_small.shape, y_train_small.shape))

    # fit the model
    start = time.perf_counter()
    model = ExtraTreesClassifier()
    model.fit(X_train_small, y_train_small)
    print_time("initial fit", start)

    # print the scoring of the model
    get_train_test_scores(model, X_train, y_train, X_test, y_test)

    # tune the model with grid search
    start = time.perf_counter()
    model = ExtraTreesClassifier()
    params = {'class_weight': ['balanced'], 
                'max_depth': [10,20,30,40],
                'n_estimators': [100, 150, 200]}
    # params = {'class_weight': ['balanced'], 
    #             'max_depth': [10],
    #             'n_estimators': [100]}
    grid_search = GridSearchCV(model, param_grid=params, cv=3, verbose=4)
    grid_search.fit(X_train_small, y_train_small)
    best_params = grid_search.best_params_
    print_time("model grid search", start)

    # recreate the model with the best params; fit on smaller dataset
    start = time.perf_counter()
    model = ExtraTreesClassifier(**best_params, random_state=random_state)
    model.fit(X_train_small, y_train_small)
    print_time("tuned fit on smaller set", start)

    # print the scoring of the model
    get_train_test_scores(model, X_train, y_train, X_test, y_test)

    # recreate the model with the best params; fit on smaller dataset
    start = time.perf_counter()
    model = ExtraTreesClassifier(**best_params, random_state=random_state)
    model.fit(X_train, y_train)
    print_time("tuned fit on bigger set", start)

    # get scores for the newly tuned model
    get_train_test_scores(model, X_train, y_train, X_test, y_test)




