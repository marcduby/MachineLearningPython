# reference page 117 of the scikit book

# imports
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def score_model(model, X_train, y_train, X_test, y_test):
    ''' method to print out performance measures for the model '''
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    print(model.get_params())
    print("model {} with train score {} and test score {}\n".format(model.__class__.__name__, score_train, score_test))

# constants
random_state = 23

if __name__ == "__main__":
    # load the data
    data = load_boston()
    X = data.data
    y = data.target

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    print("got train X of shape {} and y shape of {}".format(X_train.shape, y_train.shape))
    print("got test X of shape {} and y shape of {}".format(X_test.shape, y_test.shape))

    # create and train the model
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    # score the model
    score_model(model, X_train, y_train, X_test, y_test)

    # tune the model
    params = {'n_estimators': [50, 100, 150, 200],
                'criterion': ['mse', 'mae']}
    model = RandomForestRegressor(random_state=random_state)
    search = GridSearchCV(model, param_grid=params, cv=3, verbose=4)
    search.fit(X_train, y_train)
    best_params = search.best_params_

    # create the best model and score
    model = RandomForestRegressor(**best_params, random_state=random_state)
    model.fit(X_train, y_train)

    # score the model
    score_model(model, X_train, y_train, X_test, y_test)
