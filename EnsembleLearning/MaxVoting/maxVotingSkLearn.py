
# imports
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("the version of sklearn is {}".format(sklearn.__version__))
print("the version of numpy is {}".format(np.__version__))


# load the data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 11)
print("Got train X of shape {} and y of shape {}".format(X_train.shape, y_train.shape))
print("Got test X of shape {} and y of shape {}".format(X_test.shape, y_test.shape))

# create a nearest neighbor classifier
neighbor_search = np.arange(1, 25)
print("the neighbor search is {}".format(neighbor_search))
nn_params_search = {'n_neighbors': neighbor_search}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, nn_params_search, verbose = 1, scoring = 'accuracy')
knn_grid.fit(X_train, y_train)
# print("the grid search scores are {}".format(knn_grid.grid_scores_))
knn_best = knn_grid.best_estimator_
print("the best knn model is {}".format(knn_best.get_params()))
knn_score = knn_best.score(X_test, y_test)
print("the score for the knn model is {}\n".format(knn_score))

# try better grid search
knn_grid_weights = ['uniform', 'distance']
knn_grid_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
nn_params_search2 = {'n_neighbors': neighbor_search, 'weights': knn_grid_weights, 'algorithm': knn_grid_algorithm}
knn2 = KNeighborsClassifier()
knn_grid2 = GridSearchCV(knn, nn_params_search2, verbose = 1, cv=5, scoring = 'accuracy')
knn_grid2.fit(X_train, y_train)
# print("the grid search scores are {}".format(knn_grid.grid_scores_))
knn_best2 = knn_grid2.best_estimator_
print("the new best knn model is {}".format(knn_best2.get_params()))
knn_score2 = knn_best2.score(X_test, y_test)
print("the score for the new knn model is {}\n".format(knn_score2))

# random forest classifier
rf = RandomForestClassifier(random_state = 3)
rf_params_search = {'n_estimators': [50, 100, 150, 200]}
rf_grid = GridSearchCV(rf, rf_params_search, cv = 5)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print("the new best random forest model is {}".format(rf_best.get_params()))
rf_score = rf_best.score(X_test, y_test)
print("the score for the random forest model is {}".format(rf_score))





