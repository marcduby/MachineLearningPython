
# imports
import sklearn
import pandas  as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

print("sklearn version {}".format(sklearn.__version__))

# constants
seed = 25
test_size = 0.25 
cv_folds = 20

# load the iris dataset
iris = load_iris()

# split into features/target
X = iris.data 
y = iris.target 

print("X has shape {} and type {}".format(X.shape, type(X)))
print("y has shape {} and type {}".format(y.shape, type(y)))
print()

# fit and score the model w/o kfold CV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
print("X train has shape {} y train has shape {}".format(X_train.shape, y_train.shape))
print("X test has shape {} y test has shape {}".format(X_test.shape, y_test.shape))
print()

# fit the model without CV
rf_model = RandomForestClassifier(random_state=seed)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print("the non CV training with test split of {} has score {}".format(test_size, accuracy * 100))


# fit the model with CV
kfold = KFold(n_splits=cv_folds, random_state=seed, shuffle=True)
rf_model = RandomForestClassifier(random_state=seed)
accuracy = cross_val_score(rf_model, X, y, cv=kfold).mean()
print("the CV training with {} kfolds has score {}".format(cv_folds, accuracy * 100))

