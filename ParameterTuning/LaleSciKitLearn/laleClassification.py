# test code taken from 
# https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb

# imports
import pandas as pd 
import lale.datasets 
import sklearn 
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
lale.wrap_imported_operators()
from lale.lib.lale import Hyperopt 

print("sklearn version {}".format(sklean.__version__))

# load the data and inspect
(X_train, y_train), (X_test, y_test) = lale.datasets.californoa_housing_df()
pd.concast([X_train.head(), y_train.head()], axis=1)

# create the pipeline to tune
pca_tree_pipeline = Pipeline(steps=[('pca', PCA), ('predict', DecisionTreeRegressor)])

# train
pca_tree_trained = pca_tree_pipeline.auto_configure(X_train, y_train, optimizer=Hyperopt, cv=3, max_evals=10, verbose=True)

# get the metrics
prediction = pca_tree_trained.predict(X_test)
r2_score = sklearn.metrics.r2_score(y_test, prediction)
print("the prediction R2 score is {.2f}".format(r2_score))

# print the tuned pipeline
pca_tree_trained.pretty_print(ipython_display=True)




