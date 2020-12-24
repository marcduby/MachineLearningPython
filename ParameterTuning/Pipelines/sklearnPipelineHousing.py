
# imports
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sklearn
import pandas as pd 

print("sklearn version is {}".format(sklearn.__version__))

# get the data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# create a pipeline
legos = [('reduce_dim', PCA()), ('estimator', BayesianRidge())]
pipeline = Pipeline(legos)

# display sample information
print("X info: \n{}\n".format(X.head()))
print("y info: \n{}\n".format(y.head()))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("training features {} and labels {}".format(X_train.shape, y_train.shape))
print("test features {} and labels {}".format(X_test.shape, y_test.shape))

# loop through PCA dimensions
for dim in range(1, X.shape[1]+1):
    legos = [('reduce_dim', PCA(n_components=dim)), ('estimator', BayesianRidge())]
    pipeline = Pipeline(legos)

    # fit the model
    pipeline.fit(X_train, y_train)

    # score the     
    r2 = pipeline.score(X_test, y_test)
    print("for PCA dim {} the R2 score is {:.2f}".format(dim, r2))

# loop through PCA dimensions
for dim in range(1, X.shape[1]+1):
    legos = [('reduce_dim', PCA(n_components=dim)), 
        ('scaler', StandardScaler()),
        ('estimator', BayesianRidge())]
    pipeline = Pipeline(legos)

    # fit the model
    pipeline.fit(X_train, y_train)

    # score the     
    r2 = pipeline.score(X_test, y_test)
    # print("\npipeline: {}".format(pipeline))
    print("with scaler for PCA dim {} the R2 score is {:.2f}".format(dim, r2))

