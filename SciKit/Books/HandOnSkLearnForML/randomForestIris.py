# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# imports
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# %%
# load the iris dataset
iris_df = datasets.load_iris()

print("got iris dataset of type {}".format(type(iris_df)))


# %%
features = list(iris_df)

print("the features are {}".format(features))


# %%
iris_keys = iris_df.keys()

print("the iris keys are {}".format(iris_keys))


# %%
# fit the random tree classifier to tease out the most significant features
X = iris_df.data
y = iris_df.target

# get the shapes
print("the feature shape is {} and the target shape is {}".format(X.shape, y.shape))

feature_names = iris_df.feature_names
target_names = iris_df.target_names

print("the features are {} \nthe targets are {}".format(feature_names, target_names))


# %%
forest_model = RandomForestClassifier(random_state = 2, n_estimators = 75)
print("the model is of type {}".format(forest_model.__class__.__name__))


# fit the model
forest_model.fit(X, y)


# %%
# get the most relevant features
feature_importances = list(forest_model.feature_importances_)
print("the raw feature importances are {} and type {}".format(feature_importances, type(feature_importances)))
print("feature names are {} and type {}".format(feature_names, type(feature_names)))

zipped = zip(feature_importances, feature_names)
print(zipped)
importance = sorted(zipped, reverse= True)

# print
print("for the feature importances are:\n")
for i, row in enumerate(importance):
    print(row)


# %%


