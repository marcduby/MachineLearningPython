# imports
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# load the dataset 
# iris_df = load_iris()
iris_df = load_wine()

# get the test/train data
X_train, X_test, y_train, y_test = train_test_split(iris_df.data, iris_df.target, 
                test_size = 0.2, shuffle=True, random_state=9)
print("got train set of shape {} and a test set of {}".format(X_train.shape, X_test.shape))

# build the pipelines
pipeline_lr = Pipeline([('scalar_lr', StandardScaler()),
                        ('pca_scaler', PCA(n_components=2)),
                        ('classifier_lr', LogisticRegression(random_state=9))]) 

# fit the pipelines
pipeline_lr.fit(X_train, y_train)

# test the accuacy
accuracy = pipeline_lr.score(X_test, y_test)
print("The model accuracy is: {}".format(accuracy))

