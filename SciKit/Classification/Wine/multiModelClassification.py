
# imports
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier as gpc 
from sklearn.gaussian_process.kernels import RBF as rbf 
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import RandomForestClassifier as forest, AdaBoostClassifier as ada 
from sklearn.naive_bayes import GaussianNB as gaussian 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as quad, LinearDiscriminantAnalysis as lda 
from sklearn.linear_model import SGDClassifier sgdclassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# page 161
if __name__ = "__main__":
    # load the data
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # scale the data


    






