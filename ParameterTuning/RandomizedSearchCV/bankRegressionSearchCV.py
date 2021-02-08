
# imports
import time
import sklearn as sk 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import humanfriendly as hf
import pandas as pd 

print("sklearn version: {}".format(sk.__version__))


def get_scores(model, X_train, y_train, X_test, y_test):
    """ return the model name, train and test scores """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return (model.__name__, train_score, test_score)

def get_cross_cv(model, datam, target, groups = 10):
    """ return the corss validation score """
    return cross_val_score(model, data, target, cv=groups)

def see_elapsed_time(start_time, note):
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("{} {}".format(note, hf.format_timespan(elapsed_time, detailed=True)))


if __name__ == "__main__":
    # read in the data
    df = pd.read_csv("/home/javaprog/Code/PythonWorkspace/MachineLearningPython/Datasets/Books/HandsOnScikitLearnForML/bank.csv")
    df.info()

    # get the features and targets
    X = df.drop(['y'], axis=1)
    y = df['y']
    print("from dataset {} got X: {} and y: {}".format(df.shape, X.shape, y.shape))

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    print("got X train: {} and X test: {}".format(X_train.shape, X_test.shape))

    # buuild the model
    knn  = KNeighborsClassifier()
    print("got KNN: {}".format(knn))

    # train the model
    knn.fit(X_train, y_train)

    # get the scores for the train/test 
    name, train_score, test_score = get_scores(knn, X_train, y_train, X_test, y_test)
    print("for {} got train score: {} and test score: {}".format(name, train_score, test_score))

    

