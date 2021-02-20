# refenrence sklearn book page 56

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import humanfriendly as hf
import time 
import pandas as pd

def get_scores(model, Xtrain, Xtest, ytrain, ytest, scoring):
    ''' returns model name with scoring metrics '''
    start = time.perf_counter()
    ytest_pred = model.predict(Xtest)
    score_train = model.score(Xtrain, ytrain)
    score_test = model.score(Xtest, ytest)
    score_f1 = f1_score(ytest, ytest_pred, average=scoring)
    end = time.perf_counter()
    print("in {} for model {} got train score {}, test score {} and f1 score {}".format(hf.format_timespan(end-start, detailed=True), model.__class__.__name__, score_train, score_test, score_f1))

if __name__ == "__main__":
    # load the bank data
    data = pd.read_csv("../../../../Datasets/Books/HandsOnScikitLearnForML/bank.csv")
    data.info()
    print("first 10 rows \n{}".format(data.head(10)))

    # get list of object column types
    ctypes = data.columns.to_series().groupby(data.dtypes).groups
    print("got column types of type {} and data {}\n".format(type(ctypes), ctypes))

    # split into X and y
    y = data.y.values
    X_df = data.loc[:, data.columns != 'y']

    # preprocess the features
    object_col_list = list(X_df.select_dtypes(include=['object']).columns)
    print("got column object types of type {} and data {}\n".format(type(object_col_list), object_col_list))
    X_processed_df = pd.get_dummies(X_df, columns=object_col_list)
    # X_processed_df.info()
    X = X_processed_df.values

    # print the data shape
    print("got features of type {} and shape {}".format(type(X), X.shape))
    print("got targets of type {} and shape {}".format(type(y), y.shape))

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)
    print("got train features {} and target {}".format(X_train.shape, y_train.shape))
    print("got test features {} and target {}".format(X_test.shape, y_test.shape))

    # create the model 
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # get the scores
    get_scores(model, X_train, X_test, y_train, y_test, 'micro')



