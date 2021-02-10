# reference page 71 of sklearn book

# imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score

def get_f1_score(model, X_test, y_test):
    ''' method to print the model f1 score '''
    y_test_pred = model.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='micro')
    print("for model: {} got f1 score {}".format(model.__class__.__name__, score))

def get_confusion_matrix(y, y_pred):
    ''' returns the confusion matrix '''
    return confusion_matrix(y, y_pred)


if __name__ == "__main__":
    # get the data
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    print("the train targets of size {} are: {}".format(len(train.target_names), train.target_names))
    print("got train data of shape {} and test data of shape {}\n".format(train.target.shape, test.target.shape))

    # subset the targets 
    target_subset = ['misc.forsale', 'comp.graphics', 'sci.space', 'rec.autos', 'sci.med', 'talk.politics.guns']

    # reload the data with the subsets
    # train = fetch_20newsgroups(subset='train', categories=target_subset)
    # test = fetch_20newsgroups(subset='test', categories=target_subset)
    # print("the train targets of size {} are: {}".format(len(train.target_names), train.target_names))
    # print("got train rows of shape {} and test rows of shape {}".format(train.target.shape, test.target.shape))

    # get the train data
    X_train = train.data
    y_train = train.target
    X_test = test.data
    y_test = test.target

    # vectorize the data
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)

    # fit the model
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)

    # test the model
    X_test_vect = vectorizer.transform(X_test)
    get_f1_score(model, X_test_vect, y_test)

    # get the confusion matrix
    