# see scikit book page 71 and sklearn examples

# imports
from sklearn.feature_extraction.text import TfidfVectorizer

# build a text list
sentence_list = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]
print(sentence_list)

# build the feature extracter
extractor = TfidfVectorizer()

# transform the data
X = extractor.fit_transform(sentence_list)

# print the data
feature_names = extractor.get_feature_names()
print("got feature names: {}".format(feature_names))
print("got transformed features of type: {} and shape: {} and data: \n{}".format(type(X), X.shape, X))

# unspool the sparse matrix to print contents
X_array = X.toarray()
print("got unspooled features of type: {} and shape: {} and data: \n{}".format(type(X_array), X_array.shape, X_array))

# loop
for row in range(X_array.shape[0]):
    string = ""
    for col in range(X_array.shape[1]):
        if X_array[row, col] > 0:
            string = string + feature_names[col] + " - (" + str(X_array[row, col]) + ") "
    print("{}: {}".format(row, string))


