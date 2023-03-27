
# imports
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# test
print("test")

# get data
file = open("mobydick.txt")
mobydick = file.read.replace("\n", " " )
file.close()
file = open("hamlet.txt")
hamlet = file.read.replace("\n", " " )
file.close()

# setup dataframe as bag of words
list_labels = ['moby', 'hamlet']
list_corpus = [hamlet, mobydick]

# vectorize all the words per works

# print
df.iloc[:, 5000:5010]

# show correlation matrix in 2d
cosinematrix = np.around(cosine_similarity(df), decimals=2)
df_cosine = pd.DataFram(data=cosinematrix)
df_cosine = df_cosine.rename(index={0: 'hamlet', 1:'moby'})
df_cosine = df_cosine.rename(columns={0: 'hamlet', 1:'moby'})

df_cosine
