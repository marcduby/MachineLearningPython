

# imports
import spacy
from collections import Counter 
from spacy import displacy 

# load
nlp = spacy.load("mobydick.txt")

with open(file_name, 'r') as file:
    doc = nlp(file.read)

print("type of doc: {}".format(type(doc)))


# create tokens
tokens = [token.text for token in doc]

# remove stop words (and.or, etc)
words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
print("unfiltered word list: {}".format(len(words)))

words = [word for word in words if word != "\n' and word != '\n\n"]
print("filtered word list: {}".format(len(words)))

# word freq
word_freq = Counter(words)


# load

# get entities 
