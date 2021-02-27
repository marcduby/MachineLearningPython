# code inspired by https://www.youtube.com/watch?v=mzbJd0NhW2A from Python Simplified channel

# imports
import PyPDF2 as pypdf 
import nltk
from nltk.tokenize import word_tokenize
import string

# file locations
pdf_file = '/home/javaprog/Data/Personal/PyTorch/NLP/Coraline.pdf'
temp_size = 300

# load the pdf
pdf_handle = open(pdf_file, 'rb')
pdf_data = pypdf.PdfFileReader(pdf_handle)
pdf_number_pages = pdf_data.getNumPages()
pdf_text = ""
for i in range(pdf_number_pages):
    pdf_text += pdf_data.getPage(i).extractText()
pdf_text = pdf_text.lower()
print("got text with {} pages and {} size text".format(pdf_number_pages, len(pdf_text)))
print("first {} of data: {}".format(temp_size, pdf_text[0:temp_size]))

# remove the punctuation
punct_chars = string.punctuation
pdf_text = "".join([char for char in pdf_text if char not in punct_chars])
print("got text with {} pages and {} size text".format(pdf_number_pages, len(pdf_text)))
print("first {} of data: {}".format(temp_size, pdf_text[0:temp_size]))

# get the words
nltk.download("punkt")
pdf_words = word_tokenize(pdf_text)
pdf_word_set = set(pdf_words)
print("got {} words in the document, wit unique count of {}".format(len(pdf_words), len(pdf_word_set)))
print("tokenized words are: {}".format(pdf_words[1:temp_size]))
