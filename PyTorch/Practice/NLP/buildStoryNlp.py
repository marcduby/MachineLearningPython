# code inspired by https://www.youtube.com/watch?v=mzbJd0NhW2A from Python Simplified channel

# imports
import PyPDF2 as pypdf 
import nltk
from nltk.tokenize import word_tokenize
import string
import torch
from torch import autograd, nn, optim

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



# define the nlp network
class Nlp(nn.Module):
    '''
    use nn.Relu in case switch to activation layer with memory
    '''
    def __init__(self, vocabulary_size, embedding_vector_size):
        super().__init__();
        self.embedding01 = nn.Embedding(vocabulary_size, embedding_vector_size)
        self.linear01 = nn.Linear(embedding_vector_size, 128)
        self.activation01 = nn.ReLU()
        self.linear02 = nn.Linear(128, 512)
        self.activation02 = nn.ReLU()
        self.linear03 = nn.Linear(512, vocabulary_size)
        self.activation03 = nn.log_softmax()


    def forward(self, inputs):
        output = self.embedding01(inputs)
        output = self.activation01(self.linear01(output))
        output = self.activation02(self.linear02(output))
        output = self.activation03(self.linear03(output))

        return output






