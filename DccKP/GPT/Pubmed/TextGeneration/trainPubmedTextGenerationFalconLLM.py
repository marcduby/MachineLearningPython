

# imports
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import time
import os
import glob 

# constants 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Have ML device: {}".format((device)))

# mac
FILE_DATA_TRAIN="/Users/mduby/Data/Broad/PubmedGPT/Training/ConversationGPT/text_generation_data_train20230430.json"
DIR_MODEL="/Users/mduby/Data/Broad/PubmedGPT/Models/ConversationGPT"

# local
# FILE_DATA_TRAIN="/home/javaprog/Data/Broad/GPT/Data/ConvoPubmedV1/text_generation_data_train20230430.json"
DIR_DATA_TRAIN="/home/javaprog/Data/Broad/GPT/Data/TextGeneration"
DIR_MODEL="/home/javaprog/Data/Broad/GPT/Models"

# AWS
# FILE_DATA_TRAIN="/home/ubuntu/Data/text_generation_data_train20230430.json"
DIR_DATA_TRAIN="/home/ubuntu/Data/TextGeneration"
DIR_MODEL="/home/ubuntu/Models"
DIR_TOKENIZER="/home/ubuntu/Tokenizer"
FILE_KEYWORDS=DIR_DATA_TRAIN + "/text_generation_keywords_train_chem_100k.json"

# ML constants
ML_BATCH_SIZE = 32
# ML_BATCH_SIZE = 64
# ML_BATCH_SIZE = 96
# ML_BATCH_SIZE = 192
ML_MODEL_NAME="tiiuae/falcon-rw-1b"
DIR_MODEL_SAVING="/Text_gen_facon_{}_{}"
ML_MAX_LENGTH_TRAIN=40
# ML_MAX_LENGTH_INFER=60
ML_MAX_LENGTH_TRAIN=40
ML_MAX_LENGTH_INFER=30

# ML_NUM_SIZE_TRAIN=100
ML_NUM_SIZE_TRAIN=-1

# ML_INTERVAL_SAVE_MODEL=5
ML_INTERVAL_SAVE_MODEL=2
ML_NUM_EPOCHS=41

# methods 
def load_training_data(file_path, log=False):
    ''' 
    load the training data 
    ''' 
    # load the data and populate the training dataset 
    json_data = json.load(open(file_path, "r"))
    print("got json training data of size: {}".format(len(json_data)))

    # rerturn 
    return json_data

def save_model(dir_path, epoch, log=False):
    ''' 
    save the model to disk
    ''' 
    pass

def print_elapsed_time(start, num_epoch=0, log=False):
    '''
    prints the elapsed time
    '''
    end = time.time()
    print("epoch: {} took: {}s".format(num_epoch, (end - start)))

def train(chatData, model, optim, num_epochs=25):
    # log
    print("starting training model for epochs: {}".format(num_epochs))

    # loop though epochas to train
    for i in tqdm.tqdm(range(num_epochs)):
        start = time.time()

        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()

        # print time 
        print_elapsed_time(start, num_epoch=i)

        # write out model
        if i % ML_INTERVAL_SAVE_MODEL == 0:
            # file_model = "{}/text_gen_model_state_{}.pt".format(DIR_MODEL, i)
            # torch.save(model.state_dict(), file_model)
            dir_temp = DIR_MODEL + DIR_MODEL_SAVING.format(ML_MODEL_NAME, i)
            os.mkdir(dir_temp)
            model.save_pretrained(dir_temp)
            print("wrote out model for epoch: {} to file: {}".format(i, dir_temp))

        # test the inference
        print_infer("PCSK9 is a gene")
        print_infer("PPARG is a gene")
        print_infer("PPARG and diabetes")
        print_infer("diabetes is a disease")
        print_infer("diabetes is associated with genes")

def print_infer(str_input, log=False):
    '''
    prints the test inference
    '''
    print("\ninput: {}".format(str_input))
    text_result = infer(str_input)
    print("output: {}".format(text_result.replace("<pad>", " ")))

def infer(str_input):
    str_input = "<start> " + str_input
    str_input = tokenizer(str_input, return_tensors="pt")
    X = str_input["input_ids"].to(device)
    a = str_input["attention_mask"].to(device)
    # output = model.generate(X, attention_mask=a)
    # output = model.generate(X, attention_mask=a, max_length= 60)
    output = model.generate(X, attention_mask=a, max_length=ML_MAX_LENGTH_INFER, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output[0])
    return output

def load_tokenizer(model_family, list_keywords=[], log=False):
    print("loading tokenizer")
    # tokenizer = GPT2Tokenizer.from_pretrained(model_family)
    tokenizer = AutoTokenizer.from_pretrained(model_family, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                    "bos_token": "<start>",
                                    "eos_token": "<end>"})

    # add in the pubmed specific keywords
    if len(list_keywords) > 0:
        num_tokenizer = len(tokenizer)
        tokenizer.add_tokens(list_keywords)
        print("added tokens of size: {}. Tokenizer length went from: {} to {} ".format(len(list_keywords), num_tokenizer, len(tokenizer)))

    # return
    return tokenizer


# data loading class
class ChatData(Dataset):
    def __init__(self, list_conversations, tokenizer, size=-1):
        # self.data = json.load(open(path, "r"))

        self.X = []
        for item in list_conversations:
            self.X.append(item)

        # for idx, i in enumerate(self.X):
        #     try:
        #         self.X[idx] = "<startofstring> "+i+" <bot>: "+self.X[idx+1]+" <endofstring>"
        #     except:
        #         break

        if size > 0:
            self.X = self.X[:size]

        # for i, row in enumerate(self.X):
        #     print("{} - {}".format(i, row))

        print("first row of data: {}".format(self.X[0]))
        print("size of training data: {}".format(len(self.X)))

        # self.X_encoded = tokenizer(self.X, max_length=120, truncation=True, padding="max_length", return_tensors="pt")
        self.X_encoded = tokenizer(self.X, max_length=ML_MAX_LENGTH_TRAIN, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

# main
if __name__ == "__main__":
    # get the keyword list
    files = [file for file in glob.glob(DIR_DATA_TRAIN + "/text_generation_keywords*.json")]
    list_keywords = []
    for file_train in files:
        print("download keyword file: {}".format(file_train))
        temp_data = load_training_data(file_train)
        list_keywords = list_keywords + temp_data
    print("got final keyword list of size: {}".format(len(list_keywords)))
    # list_keywords = json.load(open(FILE_KEYWORDS, "r"))

    # get the tokenizer
    tokenizer = load_tokenizer(ML_MODEL_NAME, list_keywords)

    # load the data
    # file_train = FILE_DATA_TRAIN
    json_data = []
    # get the list of input files
    files = [file for file in glob.glob(DIR_DATA_TRAIN + "/text_generation_data*.json")]
    for file_train in files:
        print("download data file: {}".format(file_train))
        temp_data = load_training_data(file_train)
        json_data = json_data + temp_data

    # load tghe data loader
    print("got FINAL training set of size: {}".format(len(json_data)))
    chatData = ChatData(json_data, tokenizer, size=ML_NUM_SIZE_TRAIN)
    # chatData =  DataLoader(chatData, batch_size=32)
    chatData =  DataLoader(chatData, batch_size=ML_BATCH_SIZE)

    # save the tokenizer
    print("saving tokenizer to: {}".format(DIR_TOKENIZER))
    tokenizer.save_pretrained(DIR_TOKENIZER)

    # create the model
    model = AutoModelForCausalLM.from_pretrained(ML_MODEL_NAME, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # train the model
    model.train()
    optim = Adam(model.parameters(), lr=1e-3)
    print("training .... ")
    train(chatData, model, optim, num_epochs=ML_NUM_EPOCHS)


