

# imports
from torch.utils.data import Dataset
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import time

# constants 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Have ML device: {}".format((device)))

# mac
FILE_DATA_TRAIN="/Users/mduby/Data/Broad/PubmedGPT/Training/ConversationGPT/data_train.json"
DIR_MODEL="/Users/mduby/Data/Broad/PubmedGPT/Models/ConversationGPT"

# AWS
FILE_DATA_TRAIN="/home/ubuntu/Data/data_train20230430.json"
DIR_MODEL="/home/ubuntu/Models"

# local
FILE_DATA_TRAIN="/home/javaprog/Data/Broad/GPT/Data/ConvoPubmedV1/data_train.json"
DIR_MODEL="/home/javaprog/Data/Broad/GPT/Models"

# ML constants
ML_BATCH_SIZE = 64
ML_BATCH_SIZE = 96
ML_MODEL_NAME="gpt2"
ML_MAX_LENGTH_TRAIN=80
ML_MAX_LENGTH_INFER=80
ML_NUM_SIZE_TRAIN=100
ML_NUM_SAVE_MODEL=20
ML_NUM_EPOCHS=100

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
    print("training model for epochs: {}".format(num_epochs))

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
        if i % ML_NUM_SAVE_MODEL == 0:
            file_model = "{}/model_state_{}.pt".format(DIR_MODEL, i)
            torch.save(model.state_dict(), file_model)
            print("wrote out model for epoch: {} to file: {}".format(i, file_model))

        # test the inference
        text_result = infer("\n\nPCSK9 is a gene")
        print(text_result.replace("<pad>", " "))
        text_result = infer("\ndiabetes is a disease")
        print(text_result.replace("<pad>", " "))
        text_result = infer("\ndiabetes is associated with genes")
        print(text_result.replace("<pad>", " "))


def infer(inp):
    inp = "<start> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    # output = model.generate(X, attention_mask=a)
    # output = model.generate(X, attention_mask=a, max_length= 60)
    output = model.generate(X, attention_mask=a, max_length=ML_MAX_LENGTH_INFER, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output[0])
    return output

def load_tokenizer(model_family, log=False):
    tokenizer = GPT2Tokenizer.from_pretrained(model_family)
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                    "bos_token": "<start>",
                                    "eos_token": "<end>"})
    tokenizer.add_tokens(["<bot>:"])

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
    # get the tokenizer
    tokenizer = load_tokenizer(ML_MODEL_NAME)

    # load the data
    file_train = FILE_DATA_TRAIN
    print("download data file: {}".format(file_train))
    json_data = load_training_data(file_train)
    chatData = ChatData(json_data, tokenizer, size=ML_NUM_SIZE_TRAIN)
    # chatData =  DataLoader(chatData, batch_size=32)
    chatData =  DataLoader(chatData, batch_size=ML_BATCH_SIZE)

    # create the model
    model = GPT2LMHeadModel.from_pretrained(ML_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # train the model
    model.train()
    optim = Adam(model.parameters(), lr=1e-3)
    print("training .... ")
    train(chatData, model, optim, num_epochs=ML_NUM_EPOCHS)


