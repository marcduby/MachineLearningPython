
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

# constants 
FILE_MODEL = "/home/javaprog/Data/Broad/GPT/Models/20230501textGeneration39k/text_gen_model_state_60.pt"


# methods
def infer(str_input, model, tokenizer, int_length=40, log=False):
    '''
    do inference and returnt he resulting text
    '''
    result = None

    # do the inference
    inp = "<start> " + str_input
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    # output = model.generate(X, attention_mask=a)
    # output = model.generate(X, attention_mask=a, max_length= 60)
    output = model.generate(X, attention_mask=a, max_length=int_length, pad_token_id=tokenizer.eos_token_id)

    # untokenize
    result = tokenizer.decode(output[0])

    # return
    return result

if __name__ == "__main__":
    # load the model
    # model = GPT2LMHeadModel.from_pretrained(FILE_MODEL)
    model = torch.load(FILE_MODEL,map_location=torch.device('cpu'))
    # model = model.to(device)

    # load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                    "bos_token": "<start>",
                                    "eos_token": "<end>"})

    # do inference
    str_input="ACE2 is a gene"
    print("input: {}".format(str_input))
    str_output = infer(str_input, model, tokenizer)
    print("output: {}\n".format(str_output))