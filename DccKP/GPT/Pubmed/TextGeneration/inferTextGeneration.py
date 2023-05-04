
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
device="cpu"
print("Have ML device: {}".format((device)))

# constants 
FILE_MODEL = "/home/javaprog/Data/Broad/GPT/Models/20230501textGeneration39k/text_gen_model_state_60.pt"
DIR_DATA_TRAIN="/home/ubuntu/Data/TextGeneration"
DIR_MODEL="/home/ubuntu/Models/text_gen_40"
DIR_TOKENIZER="/home/ubuntu/Tokenizer"


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

def test_inference(str_input, log=False):
    '''
    test the inference
    '''
    print("input: {}".format(str_input))
    str_output = infer(str_input, model, tokenizer)
    str_output = str_output.replace(" <pad>", "")
    str_output = str_output.replace(" <end>", "")
    str_output = str_output.replace(" <start>", "")
    print("output: {}\n".format(str_output))


if __name__ == "__main__":
    # load the model
    model = GPT2LMHeadModel.from_pretrained(DIR_MODEL)
    # model = torch.load(FILE_MODEL,map_location=torch.device('cpu'))
    model = model.to(device)

    # load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(DIR_TOKENIZER)
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                    "bos_token": "<start>",
                                    "eos_token": "<end>"})

    # do inference
    test_inference("ACE2 is a gene")
    test_inference("BMI is a phenotype")
    test_inference("cystic fobrosis is a disease")
    test_inference("dili is a disease")
    test_inference("diabetes is treated by")
    test_inference("atrial fibillation is a disease")



