
# imports
from flask import Flask, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast
import torch

# constants 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device="cpu"
print("got device: {}".format((device)))
ML_TEMPERATURE=0.2
DO_SAMPLE=True
MAX_LENGTH=80
NUM_INFERENCE=10

# chem gen model
DIR_MODEL="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/ChemGenetics100k/Model"
DIR_TOKENIZER="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/ChemGenetics100k/Tokenizer"

# gen model
DIR_ROOT="/Users/mduby/Data/Broad/GPT/Models/FlaskModel/{}"
# DIR_MODEL="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/Genetics60k/Model"
# DIR_TOKENIZER="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/Genetics60k/Tokenizer"
DIR_MODEL=DIR_ROOT.format("Model")
DIR_TOKENIZER=DIR_ROOT.format(("Tokenizer"))

# chem model
# DIR_MODEL="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/Genetics60k/Model"
# DIR_TOKENIZER="/Users/mduby/Data/Broad/GPT/Pubmed/Saved/Genetics60k/Tokenizer"

# global
app = Flask(__name__)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(DIR_TOKENIZER)
# tokenizer = GPT2TokenizerFast.from_pretrained(DIR_TOKENIZER)
print("tokenizer loaded of size: {} and type: {}".format(len(tokenizer), type(tokenizer)))

# load model
model = GPT2LMHeadModel.from_pretrained(DIR_MODEL)

# log
print("model loaded")

# routing
@app.route('/hello')
def hello():
    return "test hello"


@app.route('/test', methods=['GET'])
def test():
    str_input = request.args.get("input")
    return "hello: {}".format(str_input)

@app.route('/inference', methods=['GET'])
def inference():
    str_input = request.args.get("input")
    str_number = request.args.get("times")
    num_inferences = NUM_INFERENCE

    # log
    print("running inferences of num: {} for str: {}".format(num_inferences, str_input))

    # number queries
    if str_number:
        num_inferences = int(str_number)
        
    # query
    str_response = "<html><ul>"

    # get the inference
    list_result = list_infer(str_input, num_inferences=num_inferences)

    # build the response
    for item in list_result:
        str_temp = item.replace("<pad>", "")
        str_response = str_response + "<li>" + str_temp + "</li>"

    str_response = str_response + "</ul></html>"
    # return
    return str_response


def list_infer(str_input, num_inferences=NUM_INFERENCE, log=False):
    # initialize
    list_result = []

    # build the model input
    inp = "<start> " + str_input
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    # output = model.generate(X, attention_mask=a)
    # output = model.generate(X, attention_mask=a, max_length= 60)
    for i in range(num_inferences):
        output = model.generate(X, attention_mask=a, max_length=MAX_LENGTH, pad_token_id=tokenizer.eos_token_id, temperature=ML_TEMPERATURE, do_sample=DO_SAMPLE)

        # untokenize
        result = tokenizer.decode(output[0])
        list_result.append(result)

    # return
    return list_result

