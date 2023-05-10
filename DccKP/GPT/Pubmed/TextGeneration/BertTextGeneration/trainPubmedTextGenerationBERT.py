

# imports
import os
import transformers
from transformers import BertTokenizer, BertTokenizerFast, BertForNextSentencePrediction,TextDatasetForNextSentencePrediction
import torch
# from torch.nn.functional import soft
from transformers import TextDatasetForNextSentencePrediction
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
# , DataCollatorForNextSentencePrediction

# def train(bert_model, bert_tokenizer, path, eval_path=None):
    

#     train_dataset = TextDatasetForNextSentencePrediction(
#         tokenizer = bert_tokenizer,
#         file_path = path,
#         block_size = 256
#     )
      
#     trainer = Trainer(
#       model=bert_model,
#       args=training_args,
#       data_collator=data_collator,
#       train_dataset=train_dataset,
#       tokenizer=BertTokenizer)
    
#     trainer.train()
#     trainer.save_model(out_dir)

# constants

# ML constants 
ML_MODEL_NAME = "bert-base-cased"
os.environ['WANDB_SILENT']="true"
os.environ['WANDB_MODE']="disabled"

# files
# AWS
FILE_DATA_TRAIN="/home/ubuntu/Data/TextGenerationBert/input_bert.txt"
DIR_MODEL="/home/ubuntu/Models"
DIR_TOKENIZER="/home/ubuntu/Tokenizer"

# LOCAL
FILE_DATA_TRAIN="/home/javaprog/Data/Broad/GPT/TextGenerationBert/Input/input_bert.txt"
DIR_MODEL="/home/javaprog/Data/Broad/GPT/TextGenerationBert/Models"
DIR_TOKENIZER="/home/javaprog/Data/Broad/GPT/TextGenerationBert/Tokenizer"
DIR_TRAIN_OUT="/home/javaprog/Data/Broad/GPT/TextGenerationBert/Output"

# methods
def load_model_tokenizer(name_model, log=False):
    '''
    load the model and tokenizer
    '''
    tokenizer = BertTokenizer.from_pretrained(name_model)
    model = BertForNextSentencePrediction.from_pretrained(name_model)

    # return
    return model, tokenizer

def save_model(model, dir_model, log=False):
    '''
    persists the model
    '''
    print("saving model to: {}".format(dir_model))
    model.save_pretrained(dir_model)


def save_tokenizer(tokenizer, dir_tokenizer, log=False):
    '''
    persists the tokenizer
    '''
    print("saving tokenizer to: {}".format(dir_tokenizer))
    tokenizer.save_pretrained(dir_tokenizer)

def train_model(model, tokenizer, file_input, dir_out, log=False):
    '''
    train the model
    '''
    # out_dir = "/content/drive/My Drive/next_sentence/"
    print("using file: {}".format(file_input))

    training_args = TrainingArguments(output_dir=dir_out,
                                      overwrite_output_dir=True,
                                      num_train_epochs=1,
                                      per_device_train_batch_size=30,
                                      save_steps=100,
                                      save_total_limit=5,
                                      )

    # training_args = TrainingArguments(output_dir=dir_out,
    #                                   overwrite_output_dir=True,
    #                                   num_train_epochs=1,
    #                                   save_steps=100,
    #                                   save_total_limit=5,
    #                                   )

    train_dataset = TextDatasetForNextSentencePrediction(
        tokenizer = tokenizer,
        file_path = file_input,
        block_size = 128
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    # data_collator = DataCollatorForNextSentencePrediction(tokenizer=tokenizer, block_size=256)
      
    trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset,
      tokenizer=tokenizer)
    
    trainer.train()

    # return
    return model



# main
if __name__ == "__main__":
    # load model and tokenizer
    model, tokenizer = load_model_tokenizer(ML_MODEL_NAME)

    # TODO - load phenotypes/genes and add as tokens

    # expand the tokenizer and save
    # tokenizer.add_tokens(['PPARG', 'PCSK9'])

    # train
    model_train = train_model(model, tokenizer, FILE_DATA_TRAIN, DIR_TRAIN_OUT)

    # save the model
    model.save_pretrained(DIR_MODEL)

    # predict



