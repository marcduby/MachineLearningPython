# based on the TF example at https://www.youtube.com/watch?v=Y_hzMnRXjhI&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=3

# use sklearn tools to tokenize and vectorizae the data
# use pandas to read
# use test_train_split 

# imports
import torch 
# from sklearn.preprocessing import test_train_split
import pandas as pd 
import json 

print("got torch version {}".format(torch.__version__))
print("got pandas version {}".format(pd.__version__))


# constants
file_loc = "/home/javaprog/Data/Personal/Kaggle/Sarcasm/Sarcasm_Headlines_Dataset.json"
random_seed = 23
embedding_size=100
num_epochs = 5
learning_rate = 0.001

# import the data
df_data = pd.read_json(file_loc, lines=True)
print("got headlines of length {}".format(df_data.shape))
df_data.info()

# get the features/targets
X = df_data['headlines']
y = df_data['is_sacastic']


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


# create the model
model = Nlp(vocabulary_size=input_size, embedding_vector_size=)

# add the optimizer
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# model.zero_grad()
print("the model is \n{}".format(model.parameters()))

# train the network
for epochs in range(num_epochs):
    # model.train()
    ts_output = model(ts_input)
    print("the output prediction is \n{}".format(ts_output))
    _, ts_pred = ts_output.max(1)
    print("Epoch: {}==================".format(epochs))
    print("the target classes are {}".format((ts_target * output_size).long()))
    print("the prediction classes are {}".format(ts_pred))

    # get the loss
    loss = F.l1_loss(ts_pred, ts_target)
    # loss = F.mse_loss(ts_pred, ts_target)
    # loss = F.nll_loss(ts_output, ts_target)
    # loss = nn.CrossEntropyLoss(ts_pred, ts_target)
    print("the loss is {}".format(loss))

    # zero the gradients and then back propogate the loss
    model.zero_grad()
    loss.backward()
    optimizer.step()




