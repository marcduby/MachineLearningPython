
# imports
import torch
from torch import autograd, nn
import torch.nn.functional as F
print("the torch version is {}".format(torch.__version__))

# hyperparameters
torch.manual_seed(123)
batch_size = 5
input_size = 3
hidden_size = 10
output_size = 4
print("the batch size is: {} and the input size is: {}".format(batch_size, input_size))

# test tensor
ts_input = torch.rand(batch_size, input_size)
print("the test tensor is \n{}".format(ts_input))

# define the network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden1(x)
        # x = F.relu(x)
        x = F.tanh(x)
        x = self.hidden2(x)

        # return
        return x

# instantiate the model
model = SimpleNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# model.zero_grad()
print("the model is \n{}".format(model.parameters()))

# run the network
ts_output = model(ts_input)
print("the output prediction is \n{}".format(ts_output))

