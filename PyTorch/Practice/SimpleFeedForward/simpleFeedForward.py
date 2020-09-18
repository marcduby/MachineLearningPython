
# imports
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
print("the torch version is {}".format(torch.__version__))

# hyperparameters
torch.manual_seed(123)
batch_size = 5
input_size = 7
hidden_size = 10
output_size = 4
learning_rate = 0.001
num_epochs=100
print("the batch size is: {} and the input size is: {}".format(batch_size, input_size))

# test tensor
# autograd.Variable was deprected in pytorch 0.4
# ts_input = autograd.Variable(torch.rand(batch_size, input_size))
# ts_target = autograd.Variable(torch.rand(batch_size))
ts_input = torch.rand(batch_size, input_size, requires_grad=True)
ts_target = torch.rand(batch_size, requires_grad=True)
print("the input tensor is \n{}".format(ts_input))
print("the target tensor is {}".format((ts_target * output_size).long()))

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
        # x = F.softmax(x)
        # x = F.log_softmax(x)
        x = F.sigmoid(x)

        # return
        return x

# create the model
model = SimpleNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

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

