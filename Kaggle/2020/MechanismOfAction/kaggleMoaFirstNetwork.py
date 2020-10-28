# imports
import torch
import torch.nn as nn
import pandas as pd 
from torch.utils import data
import torch.optim as optim 
import datetime

print("the torch version is {}".format(torch.__version__))

# data location
file_features = "/home/javaprog/Data/Personal/Kaggle/MechanismOfAction/train_features.csv"
file_labels = "/home/javaprog/Data/Personal/Kaggle/MechanismOfAction/train_targets_scored.csv"
batch_size = 32
larning_rate = 1e-2
number_epochs = 30

# create the dataset class
class MoaDataset(data.Dataset):
    def __init__(self, feature_location, label_location):
        # load features, one hot 2 columns, drop id, convert to tensor
        self.features = pd.read_csv(feature_location)
        self.X = self.features.drop(columns=['sig_id', 'cp_type', 'cp_dose'])
        self.X = torch.tensor(self.X.values)

        # load the labels, drop the id, convert to tensor
        self.labels = pd.read_csv(label_location)
        self.y = torch.tensor(self.labels.drop(columns=['sig_id']).values)

        # get the number of rows
        self.n_samples = len(self.features)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples



# create the network class
class MoaNetwork(nn.Module):
    def __init__(self, input_num):
        super().__init__()
        self.linear_01_input = nn.Linear(input_num, 1024)
        self.linear_01_activation = nn.Tanh()
        self.linear_02 = nn.Linear(1024, 512)
        self.linear_02_activation = nn.Tanh()
        self.linear_03 = nn.Linear(512, 206)
        self.linear_03_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_01_input(x)
        x = self.linear_01_activation(x)
        x = self.linear_02(x)
        x = self.linear_02_activation(x)
        x = self.linear_03(x)
        x = self.linear_03_activation(x)

        return x

# instantiate and test
moa_dataset = MoaDataset(file_features, file_labels)
moa_X, moa_y = moa_dataset[10]

print("the features item has shape {} and labels item has shape {} with length {}".format(moa_X.shape, moa_y.shape, moa_y.shape[0]))
# print("the features are {}".format(moa_X))
# print("the labels are {}".format(moa_y))

# create the new network
moa_model = MoaNetwork(moa_X.shape[0])
print(moa_model)

# create the training loader
train_num = int(moa_dataset.n_samples * 0.8)
val_num = int(moa_dataset.n_samples * 0.2) + 1
print("splitting data in train count {} and val count {} for total dataset count {}".format(train_num, val_num, moa_dataset.n_samples))
train_val_split = [train_num, val_num]
train_set, val_set = data.random_split(moa_dataset, train_val_split)
print("the training set has length {} and the validation set {}".format(len(train_set), len(val_set)))

# build the data loaders
train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

# build the optimizer and loss
optimizer = optim.SGD(moa_model.parameters(), lr=larning_rate)
loss_function = nn.BCELoss()

# train
for epoch in range(1, number_epochs):
    loss_train = 0.0
    for features, labels in train_loader:
        # print(features)
        outputs = moa_model(features.float())
        # print(outputs)
        # print("outputs have shape {} and labels have shape {}".format(outputs.shape, labels.shape))
        loss = loss_function(outputs, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train = loss_train + loss.item()
    
    if epoch == 1 or epoch % 5 == 0:
        now_date = datetime.datetime.now()
        training_loss = loss_train / len(train_loader)
        print("{} Epoch {}, training loss {}".format(now_date, epoch, training_loss))

    print(outputs)