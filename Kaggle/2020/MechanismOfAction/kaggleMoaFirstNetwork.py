# imports
import torch
import torch.nn as nn
import pandas as pd 
from torch.utils.data import Dataset

print("the torch version is {}".format(torch.__version__))

# data location
file_features = "/home/javaprog/Data/Personal/Kaggle/MechanismOfAction/train_features.csv"
file_labels = "/home/javaprog/Data/Personal/Kaggle/MechanismOfAction/train_targets_scored.csv"

# create the dataset class
class MoaDataset(Dataset):
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

# instantiate and test
moa_dataset = MoaDataset(file_features, file_labels)
moa_X, moa_y = moa_dataset[10]

# print("the features and shape {} and labels had shape {} with length {}".format(moa_X.shape, moa_y.shape, moa_y.shape[0]))
# print("the features are {}".format(moa_X))
# print("the labels are {}".format(moa_y))


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

    def foward(self, x):
        x = self.linear_01_input(x)
        x = self.linear_01_activation(x)
        x = self.linear_02(x)
        x = self.linear_02_activation(x)
        x = self.linear_03(x)
        x = self.linear_03_activation(x)

        return x

# create the new network
moa_model = MoaNetwork(moa_X.shape[0])
print(moa_model)


