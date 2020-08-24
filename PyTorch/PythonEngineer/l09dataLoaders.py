
#imports
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("pytorch versino is {}".format(torch.__version__))
print("numpy version is {}".format(np.__version__))

class WhiteWineDataset(Dataset):
    def __init__(self):
        # load the data
        data = np.loadtxt("../../Datasets/Books/HandsOnScikitLearnForML/whitewine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(data[:, :-1])
        self.y = torch.from_numpy(data[:, [-1]])
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WhiteWineDataset()
x, y = dataset[2]

print("the x is {} and the y is {}".format(x, y))

