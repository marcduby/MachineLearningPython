

# import
import torch
import torch.nn as nn 

print("the pytorch version is {}".format(torch.__version__))


# define the model
def CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input (1, 28, 28), output (132, 28, 28)
        self.conv2d_01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)    
        self.batchnorm_01 = nn.BatchNorm2d()
        
        # input (32, 28, 28), output (32, 28, 28)
        self.conv2d_02 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # input (32, 28, 28), output (32, 14, 14)
        self.maxpool_02 = nn.MaxPool2d(kernel_size=2)
        self.batchnorm_02 = nn.BatchNorm2d()

        # input (32, 14, 14), output (64, 14, 14)
        self.conv2d_03 = nn.Conv2d(in_channels=32, out_channels=64 kernel_size=3)    
        self.batchnorm_03 = nn.BatchNorm2d()
        
        # input (64, 14, 14), output (64, 14, 14)
        self.conv2d_04 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        # input (64, 14, 14), output (64, 7, 7)
        self.maxpool_04 = nn.MaxPool2d(kernel_size=2)
        self.batchnorm_04 = nn.BatchNorm2d()

        self.flatten_05 = nn.Flatten(start_dim=1)
        self.batchnorm_05 = nn.BatchNorm2d()
        self.linear_05 = nn.Linear(in_features=3136, out_features=512)

        self.batchnorm_06 = nn.BatchNorm2d()
        self.linear_06 = nn.Linear(in_features=512, out_features=10)

        # common
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv2d_01(x))
        x = self.batchnorm_01(x)

        x = self.relu(self.conv2d_02(x))
        x = self.maxpool_02(x)
        x = self.batchnorm_02(x)

        x = self.relu(self.conv2d_03(x))
        x = self.batchnorm_02(x)

        x = self.relu(self.conv2d_04(x))
        x = self.maxpool_04(x)
        x = self.batchnorm_04(x)

        x = self.flatten_05(x)
        x = self.batchnorm_05(x)
        x = self.relu(self.linear_05(x))

        x = self.batchnorm_06(x)
        x = self.linear_06(x)

# load the mnist data

